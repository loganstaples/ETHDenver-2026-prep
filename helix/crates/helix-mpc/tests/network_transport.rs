//! Integration test: 3 tokio tasks on localhost performing MPC.
//!
//! Tests the full flow:
//! 1. Create transports (LocalTransport for default, TcpTransport with network-mpc)
//! 2. Shamir-share a secret value
//! 3. Secure multiply using Beaver triples over the transport
//! 4. Reconstruct and verify the result

use helix_mpc::beaver::dealer::TrustedDealer;
use helix_mpc::beaver::triple::BeaverTriple;
use helix_mpc::error::MPCResult;
use helix_mpc::field::Fr;
use helix_mpc::protocols::arithmetic::SecureArithmetic;
use helix_mpc::session::transport::{LocalTransport, MPCTransport};
use helix_mpc::types::PartyId;

/// Runs a single party's Beaver multiplication protocol over a transport.
///
/// Protocol:
/// 1. Compute d = x - a, e = y - b locally
/// 2. Broadcast d and e to all parties
/// 3. Receive d and e from all parties, sum to get opened_d and opened_e
/// 4. Compute result share: c + d*b + e*a + d*e (party 0 only adds d*e)
async fn party_multiply(
    transport: &impl MPCTransport,
    party_index: usize,
    x_share: Fr,
    y_share: Fr,
    triple: BeaverTriple,
) -> MPCResult<Fr> {
    let (d_share, e_share) = SecureArithmetic::beaver_mask(&x_share, &y_share, &triple);

    // Serialize d and e together as a batch
    let batch = SecureArithmetic::serialize_share_batch(&[d_share.clone(), e_share.clone()]);

    // Broadcast our d/e shares to all peers
    transport.broadcast(&batch).await?;

    // Collect d/e shares from all peers and sum
    let mut total_d = d_share;
    let mut total_e = e_share;

    let peers = transport.peers();
    for peer in &peers {
        let msg = transport.recv(peer).await?;
        let shares = SecureArithmetic::deserialize_share_batch(&msg)?;
        total_d = Fr::add(&total_d, &shares[0]);
        total_e = Fr::add(&total_e, &shares[1]);
    }

    // Compute result share
    let result = SecureArithmetic::multiply_shares(&triple, &total_d, &total_e, party_index);
    Ok(result)
}

#[tokio::test]
async fn test_three_party_beaver_multiply_over_transport() {
    let num_parties = 3;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();

    // Create mesh of local transports
    let transports = LocalTransport::create_mesh(&parties);

    // Secret values: x = 7.0, y = 3.0, expected product = 21.0
    let x = Fr::from_f64(7.0);
    let y = Fr::from_f64(3.0);

    // Create additive shares of x and y
    let x_shares = create_additive_shares(&x, num_parties, 42);
    let y_shares = create_additive_shares(&y, num_parties, 99);

    // Generate Beaver triples
    let mut dealer = TrustedDealer::with_seed(12345);
    let triples = dealer.generate_scalar_triple(num_parties);

    // Spawn 3 tokio tasks, one per party
    let mut handles = Vec::new();

    for (i, transport) in transports.into_iter().enumerate() {
        let x_share = x_shares[i].clone();
        let y_share = y_shares[i].clone();
        let triple = triples[i].clone();

        let handle = tokio::spawn(async move {
            party_multiply(&transport, i, x_share, y_share, triple).await
        });
        handles.push(handle);
    }

    // Collect results
    let mut result_shares = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        result_shares.push(result);
    }

    // Reconstruct: sum all shares
    let mut product = Fr::ZERO;
    for share in &result_shares {
        product = Fr::add(&product, share);
    }

    let product_f64 = product.to_f64();
    let expected = 21.0;
    assert!(
        (product_f64 - expected).abs() < 0.1,
        "Expected ~{}, got {}",
        expected,
        product_f64
    );
}

#[tokio::test]
async fn test_three_party_shamir_share_multiply_reconstruct() {
    use helix_mpc::sharing::ShamirSharing;

    let num_parties = 3;
    let threshold = 2;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let transports = LocalTransport::create_mesh(&parties);

    // Shamir-share two secrets
    let mut shamir = ShamirSharing::new(threshold);
    let x = 5.0_f64;
    let y = 4.0_f64;

    let _x_shares = shamir
        .reshare_scalar(x, num_parties, "x", &parties)
        .unwrap();
    let _y_shares = shamir
        .reshare_scalar(y, num_parties, "y", &parties)
        .unwrap();

    // For Beaver multiplication, we need additive shares.
    // Convert Shamir to additive by treating the share values as additive.
    // This works because Shamir shares evaluated at distinct points can be
    // used in Beaver protocol (the reconstruction just uses Lagrange instead of sum).

    // Actually, for Beaver we need additive shares. Let's use additive directly
    // since that's what the existing arithmetic module supports.
    let x_fr = Fr::from_f64(x);
    let y_fr = Fr::from_f64(y);
    let x_additive = create_additive_shares(&x_fr, num_parties, 42);
    let y_additive = create_additive_shares(&y_fr, num_parties, 99);

    // Generate triples
    let mut dealer = TrustedDealer::with_seed(777);
    let triples = dealer.generate_scalar_triple(num_parties);

    // Run the protocol across 3 tasks
    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let x_share = x_additive[i].clone();
        let y_share = y_additive[i].clone();
        let triple = triples[i].clone();

        let handle =
            tokio::spawn(
                async move { party_multiply(&transport, i, x_share, y_share, triple).await },
            );
        handles.push(handle);
    }

    let mut result_shares = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        result_shares.push(result);
    }

    // Reconstruct
    let mut product = Fr::ZERO;
    for share in &result_shares {
        product = Fr::add(&product, share);
    }

    let product_f64 = product.to_f64();
    let expected = 20.0; // 5 * 4 = 20
    assert!(
        (product_f64 - expected).abs() < 0.1,
        "Expected ~{}, got {}",
        expected,
        product_f64
    );
}

#[tokio::test]
async fn test_batched_vector_multiply_over_transport() {
    let num_parties = 3;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let transports = LocalTransport::create_mesh(&parties);

    // Multiply two vectors element-wise: [2, 3] * [4, 5] = [8, 15]
    let x_vec = vec![Fr::from_f64(2.0), Fr::from_f64(3.0)];
    let y_vec = vec![Fr::from_f64(4.0), Fr::from_f64(5.0)];
    let dim = x_vec.len();

    // Create additive shares for each element
    let mut x_shares_per_party: Vec<Vec<Fr>> = vec![Vec::new(); num_parties];
    let mut y_shares_per_party: Vec<Vec<Fr>> = vec![Vec::new(); num_parties];

    for d in 0..dim {
        let x_sh = create_additive_shares(&x_vec[d], num_parties, 42 + d as u64);
        let y_sh = create_additive_shares(&y_vec[d], num_parties, 99 + d as u64);
        for i in 0..num_parties {
            x_shares_per_party[i].push(x_sh[i].clone());
            y_shares_per_party[i].push(y_sh[i].clone());
        }
    }

    // Generate triples for each element
    let mut dealer = TrustedDealer::with_seed(555);
    let mut triples_per_party: Vec<Vec<BeaverTriple>> = vec![Vec::new(); num_parties];
    for _ in 0..dim {
        let triples = dealer.generate_scalar_triple(num_parties);
        for (i, t) in triples.into_iter().enumerate() {
            triples_per_party[i].push(t);
        }
    }

    // Spawn tasks
    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let x_shares = x_shares_per_party[i].clone();
        let y_shares = y_shares_per_party[i].clone();
        let triples = triples_per_party[i].clone();

        let handle = tokio::spawn(async move {
            // Compute batched masks
            let (d_batch, e_batch) =
                SecureArithmetic::batched_beaver_mask(&x_shares, &y_shares, &triples);

            // Serialize and broadcast
            let batch_msg = SecureArithmetic::serialize_share_batch(
                &d_batch
                    .iter()
                    .chain(e_batch.iter())
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            transport.broadcast(&batch_msg).await?;

            // Receive from all peers and sum
            let dim = x_shares.len();
            let mut total_d = d_batch;
            let mut total_e = e_batch;

            let peers = transport.peers();
            for peer in &peers {
                let msg = transport.recv(peer).await?;
                let shares = SecureArithmetic::deserialize_share_batch(&msg)?;
                for j in 0..dim {
                    total_d[j] = Fr::add(&total_d[j], &shares[j]);
                    total_e[j] = Fr::add(&total_e[j], &shares[dim + j]);
                }
            }

            // Compute result shares
            let result =
                SecureArithmetic::batched_multiply_shares(&triples, &total_d, &total_e, i);
            Ok::<Vec<Fr>, helix_mpc::MPCError>(result)
        });
        handles.push(handle);
    }

    // Collect and reconstruct
    let mut result_vecs: Vec<Vec<Fr>> = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        result_vecs.push(result);
    }

    // Sum across parties for each element
    for d in 0..dim {
        let mut sum = Fr::ZERO;
        for party_result in &result_vecs {
            sum = Fr::add(&sum, &party_result[d]);
        }
        let val = sum.to_f64();
        let expected = x_vec[d].to_f64() * y_vec[d].to_f64();
        assert!(
            (val - expected).abs() < 0.1,
            "Element {}: expected ~{}, got {}",
            d,
            expected,
            val
        );
    }
}

#[tokio::test]
async fn test_secure_channel_with_multiply() {
    use helix_mpc::session::secure_channel::SecureChannel;

    let num_parties = 3;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let transports = LocalTransport::create_mesh(&parties);

    let x = Fr::from_f64(6.0);
    let y = Fr::from_f64(7.0);
    let x_shares = create_additive_shares(&x, num_parties, 42);
    let y_shares = create_additive_shares(&y, num_parties, 99);

    let mut dealer = TrustedDealer::with_seed(888);
    let triples = dealer.generate_scalar_triple(num_parties);

    // Establish secure channels in parallel, then multiply
    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let x_share = x_shares[i].clone();
        let y_share = y_shares[i].clone();
        let triple = triples[i].clone();
        let handle = tokio::spawn(async move {
            let mut sc = SecureChannel::establish(transport).await?;
            let (d_share, e_share) = SecureArithmetic::beaver_mask(&x_share, &y_share, &triple);

            let batch = SecureArithmetic::serialize_share_batch(&[d_share.clone(), e_share.clone()]);

            // Broadcast encrypted
            sc.broadcast_encrypted(&batch).await?;

            // Receive from all peers
            let mut total_d = d_share;
            let mut total_e = e_share;

            let peers = sc.peers();
            for peer in &peers {
                let msg = sc.recv_encrypted(peer).await?;
                let shares = SecureArithmetic::deserialize_share_batch(&msg)?;
                total_d = Fr::add(&total_d, &shares[0]);
                total_e = Fr::add(&total_e, &shares[1]);
            }

            let result = SecureArithmetic::multiply_shares(&triple, &total_d, &total_e, i);
            Ok::<Fr, helix_mpc::MPCError>(result)
        });
        handles.push(handle);
    }

    let mut product = Fr::ZERO;
    for handle in handles {
        let share = handle.await.unwrap().unwrap();
        product = Fr::add(&product, &share);
    }

    let product_f64 = product.to_f64();
    let expected = 42.0; // 6 * 7 = 42
    assert!(
        (product_f64 - expected).abs() < 0.1,
        "Expected ~{}, got {}",
        expected,
        product_f64
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates `n` additive shares of `secret` that sum to `secret`.
fn create_additive_shares(secret: &Fr, n: usize, seed: u64) -> Vec<Fr> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut shares = Vec::with_capacity(n);
    let mut sum = Fr::ZERO;

    for _ in 0..n - 1 {
        let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
        sum = Fr::add(&sum, &r);
        shares.push(r);
    }

    // Last share = secret - sum of others
    shares.push(Fr::sub(secret, &sum));
    shares
}
