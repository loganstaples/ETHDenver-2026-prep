// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract Halo2VerifyingKey {
    constructor() {
        assembly {
            mstore(0x0000, 0x03d0e3e088325ed8841e5288d7f9e86dced9bab961e3a89ab289ddb8edd24b77) // vk_digest
            mstore(0x0020, 0x0000000000000000000000000000000000000000000000000000000000000008) // num_instances
            mstore(0x0040, 0x000000000000000000000000000000000000000000000000000000000000000e) // k
            mstore(0x0060, 0x30638ce1a7661b6337a964756aa75257c6bf4778d89789ab819ce60c19b04001) // n_inv
            mstore(0x0080, 0x2337acd19f40bf2b2aa212849e9a0c07d626d9ca335d73a09119dbe6eaab3cac) // omega
            mstore(0x00a0, 0x2f9c1d051b2a29bd1d13a09c1489aec5303c2fb2ac7d853ee7a58fdb65b90d7d) // omega_inv
            mstore(0x00c0, 0x2c34760cd8ba6180d92ad9da798faeb6fbf85f3f63d3dfca54b3b798e4f5f37e) // omega_inv_to_l
            mstore(0x00e0, 0x0000000000000000000000000000000000000000000000000000000000000000) // has_accumulator
            mstore(0x0100, 0x0000000000000000000000000000000000000000000000000000000000000000) // acc_offset
            mstore(0x0120, 0x0000000000000000000000000000000000000000000000000000000000000000) // num_acc_limbs
            mstore(0x0140, 0x0000000000000000000000000000000000000000000000000000000000000000) // num_acc_limb_bits
            mstore(0x0160, 0x0000000000000000000000000000000000000000000000000000000000000001) // g1_x
            mstore(0x0180, 0x0000000000000000000000000000000000000000000000000000000000000002) // g1_y
            mstore(0x01a0, 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2) // g2_x_1
            mstore(0x01c0, 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed) // g2_x_2
            mstore(0x01e0, 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b) // g2_y_1
            mstore(0x0200, 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa) // g2_y_2
            mstore(0x0220, 0x249aa3d9ecb4370d659e39df81fee5c205418e4a57d2361a31b39822354e57cd) // neg_s_g2_x_1
            mstore(0x0240, 0x2dd1945705122e4244b109556359b4d20e0f593009947a1a9c172de37d99e868) // neg_s_g2_x_2
            mstore(0x0260, 0x2bd9683bc957fefbcca82a0159d223e3d1b4cbff3a2851ea9959ea81335b61a6) // neg_s_g2_y_1
            mstore(0x0280, 0x05356fd6a3bfccf6ef9d4ac67c9b8acaefb923c3a6f91f25f89ee476403dfc03) // neg_s_g2_y_2
            mstore(0x02a0, 0x1a16aadfb9edabb8f7c8b27a796fd3542fc99c3431d431dcfbda901f73966838) // fixed_comms[0].x
            mstore(0x02c0, 0x075e57465c09c2b63a634f3e1189a4f7f14ecbc0b9c205d50b837a74cf30f90d) // fixed_comms[0].y
            mstore(0x02e0, 0x2d07f6745d595dddb9d499e16e2f7220fe6fae82eed1eb3087e7e21376e5faff) // fixed_comms[1].x
            mstore(0x0300, 0x0db3a3cf35dc1f028a6c611eb38a964c3b79a56e6b21f2130f23ec8826b88567) // fixed_comms[1].y
            mstore(0x0320, 0x04bd3541d3e7c93801aaec422d1b4f97df5b4d89903a2dbc8e38e03d1172b723) // fixed_comms[2].x
            mstore(0x0340, 0x27bc3fbfab036924f65b75c7bd924e10152ab6f491e15c8238969f417b40e56c) // fixed_comms[2].y
            mstore(0x0360, 0x04bd3541d3e7c93801aaec422d1b4f97df5b4d89903a2dbc8e38e03d1172b723) // fixed_comms[3].x
            mstore(0x0380, 0x27bc3fbfab036924f65b75c7bd924e10152ab6f491e15c8238969f417b40e56c) // fixed_comms[3].y
            mstore(0x03a0, 0x2938ac6966273c9eb30b99913937f241c98fee2febb994910aac1adee8061705) // fixed_comms[4].x
            mstore(0x03c0, 0x2af091ad532ff47c7648fc4ac3c4878b4e23ac31dc3a8048891a47efc5c3d35c) // fixed_comms[4].y
            mstore(0x03e0, 0x05330343375f007d6cec95bf3d75fbf1087928a040b8e7ee790ba884e9f8c14d) // fixed_comms[5].x
            mstore(0x0400, 0x20f0dd58608116d0608b7229292abd7e0e586448afa45233ed48c064188db856) // fixed_comms[5].y
            mstore(0x0420, 0x084eb1316f3fd78640d2aacc94528b70b311a2fb898207241797485f4ba54d71) // fixed_comms[6].x
            mstore(0x0440, 0x2057d24867a0868ca7391af06dac8d6c2038d8a14c2e3180954ee56c7266965b) // fixed_comms[6].y
            mstore(0x0460, 0x2864a683c1675980f5b6b82106c5778afe5eb1ed2235327d80c2f79e89bd2866) // fixed_comms[7].x
            mstore(0x0480, 0x2880f8d891022d3072a98a734ef460d583d02bd2dbe88fc21bc89c6df615e255) // fixed_comms[7].y
            mstore(0x04a0, 0x2badfff862f7cf885e6696e79f485f54cc1ce4091b8628c840bf33b852cee70a) // permutation_comms[0].x
            mstore(0x04c0, 0x1ac1e6995cf85cfebdafd176a3e3733643de44554241afd6e62a21372d9dfa4c) // permutation_comms[0].y
            mstore(0x04e0, 0x14494b0e2b609bf42cb5d97eebd76ab3d09681fcc734f8f0305b52e7338f526d) // permutation_comms[1].x
            mstore(0x0500, 0x0f6ff7b78cf60a0e03504475b395b0513bde188e17a81ff102f57553988e30c5) // permutation_comms[1].y
            mstore(0x0520, 0x173c2168b1f2809f49b8979f1bc87d746ec25869f2085bdb32b3b3a9aaa9c5eb) // permutation_comms[2].x
            mstore(0x0540, 0x002454fafcf986d71bf8ed53d0889a76a66a3bf317b18b0cce649776615bc654) // permutation_comms[2].y
            mstore(0x0560, 0x19c04aafc513b9ec8ce76dc051c253b4a1cccf54e72fbb818dbf0cea93c845ba) // permutation_comms[3].x
            mstore(0x0580, 0x25bd5838b09f6103fe2e22901bc65ed7461e023d93de85bfb82d47d51334101c) // permutation_comms[3].y
            mstore(0x05a0, 0x0d0c375ca07fe11ab21fdef54ded7eafe51f45f4a5ca27a6e91ce9f6f42ab090) // permutation_comms[4].x
            mstore(0x05c0, 0x2e5bbf396339893c6e4e4f5f7543c74f7acf5d4ce4f4a99c58d6db1bbdf58434) // permutation_comms[4].y

            return(0, 0x05e0)
        }
    }
}