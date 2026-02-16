// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract Halo2VerifyingKey {
    constructor() {
        assembly {
            mstore(0x0000, 0x0c2fc964bfc119dfee7a08380d1a335ed728d9d6bcb98e72d31e2b1d4f0dd89e) // vk_digest
            mstore(0x0020, 0x0000000000000000000000000000000000000000000000000000000000000006) // num_instances
            mstore(0x0040, 0x000000000000000000000000000000000000000000000000000000000000000c) // k
            mstore(0x0060, 0x3061482dfa038d0fb5b4c0b226194047a2616509f531d4fa3acdb77496c10001) // n_inv
            mstore(0x0080, 0x2f6122bbf1d35fdaa9953f60087a423238aa810773efee2a251aa6161f2e6ee6) // omega
            mstore(0x00a0, 0x179c2392139def1b24f4e92b4bfba20a0fa885cb6bfc2f2cb92790e00237d0c0) // omega_inv
            mstore(0x00c0, 0x28771071ab1633014eae27cfc16d5ebe08a8fe2fc9e85044e4a45f82c14cd825) // omega_inv_to_l
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
            mstore(0x02a0, 0x091e5a1c35ad36c0f3547d6ce2bb2f675da2d09318ac0bf84244a04abb6d30aa) // fixed_comms[0].x
            mstore(0x02c0, 0x0b7432d1f860a08b278487211262072b02cd6c4027df9dc6f99271cf66e4c200) // fixed_comms[0].y
            mstore(0x02e0, 0x06e974351732f8089ce6fa83004664cfd0efb3d4edea524a6950b806c6ef4bcf) // fixed_comms[1].x
            mstore(0x0300, 0x248c11f212e11b2e233e262ef106bab749662f6af857619bc8ac4b74da1b02d0) // fixed_comms[1].y
            mstore(0x0320, 0x11d1d9c9c36ead4e08dc6cc7cfc5b53e479adc73ae5b5cf93c8eaa64457b8489) // permutation_comms[0].x
            mstore(0x0340, 0x0f1639d47aaee84fc9a3d2c2fc98d0749a550e1863c93203f6181006db40f686) // permutation_comms[0].y
            mstore(0x0360, 0x14494b0e2b609bf42cb5d97eebd76ab3d09681fcc734f8f0305b52e7338f526d) // permutation_comms[1].x
            mstore(0x0380, 0x0f6ff7b78cf60a0e03504475b395b0513bde188e17a81ff102f57553988e30c5) // permutation_comms[1].y
            mstore(0x03a0, 0x173c2168b1f2809f49b8979f1bc87d746ec25869f2085bdb32b3b3a9aaa9c5eb) // permutation_comms[2].x
            mstore(0x03c0, 0x002454fafcf986d71bf8ed53d0889a76a66a3bf317b18b0cce649776615bc654) // permutation_comms[2].y
            mstore(0x03e0, 0x29d94a99c4e44a8075432b5f4d3b827d5fa2c7eb73cfbb3d77550ffbda013037) // permutation_comms[3].x
            mstore(0x0400, 0x1ea9c4f25d613ef8007e0912cc6dca8de103aac0450e07d2cb91c2103a469c0f) // permutation_comms[3].y

            return(0, 0x0420)
        }
    }
}