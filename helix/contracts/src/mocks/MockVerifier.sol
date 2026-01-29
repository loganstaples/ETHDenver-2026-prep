// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

contract MockVerifier is IHelixVerifier {
    bool public shouldPass = true;

    function setShouldPass(bool _shouldPass) external {
        shouldPass = _shouldPass;
    }

    function verifyProof(
        bytes memory /* proof */,
        uint256[] memory /* publicInputs */
    ) external view override returns (bool) {
        return shouldPass;
    }
}
