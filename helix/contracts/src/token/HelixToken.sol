// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/// @title HelixToken
/// @notice The native token for the HELIX protocol
/// @dev Used for staking, governance, and rewards distribution.
///      Extends ERC20Votes for snapshot-based governance (prevents flash loan voting).
///      Users must delegate to themselves (or a representative) to activate vote tracking.
contract HelixToken is ERC20, ERC20Burnable, ERC20Permit, ERC20Votes, AccessControl {
    /// @notice Role for minting tokens (assigned to Rewards contract)
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    /// @notice Maximum total supply (100 million tokens)
    uint256 public constant MAX_SUPPLY = 100_000_000 * 1e18;

    /// @notice Initial supply to deployer (20 million tokens)
    uint256 public constant INITIAL_SUPPLY = 20_000_000 * 1e18;

    /// @notice Treasury allocation (30 million tokens)
    uint256 public constant TREASURY_ALLOCATION = 30_000_000 * 1e18;

    /// @notice Treasury address
    address public treasury;

    /// @notice Whether initial distribution has occurred
    bool public initialDistributionComplete;

    /// @notice Events
    event TreasuryUpdated(address indexed oldTreasury, address indexed newTreasury);
    event InitialDistributionComplete(address indexed treasury, uint256 amount);

    constructor(address _treasury)
        ERC20("Helix", "HELIX")
        ERC20Permit("Helix")
    {
        require(_treasury != address(0), "Invalid treasury");

        treasury = _treasury;

        // Grant roles
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);

        // Mint initial supply to deployer
        _mint(msg.sender, INITIAL_SUPPLY);
    }

    /// @notice Complete initial distribution to treasury
    function completeInitialDistribution() external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(!initialDistributionComplete, "Already distributed");

        initialDistributionComplete = true;
        _mint(treasury, TREASURY_ALLOCATION);

        emit InitialDistributionComplete(treasury, TREASURY_ALLOCATION);
    }

    /// @notice Mint new tokens (only for rewards)
    /// @param to Recipient address
    /// @param amount Amount to mint
    function mint(address to, uint256 amount) external onlyRole(MINTER_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }

    /// @notice Update treasury address
    /// @param newTreasury New treasury address
    function setTreasury(address newTreasury) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newTreasury != address(0), "Invalid treasury");
        emit TreasuryUpdated(treasury, newTreasury);
        treasury = newTreasury;
    }

    /// @notice Grant minter role to an address (e.g., Rewards contract)
    /// @param account Address to grant minter role
    function addMinter(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _grantRole(MINTER_ROLE, account);
    }

    /// @notice Revoke minter role from an address
    /// @param account Address to revoke minter role
    function removeMinter(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(MINTER_ROLE, account);
    }

    /// @notice Check if an address has minter role
    function isMinter(address account) external view returns (bool) {
        return hasRole(MINTER_ROLE, account);
    }

    /// @notice Get remaining mintable supply
    function remainingMintableSupply() external view returns (uint256) {
        return MAX_SUPPLY - totalSupply();
    }

    // ============ Required Overrides for ERC20Votes ============

    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Votes)
    {
        super._update(from, to, value);
    }

    function nonces(address owner_)
        public
        view
        override(ERC20Permit, Nonces)
        returns (uint256)
    {
        return super.nonces(owner_);
    }
}
