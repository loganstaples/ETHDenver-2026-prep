// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixModelStore.sol";

contract HelixModelStoreTest is Test {
    HelixModelStore public store;

    address public alice;
    address public bob;
    address public charlie;

    event ModelCreated(uint256 indexed tokenId, address indexed creator, string slug, string name);
    event VersionAdded(uint256 indexed tokenId, uint256 indexed versionIndex, string semver, string rootHash);
    event ModelPublicityChanged(uint256 indexed tokenId, bool isPublic);
    event InferenceFeeChanged(uint256 indexed tokenId, uint16 feeBps);
    event AccessChanged(uint256 indexed tokenId, address indexed account, bool granted);
    event ModelForSaleChanged(uint256 indexed tokenId, bool forSale);
    event SalePriceChanged(uint256 indexed tokenId, uint256 price);
    event ModelSold(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price);
    event InferencePaid(uint256 indexed tokenId, address indexed payer, uint256 nonce, uint256 amount, uint256 ownerShare);

    function setUp() public {
        store = new HelixModelStore();
        alice = makeAddr("alice");
        bob = makeAddr("bob");
        charlie = makeAddr("charlie");
        vm.deal(alice, 10 ether);
        vm.deal(bob, 10 ether);
    }

    // -------------------------------------------------------
    // createModel
    // -------------------------------------------------------

    function test_createModel_mintsNFTAndStoresMetadata() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("mnist-v1", "MNIST Classifier", "A simple MNIST model");

        assertEq(tokenId, 0);
        assertEq(store.ownerOf(tokenId), alice);

        (
            string memory slug,
            string memory name,
            string memory description,
            address creator,
            uint40 createdAt,
            bool isPublic,
            uint16 inferenceFee
        ) = store.models(tokenId);

        assertEq(slug, "mnist-v1");
        assertEq(name, "MNIST Classifier");
        assertEq(description, "A simple MNIST model");
        assertEq(creator, alice);
        assertEq(createdAt, uint40(block.timestamp));
        assertEq(isPublic, false);
        assertEq(inferenceFee, 0);
    }

    function test_createModel_incrementsTokenIds() public {
        vm.startPrank(alice);
        uint256 id0 = store.createModel("model-a", "Model A", "");
        uint256 id1 = store.createModel("model-b", "Model B", "");
        uint256 id2 = store.createModel("model-c", "Model C", "");
        vm.stopPrank();

        assertEq(id0, 0);
        assertEq(id1, 1);
        assertEq(id2, 2);
    }

    function test_createModel_slugLookup() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("my-slug", "My Model", "desc");

        uint256 found = store.getModelBySlug("my-slug");
        assertEq(found, tokenId);
    }

    function test_createModel_revertsOnDuplicateSlug() public {
        vm.prank(alice);
        store.createModel("unique-slug", "Model 1", "");

        vm.prank(bob);
        vm.expectRevert("Slug already taken");
        store.createModel("unique-slug", "Model 2", "");
    }

    function test_createModel_revertsOnEmptySlug() public {
        vm.prank(alice);
        vm.expectRevert("Slug required");
        store.createModel("", "Model", "");
    }

    function test_createModel_revertsOnEmptyName() public {
        vm.prank(alice);
        vm.expectRevert("Name required");
        store.createModel("slug", "", "");
    }

    function test_createModel_emitsEvent() public {
        vm.prank(alice);
        vm.expectEmit(true, true, false, true);
        emit ModelCreated(0, alice, "test-model", "Test Model");
        store.createModel("test-model", "Test Model", "description");
    }

    function test_createModel_allowsEmptyDescription() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("no-desc", "No Desc Model", "");
        (, , string memory description, , , , ) = store.models(tokenId);
        assertEq(description, "");
    }

    // -------------------------------------------------------
    // addVersion
    // -------------------------------------------------------

    function test_addVersion_ownerCanAddVersion() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("versioned", "Versioned Model", "");

        uint256 vIdx = store.addVersion(
            tokenId,
            "1.0.0",
            "0xabc123rootHash",
            9500,
            "session-001",
            true
        );
        vm.stopPrank();

        assertEq(vIdx, 0);
        assertEq(store.getVersionCount(tokenId), 1);

        HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
        assertEq(v.semver, "1.0.0");
        assertEq(v.rootHash, "0xabc123rootHash");
        assertEq(v.accuracy, 9500);
        assertEq(v.timestamp, uint40(block.timestamp));
        assertEq(v.sessionId, "session-001");
        assertEq(v.weightsStored, true);
    }

    function test_addVersion_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("owned", "Owned Model", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.addVersion(tokenId, "1.0.0", "hash", 9000, "sess", true);
    }

    function test_addVersion_multipleVersions() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("multi-ver", "Multi Version", "");

        store.addVersion(tokenId, "0.1.0", "", 5000, "sess-1", false);
        vm.warp(block.timestamp + 1 hours);
        store.addVersion(tokenId, "0.2.0", "hash-v2", 7500, "sess-2", true);
        vm.warp(block.timestamp + 1 hours);
        store.addVersion(tokenId, "1.0.0", "hash-v3", 9500, "sess-3", true);
        vm.stopPrank();

        assertEq(store.getVersionCount(tokenId), 3);

        HelixModelStore.Version[] memory versions = store.getVersions(tokenId);
        assertEq(versions.length, 3);
        assertEq(versions[0].semver, "0.1.0");
        assertEq(versions[1].semver, "0.2.0");
        assertEq(versions[2].semver, "1.0.0");
        assertEq(versions[0].accuracy, 5000);
        assertEq(versions[2].accuracy, 9500);
    }

    function test_addVersion_withoutWeights() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("no-weights", "No Weights", "");

        store.addVersion(tokenId, "1.0.0", "", 8000, "sess-1", false);
        vm.stopPrank();

        HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
        assertEq(v.rootHash, "");
        assertEq(v.weightsStored, false);
    }

    function test_addVersion_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("event-ver", "Event Model", "");

        vm.expectEmit(true, true, false, true);
        emit VersionAdded(tokenId, 0, "1.0.0", "root-hash");
        store.addVersion(tokenId, "1.0.0", "root-hash", 9500, "sess", true);
        vm.stopPrank();
    }

    function test_getVersion_revertsOnInvalidIndex() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("invalid-idx", "Model", "");

        vm.expectRevert("Invalid version index");
        store.getVersion(tokenId, 0);
    }

    function test_getVersionCount_zeroForNewModel() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("new-model", "New", "");

        assertEq(store.getVersionCount(tokenId), 0);
    }

    // -------------------------------------------------------
    // Access control
    // -------------------------------------------------------

    function test_hasModelAccess_ownerAlwaysHasAccess() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("access-test", "Access Test", "");

        assertTrue(store.hasModelAccess(tokenId, alice));
    }

    function test_hasModelAccess_nonOwnerNoAccessByDefault() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("private", "Private", "");

        assertFalse(store.hasModelAccess(tokenId, bob));
    }

    function test_grantAccess_grantsAccessToNonOwner() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("grant-test", "Grant Test", "");
        store.grantAccess(tokenId, bob);
        vm.stopPrank();

        assertTrue(store.hasModelAccess(tokenId, bob));
    }

    function test_grantAccess_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("grant-event", "Grant Event", "");

        vm.expectEmit(true, true, false, true);
        emit AccessChanged(tokenId, bob, true);
        store.grantAccess(tokenId, bob);
        vm.stopPrank();
    }

    function test_revokeAccess_removesAccess() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("revoke-test", "Revoke Test", "");
        store.grantAccess(tokenId, bob);
        assertTrue(store.hasModelAccess(tokenId, bob));

        store.revokeAccess(tokenId, bob);
        vm.stopPrank();

        assertFalse(store.hasModelAccess(tokenId, bob));
    }

    function test_revokeAccess_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("revoke-event", "Revoke Event", "");
        store.grantAccess(tokenId, bob);

        vm.expectEmit(true, true, false, true);
        emit AccessChanged(tokenId, bob, false);
        store.revokeAccess(tokenId, bob);
        vm.stopPrank();
    }

    function test_grantAccess_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("no-grant", "No Grant", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.grantAccess(tokenId, charlie);
    }

    function test_revokeAccess_nonOwnerReverts() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("no-revoke", "No Revoke", "");
        store.grantAccess(tokenId, bob);
        vm.stopPrank();

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.revokeAccess(tokenId, charlie);
    }

    // -------------------------------------------------------
    // setPublic
    // -------------------------------------------------------

    function test_setPublic_makesModelAccessibleToEveryone() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("public-model", "Public Model", "");

        assertFalse(store.hasModelAccess(tokenId, bob));
        assertFalse(store.hasModelAccess(tokenId, charlie));

        store.setPublic(tokenId, true);
        vm.stopPrank();

        assertTrue(store.hasModelAccess(tokenId, bob));
        assertTrue(store.hasModelAccess(tokenId, charlie));
    }

    function test_setPublic_canBeRevertedToPrivate() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("toggle-public", "Toggle", "");

        store.setPublic(tokenId, true);
        assertTrue(store.hasModelAccess(tokenId, bob));

        store.setPublic(tokenId, false);
        assertFalse(store.hasModelAccess(tokenId, bob));
        vm.stopPrank();
    }

    function test_setPublic_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("public-event", "Public Event", "");

        vm.expectEmit(true, false, false, true);
        emit ModelPublicityChanged(tokenId, true);
        store.setPublic(tokenId, true);
        vm.stopPrank();
    }

    function test_setPublic_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("no-public", "No Public", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.setPublic(tokenId, true);
    }

    // -------------------------------------------------------
    // setInferenceFee
    // -------------------------------------------------------

    function test_setInferenceFee_setsFee() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("fee-model", "Fee Model", "");

        store.setInferenceFee(tokenId, 500); // 5%
        vm.stopPrank();

        (, , , , , , uint16 fee) = store.models(tokenId);
        assertEq(fee, 500);
    }

    function test_setInferenceFee_maxCap() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("max-fee", "Max Fee", "");

        store.setInferenceFee(tokenId, 5000); // 50% - exactly at cap
        vm.stopPrank();

        (, , , , , , uint16 fee) = store.models(tokenId);
        assertEq(fee, 5000);
    }

    function test_setInferenceFee_revertsAboveCap() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("over-fee", "Over Fee", "");

        vm.expectRevert("Fee exceeds 50%");
        store.setInferenceFee(tokenId, 5001);
        vm.stopPrank();
    }

    function test_setInferenceFee_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("fee-event", "Fee Event", "");

        vm.expectEmit(true, false, false, true);
        emit InferenceFeeChanged(tokenId, 1000);
        store.setInferenceFee(tokenId, 1000);
        vm.stopPrank();
    }

    function test_setInferenceFee_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("no-fee", "No Fee", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.setInferenceFee(tokenId, 100);
    }

    function test_setInferenceFee_zeroIsValid() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("zero-fee", "Zero Fee", "");
        store.setInferenceFee(tokenId, 500);
        store.setInferenceFee(tokenId, 0);
        vm.stopPrank();

        (, , , , , , uint16 fee) = store.models(tokenId);
        assertEq(fee, 0);
    }

    // -------------------------------------------------------
    // Transfer: preserves creator, new owner can manage
    // -------------------------------------------------------

    function test_transfer_preservesCreator() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("transfer-model", "Transfer Model", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        assertEq(store.ownerOf(tokenId), bob);

        (, , , address creator, , , ) = store.models(tokenId);
        assertEq(creator, alice); // creator is preserved
    }

    function test_transfer_newOwnerCanAddVersion() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("transfer-ver", "Transfer Version", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        vm.prank(bob);
        uint256 vIdx = store.addVersion(tokenId, "2.0.0", "new-hash", 9800, "sess-new", true);
        assertEq(vIdx, 0);
    }

    function test_transfer_oldOwnerCannotManage() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("old-owner", "Old Owner", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        vm.prank(alice);
        vm.expectRevert("Not model owner");
        store.addVersion(tokenId, "1.0.0", "hash", 9000, "sess", true);
    }

    function test_transfer_newOwnerCanSetPublic() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("transfer-pub", "Transfer Public", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        vm.prank(bob);
        store.setPublic(tokenId, true);

        (, , , , , bool isPublic, ) = store.models(tokenId);
        assertTrue(isPublic);
    }

    function test_transfer_newOwnerCanManageAccess() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("transfer-access", "Transfer Access", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        vm.prank(bob);
        store.grantAccess(tokenId, charlie);
        assertTrue(store.hasModelAccess(tokenId, charlie));

        vm.prank(bob);
        store.revokeAccess(tokenId, charlie);
        assertFalse(store.hasModelAccess(tokenId, charlie));
    }

    function test_transfer_newOwnerCanSetFee() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("transfer-fee", "Transfer Fee", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        vm.prank(bob);
        store.setInferenceFee(tokenId, 2500);

        (, , , , , , uint16 fee) = store.models(tokenId);
        assertEq(fee, 2500);
    }

    // -------------------------------------------------------
    // Enumeration: balanceOf, totalSupply
    // -------------------------------------------------------

    function test_enumeration_totalSupply() public {
        assertEq(store.totalSupply(), 0);

        vm.startPrank(alice);
        store.createModel("enum-1", "Enum 1", "");
        assertEq(store.totalSupply(), 1);

        store.createModel("enum-2", "Enum 2", "");
        assertEq(store.totalSupply(), 2);
        vm.stopPrank();

        vm.prank(bob);
        store.createModel("enum-3", "Enum 3", "");
        assertEq(store.totalSupply(), 3);
    }

    function test_enumeration_balanceOf() public {
        assertEq(store.balanceOf(alice), 0);
        assertEq(store.balanceOf(bob), 0);

        vm.startPrank(alice);
        store.createModel("bal-1", "Bal 1", "");
        store.createModel("bal-2", "Bal 2", "");
        vm.stopPrank();

        vm.prank(bob);
        store.createModel("bal-3", "Bal 3", "");

        assertEq(store.balanceOf(alice), 2);
        assertEq(store.balanceOf(bob), 1);
    }

    function test_enumeration_tokenOfOwnerByIndex() public {
        vm.startPrank(alice);
        store.createModel("tok-a", "Tok A", "");
        store.createModel("tok-b", "Tok B", "");
        vm.stopPrank();

        assertEq(store.tokenOfOwnerByIndex(alice, 0), 0);
        assertEq(store.tokenOfOwnerByIndex(alice, 1), 1);
    }

    function test_enumeration_tokenByIndex() public {
        vm.prank(alice);
        store.createModel("global-1", "Global 1", "");
        vm.prank(bob);
        store.createModel("global-2", "Global 2", "");

        assertEq(store.tokenByIndex(0), 0);
        assertEq(store.tokenByIndex(1), 1);
    }

    function test_enumeration_balanceChangesOnTransfer() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("xfer-bal", "Transfer Balance", "");

        assertEq(store.balanceOf(alice), 1);
        assertEq(store.balanceOf(bob), 0);

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        assertEq(store.balanceOf(alice), 0);
        assertEq(store.balanceOf(bob), 1);
    }

    // -------------------------------------------------------
    // Slug lookup edge cases
    // -------------------------------------------------------

    function test_getModelBySlug_revertsForNonexistentSlug() public {
        vm.expectRevert("Slug not found");
        store.getModelBySlug("does-not-exist");
    }

    function test_slugUniqueness_differentCasesAreDifferent() public {
        vm.startPrank(alice);
        store.createModel("MyModel", "My Model Upper", "");
        store.createModel("mymodel", "My Model Lower", ""); // different slug
        vm.stopPrank();

        uint256 id0 = store.getModelBySlug("MyModel");
        uint256 id1 = store.getModelBySlug("mymodel");
        assertTrue(id0 != id1);
    }

    // -------------------------------------------------------
    // ERC-721 basics
    // -------------------------------------------------------

    function test_erc721_nameAndSymbol() public view {
        assertEq(store.name(), "Helix Model");
        assertEq(store.symbol(), "HMODEL");
    }

    function test_erc721_supportsInterface() public view {
        // ERC-721 interface ID
        assertTrue(store.supportsInterface(0x80ac58cd));
        // ERC-721 Enumerable interface ID
        assertTrue(store.supportsInterface(0x780e9d63));
    }

    // -------------------------------------------------------
    // setForSale
    // -------------------------------------------------------

    function test_setForSale() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("sale-1", "Sale 1", "");
        assertFalse(store.isForSale(tokenId));

        store.setForSale(tokenId, true);
        assertTrue(store.isForSale(tokenId));

        store.setForSale(tokenId, false);
        assertFalse(store.isForSale(tokenId));
        vm.stopPrank();
    }

    function test_setForSale_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("sale-2", "Sale 2", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.setForSale(tokenId, true);
    }

    function test_setForSale_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("sale-evt", "Sale Evt", "");

        vm.expectEmit(true, false, false, true);
        emit ModelForSaleChanged(tokenId, true);
        store.setForSale(tokenId, true);
        vm.stopPrank();
    }

    // -------------------------------------------------------
    // setSalePrice
    // -------------------------------------------------------

    function test_setSalePrice() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("price-1", "Price 1", "");
        store.setSalePrice(tokenId, 1 ether);
        assertEq(store.salePrice(tokenId), 1 ether);
        vm.stopPrank();
    }

    function test_setSalePrice_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("price-2", "Price 2", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.setSalePrice(tokenId, 1 ether);
    }

    function test_setSalePrice_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("price-evt", "Price Evt", "");

        vm.expectEmit(true, false, false, true);
        emit SalePriceChanged(tokenId, 2 ether);
        store.setSalePrice(tokenId, 2 ether);
        vm.stopPrank();
    }

    // -------------------------------------------------------
    // buyModel
    // -------------------------------------------------------

    function test_buyModel_transfersAndPays() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("buy-1", "Buy 1", "");
        store.setForSale(tokenId, true);
        store.setSalePrice(tokenId, 1 ether);
        vm.stopPrank();

        uint256 aliceBalBefore = alice.balance;

        vm.prank(bob);
        store.buyModel{value: 1 ether}(tokenId);

        assertEq(store.ownerOf(tokenId), bob);
        assertEq(alice.balance, aliceBalBefore + 1 ether);
    }

    function test_buyModel_clearsListing() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("buy-clear", "Buy Clear", "");
        store.setForSale(tokenId, true);
        store.setSalePrice(tokenId, 1 ether);
        vm.stopPrank();

        vm.prank(bob);
        store.buyModel{value: 1 ether}(tokenId);

        assertFalse(store.isForSale(tokenId));
        assertEq(store.salePrice(tokenId), 0);
    }

    function test_buyModel_revertsNotForSale() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("buy-nfs", "Not For Sale", "");

        vm.prank(bob);
        vm.expectRevert("Model not for sale");
        store.buyModel{value: 1 ether}(tokenId);
    }

    function test_buyModel_revertsInsufficientPayment() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("buy-low", "Buy Low", "");
        store.setForSale(tokenId, true);
        store.setSalePrice(tokenId, 2 ether);
        vm.stopPrank();

        vm.prank(bob);
        vm.expectRevert("Insufficient payment");
        store.buyModel{value: 1 ether}(tokenId);
    }

    function test_buyModel_revertsOwnModel() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("buy-self", "Buy Self", "");
        store.setForSale(tokenId, true);
        store.setSalePrice(tokenId, 1 ether);

        vm.expectRevert("Cannot buy own model");
        store.buyModel{value: 1 ether}(tokenId);
        vm.stopPrank();
    }

    function test_buyModel_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("buy-evt", "Buy Evt", "");
        store.setForSale(tokenId, true);
        store.setSalePrice(tokenId, 1 ether);
        vm.stopPrank();

        vm.prank(bob);
        vm.expectEmit(true, true, true, true);
        emit ModelSold(tokenId, alice, bob, 1 ether);
        store.buyModel{value: 1 ether}(tokenId);
    }

    function test_buyModel_revertsSalePriceNotSet() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("buy-noprice", "No Price", "");
        store.setForSale(tokenId, true);
        // salePrice is 0
        vm.stopPrank();

        vm.prank(bob);
        vm.expectRevert("Sale price not set");
        store.buyModel{value: 1 ether}(tokenId);
    }

    // -------------------------------------------------------
    // payForInference
    // -------------------------------------------------------

    function test_payForInference() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("infer-1", "Infer 1", "");
        store.setPublic(tokenId, true);
        store.setInferenceFee(tokenId, 1000); // 10%
        vm.stopPrank();

        vm.prank(bob);
        uint256 nonce = store.payForInference{value: 0.01 ether}(tokenId);
        assertEq(nonce, 0);
    }

    function test_payForInference_ownerShareCorrect() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("infer-share", "Infer Share", "");
        store.setPublic(tokenId, true);
        store.setInferenceFee(tokenId, 500); // 5%
        vm.stopPrank();

        vm.prank(bob);
        store.payForInference{value: 1 ether}(tokenId);

        // 5% of 1 ether = 0.05 ether
        assertEq(store.inferenceFeesAccrued(tokenId), 0.05 ether);
    }

    function test_payForInference_revertsPrivate() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("infer-priv", "Infer Priv", "");
        // isPublic defaults to false

        vm.prank(bob);
        vm.expectRevert("Model is not public");
        store.payForInference{value: 0.01 ether}(tokenId);
    }

    function test_payForInference_revertsOwner() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("infer-own", "Infer Own", "");
        store.setPublic(tokenId, true);

        vm.expectRevert("Owner does not pay for inference");
        store.payForInference{value: 0.01 ether}(tokenId);
        vm.stopPrank();
    }

    function test_payForInference_revertsZeroPayment() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("infer-zero", "Infer Zero", "");
        store.setPublic(tokenId, true);
        vm.stopPrank();

        vm.prank(bob);
        vm.expectRevert("Payment required");
        store.payForInference{value: 0}(tokenId);
    }

    function test_payForInference_incrementsNonce() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("infer-nonce", "Infer Nonce", "");
        store.setPublic(tokenId, true);
        vm.stopPrank();

        vm.startPrank(bob);
        uint256 n0 = store.payForInference{value: 0.01 ether}(tokenId);
        uint256 n1 = store.payForInference{value: 0.01 ether}(tokenId);
        vm.stopPrank();

        assertEq(n0, 0);
        assertEq(n1, 1);
    }

    function test_payForInference_emitsEvent() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("infer-evt", "Infer Evt", "");
        store.setPublic(tokenId, true);
        store.setInferenceFee(tokenId, 1000); // 10%
        vm.stopPrank();

        vm.prank(bob);
        vm.expectEmit(true, true, false, true);
        // ownerShare = 0.01 ether * 1000 / 10000 = 0.001 ether
        emit InferencePaid(tokenId, bob, 0, 0.01 ether, 0.001 ether);
        store.payForInference{value: 0.01 ether}(tokenId);
    }

    // -------------------------------------------------------
    // withdrawInferenceFees
    // -------------------------------------------------------

    function test_withdrawInferenceFees() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("withdraw-1", "Withdraw 1", "");
        store.setPublic(tokenId, true);
        store.setInferenceFee(tokenId, 1000); // 10%
        vm.stopPrank();

        vm.prank(bob);
        store.payForInference{value: 1 ether}(tokenId);
        // ownerShare = 0.1 ether

        uint256 aliceBalBefore = alice.balance;

        vm.prank(alice);
        store.withdrawInferenceFees(tokenId);

        assertEq(alice.balance, aliceBalBefore + 0.1 ether);
        assertEq(store.inferenceFeesAccrued(tokenId), 0);
    }

    function test_withdrawInferenceFees_revertsNoFees() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("withdraw-none", "Withdraw None", "");

        vm.expectRevert("No fees to withdraw");
        store.withdrawInferenceFees(tokenId);
        vm.stopPrank();
    }

    function test_withdrawInferenceFees_nonOwnerReverts() public {
        vm.startPrank(alice);
        uint256 tokenId = store.createModel("withdraw-no", "Withdraw No", "");
        store.setPublic(tokenId, true);
        store.setInferenceFee(tokenId, 1000);
        vm.stopPrank();

        vm.prank(bob);
        store.payForInference{value: 1 ether}(tokenId);

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.withdrawInferenceFees(tokenId);
    }
}
