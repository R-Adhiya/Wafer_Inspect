"""Regression test asserting exact bitwise reproducibility of synthetic wafer generator against Phase 8 baseline."""

import hashlib

import pytest
from app.utils.synthetic import (
    create_standard_synthetic_pair,
    generate_layout_wafer,
    generate_synthetic_wafer,
)

# Hard-coded SHA-256 digests of the Phase 8 generator outputs across multiple seeds and layouts
EXPECTED_PHASE8_HASHES = {
    # Seed 42
    ("clean", 42): "3c5389974f7c1fb7d6a510e420adacf88c111e2ed4d785be5c1d3bb78fc9324e",
    ("edge_ring", 42): "69251d17f72cb00005b0b7812db7391d709fd5caf443e83a2bf62af617221729",
    ("center_cluster", 42): "0edd9abde001eab84b40c9df94b829c973fb050e0ccbf1ff1361c384fce52b23",
    ("donut", 42): "1c0e88884d1fc7dd9bd9d015803acd8990214af35676e5a62e9a357db5b3bad2",
    ("scratch_line", 42): "4c56eea7139ab792a0e263d1b05408b3cdac8d663b149fb4aafe1ee475572014",
    ("local_cluster", 42): "b7d849b99ba78b72eeca7654337da6fc215cc51b6123136aa98cb415836775cb",
    ("random", 42): "74b58a3f94ba028133a33c364d31acc18be20b862be6ebd9c9d40ad57b3b855e",
    ("pair_ref", 42): "3c5389974f7c1fb7d6a510e420adacf88c111e2ed4d785be5c1d3bb78fc9324e",
    ("pair_test", 42): "49829d858528ae781bf3702d9a05aa8b50a16264faa3720aed08e06a39315235",
    # Seed 123
    ("clean", 123): "ac02e741221b6f1e6d12ff075db6cb2056097cea8c3584354cb0c4f76efb3e89",
    ("edge_ring", 123): "10ee584f31b983ff9f702830b66105f184beddae7e587a95d4f173640c6e9756",
    ("center_cluster", 123): "0c80590fd0c6aa9c5137e1932ca578c8440985de9d486d8572886af552c290b7",
    ("donut", 123): "3cc8d2440756dea4b0ac4f957ebde6188f01769854e13666e40d9cb2330c90d9",
    ("scratch_line", 123): "70e1e17b24df5925cce50ed7fedbaf068df00913c3699e8a12e61dbd26624d29",
    ("local_cluster", 123): "0b64219f6de2acaf4f257a44777db8973ed50ed7dfe2db2e7760ad959ce2e4f4",
    ("random", 123): "d054d863e7b322b076aada9141ba74885908e63627fc31083ce91df79035e895",
    ("pair_ref", 123): "ac02e741221b6f1e6d12ff075db6cb2056097cea8c3584354cb0c4f76efb3e89",
    ("pair_test", 123): "e6c3cbb89eae8a10310780b178ed312cf81e88eabc0bdb26b12439f4693633d2",
    # Seed 999
    ("clean", 999): "b7618ae49945850cefddc7225e7ca65596e9b5a8d699fe9e2d2bd60602096c26",
    ("edge_ring", 999): "fe44e74f9d2bf58f4ec50da8a6d6e298357788ea5f1e19b7ec5b9552aa780398",
    ("center_cluster", 999): "ba2402b53fd4036d5923c1fa2e40b5f38acb3f4257a2f44ec3c7929c794e9558",
    ("donut", 999): "ba3c4602e3deedeee0119de5268da1c43baf800a2eba4f1c640163e49c1ca95b",
    ("scratch_line", 999): "ad80929254d2faa657b4864fb0bc585eee0ee41a0116271c41f20dada845b886",
    ("local_cluster", 999): "6502f2e381bb1500dbcdb413818be85edb86cd07b484c703edd4d6c984894496",
    ("random", 999): "6fb8e95b3705799322cf2d7c7597722f4d6d6478e7cd8ae4353b40865e9441dd",
    ("pair_ref", 999): "b7618ae49945850cefddc7225e7ca65596e9b5a8d699fe9e2d2bd60602096c26",
    ("pair_test", 999): "67ac201e3cf52f333b6039d526270df56e38a1da98cacbce636b161e31fbcfd7",
}


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_clean_wafer_generator_matches_phase8_hash(seed: int) -> None:
    """Verify clean wafer generator outputs match phase-8 SHA256 hashes bit-for-bit."""
    img, _ = generate_synthetic_wafer(size=512, seed=seed)
    digest = hashlib.sha256(img.tobytes()).hexdigest()
    expected = EXPECTED_PHASE8_HASHES[("clean", seed)]
    assert digest == expected, f"Clean wafer hash mismatch for seed {seed}: got {digest}, expected {expected}"


@pytest.mark.parametrize("seed", [42, 123, 999])
@pytest.mark.parametrize(
    "layout",
    ["edge_ring", "center_cluster", "donut", "scratch_line", "local_cluster", "random"],
)
def test_layout_wafer_generator_matches_phase8_hash(seed: int, layout: str) -> None:
    """Verify layout wafer generator outputs match phase-8 SHA256 hashes for all 6 patterns."""
    img, _, _ = generate_layout_wafer(defect_layout=layout, size=512, seed=seed)
    digest = hashlib.sha256(img.tobytes()).hexdigest()
    expected = EXPECTED_PHASE8_HASHES[(layout, seed)]
    assert digest == expected, f"Layout {layout} hash mismatch for seed {seed}: got {digest}, expected {expected}"


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_standard_synthetic_pair_matches_phase8_hash(seed: int) -> None:
    """Verify standard synthetic pair outputs match phase-8 SHA256 hashes bit-for-bit."""
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=seed)
    ref_digest = hashlib.sha256(ref_img.tobytes()).hexdigest()
    test_digest = hashlib.sha256(test_img.tobytes()).hexdigest()

    expected_ref = EXPECTED_PHASE8_HASHES[("pair_ref", seed)]
    expected_test = EXPECTED_PHASE8_HASHES[("pair_test", seed)]

    assert ref_digest == expected_ref, f"Pair ref hash mismatch for seed {seed}"
    assert test_digest == expected_test, f"Pair test hash mismatch for seed {seed}"
