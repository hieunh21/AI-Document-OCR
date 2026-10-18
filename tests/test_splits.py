import json
from collections import defaultdict
from pathlib import Path

import pytest

from evaluation.splits import DEFAULT_RATIOS, SPLITS_DIR, assign_groups
from tests.conftest import requires_dataset

SPLITS = ("train", "dev", "test")


def test_assign_groups_is_deterministic_and_never_splits_a_group():
    groups = {f"g{i:03d}": [f"g{i:03d}_{k}" for k in range(1 + i % 3)] for i in range(120)}
    strata = {g: ("big" if i % 2 else "small") for i, g in enumerate(sorted(groups))}
    a = assign_groups(groups, strata, DEFAULT_RATIOS, seed=1)
    b = assign_groups(groups, strata, DEFAULT_RATIOS, seed=1)
    c = assign_groups(groups, strata, DEFAULT_RATIOS, seed=2)
    assert a == b
    assert a != c
    sizes = defaultdict(int)
    for g, s in a.items():
        sizes[s] += len(groups[g])
    total = sum(sizes.values())
    for name, ratio in DEFAULT_RATIOS.items():
        assert abs(sizes[name] / total - ratio) < 0.03


def _manifest(name: str) -> dict:
    return json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(not (SPLITS_DIR / "train.json").exists(), reason="manifests not generated")
def test_manifests_are_disjoint_and_groups_do_not_cross():
    split_of_img, split_of_group = {}, defaultdict(set)
    for s in SPLITS:
        m = _manifest(s)
        assert m["num_images"] == len(m["items"])
        for it in m["items"]:
            assert it["img_id"] not in split_of_img, it["img_id"]
            split_of_img[it["img_id"]] = s
            split_of_group[it["group_id"]].add(s)
    assert len(split_of_img) == 1155
    assert all(len(v) == 1 for v in split_of_group.values())
    smoke = {it["img_id"] for it in _manifest("smoke")["items"]}
    assert smoke and all(split_of_img[i] == "dev" for i in smoke)


@requires_dataset
@pytest.mark.skipif(not (SPLITS_DIR / "train.json").exists(), reason="manifests not generated")
def test_manifest_hashes_match_local_files():
    from evaluation.dataset_adapter import DEFAULT_ROOT
    from evaluation.dedup import sha256_file

    for it in _manifest("dev")["items"][:25]:
        assert sha256_file(Path(DEFAULT_ROOT) / it["path"]) == it["sha256"]
