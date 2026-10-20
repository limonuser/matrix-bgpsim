"""
Tests for the CPU backend.

Routes are checked for every AS pair against TopoSim, the route-propagation
simulator in benchmark/correctness_test.py, on small random topologies. The
topologies are small enough to run on any machine in a few seconds.
"""
import json
import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from matrix_bgpsim import RMatrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from correctness_test import TopoSim, P2C  # noqa: E402


def random_rels(seed):
    """
    Random topology with a core part (providers always rank higher than
    customers, so there are no provider cycles), chains of stub ASes hanging
    off it, and some parts that are disconnected from it.
    """
    rng = random.Random(seed)
    new_asns = iter(str(asn) for asn in rng.sample(range(1, 400000), 400))

    core = [next(new_asns) for _ in range(rng.randint(3, 25))]
    p_c2p, p_p2p = rng.uniform(0.05, 0.3), rng.uniform(0.0, 0.25)
    rels = []
    for i, asn1 in enumerate(core):
        for asn2 in core[i + 1:]:
            r = rng.random()
            if r < p_c2p:
                rels.append((asn1, asn2, -1))
            elif r < p_c2p + p_p2p:
                rels.append((asn1, asn2, 0))

    # chains of single-homed ASes (branches), sometimes forking into sub-trees
    attach_points = list(core)
    for _ in range(rng.randint(0, 6)):
        parent = rng.choice(attach_points)
        for _ in range(rng.randint(1, 4)):
            child = next(new_asns)
            rels.append((parent, child, -1))
            attach_points.append(child)
            parent = child

    # parts that are disconnected from the rest: a tree and a peer pair
    if rng.random() < 0.5:
        root, child1, child2, grandchild = (next(new_asns) for _ in range(4))
        rels += [(root, child1, -1), (root, child2, -1), (child1, grandchild, -1)]
    if rng.random() < 0.5:
        rels.append((next(new_asns), next(new_asns), 0))

    rng.shuffle(rels)
    return rels


def write_rels(rels, path):
    path.write_text("".join(f"{asn1}|{asn2}|{rel}\n" for asn1, asn2, rel in rels))
    return path


def simulate(rels, n_jobs=2, save_next_hop=True, **kwargs):
    rmatrix = RMatrix(rels, **kwargs)
    rmatrix.run(n_jobs=n_jobs, max_iter=32, save_next_hop=save_next_hop, backend="cpu")
    return rmatrix


def assert_matches_reference(rmatrix, rels_file):
    # same deterministic tie-break as check_correctness(): lowest core index
    def break_tie(ties):
        if len(ties) == 1:
            return ties[0]
        return ties[int(np.argmin([rmatrix.asn2idx(t) for t in ties]))]

    ts = TopoSim(rels_file, break_tie, prefer_change=True, exclude=set())
    for asn2 in ts.ngbrs:
        routes = ts.sim_all_routes_to(asn2)
        for asn1 in ts.ngbrs:
            if asn1 in routes:
                path, rel = routes[asn1]
                expected = (P2C if rel is None else -rel, len(path)), path
            else:
                expected = (None, 0), None
            actual = rmatrix.get_state(asn1, asn2), rmatrix.get_path(asn1, asn2)
            assert actual == expected, f"{asn1} -> {asn2}"


@pytest.mark.parametrize("seed", range(15))
def test_matches_reference(seed, tmp_path):
    rels = random_rels(seed)
    rels_file = write_rels(rels, tmp_path / "rels.txt")
    assert_matches_reference(simulate(rels_file), rels_file)


def test_unreachable_route():
    # 1 - 2 - 3 are peers, so 1 cannot reach 3 (valley-free); 4 is a stub under 1
    rels = [("1", "2", 0), ("2", "3", 0), ("1", "4", -1)]
    rmatrix = simulate(rels)
    for asn1, asn2 in [("1", "3"), ("3", "1"), ("4", "3"), ("3", "4")]:
        assert rmatrix.get_state(asn1, asn2) == (None, 0)
        assert rmatrix.get_path(asn1, asn2) is None


def test_get_state_returns_python_ints():
    rels = [("1", "2", -1), ("1", "3", -1), ("2", "4", -1), ("3", "4", -1), ("4", "5", -1)]
    rmatrix = simulate(rels)
    for asn1, asn2 in [("1", "4"), ("5", "1"), ("1", "5")]:
        s_type, s_len = rmatrix.get_state(asn1, asn2)
        assert type(s_len) is int
        json.dumps([s_type, s_len])


EXCLUDED = ["3", "5"]


@pytest.mark.parametrize("make_excluded", [
    set, frozenset, list, tuple, dict.fromkeys,
    lambda asns: (asn for asn in asns),  # one-shot generator
    lambda asns: (lambda asn: asn in asns),  # callable
], ids=["set", "frozenset", "list", "tuple", "dict", "generator", "callable"])
def test_excluded(make_excluded):
    rels = [("1", "2", 0), ("1", "3", -1), ("2", "4", -1), ("3", "5", -1),
            ("4", "5", -1), ("4", "6", -1), ("6", "7", -1)]
    rmatrix = RMatrix(rels, excluded=make_excluded(EXCLUDED))
    for asn in EXCLUDED:
        assert not rmatrix.has_asn(asn)
    for asn in ["1", "2", "4", "6", "7"]:
        assert rmatrix.has_asn(asn)


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_start_methods(method, tmp_path):
    rels = random_rels(0)
    expected = simulate(rels)
    old_method = multiprocessing.get_start_method()
    multiprocessing.set_start_method(method, force=True)
    try:
        rmatrix = simulate(rels)
    finally:
        multiprocessing.set_start_method(old_method, force=True)
    np.testing.assert_array_equal(rmatrix.__state__, expected.__state__)
    np.testing.assert_array_equal(rmatrix.__next_hop__, expected.__next_hop__)


# random_rels(2) has 5 core ASes, so n_jobs=8 leaves some workers with no columns.
# (Not more than 61: on Windows, multiprocessing.Pool in older Pythons hangs with more.)
@pytest.mark.parametrize("n_jobs", [1, 3, 8])
def test_n_jobs(n_jobs):
    rels = random_rels(2)
    expected = simulate(rels, n_jobs=1)
    rmatrix = simulate(rels, n_jobs=n_jobs)
    np.testing.assert_array_equal(rmatrix.__state__, expected.__state__)
    np.testing.assert_array_equal(rmatrix.__next_hop__, expected.__next_hop__)


def test_without_next_hop():
    rels = random_rels(2)
    expected = simulate(rels)
    rmatrix = simulate(rels, save_next_hop=False)
    np.testing.assert_array_equal(rmatrix.__state__, expected.__state__)
    assert rmatrix.__next_hop__ is None
    asn = rmatrix.idx2asn(0)
    with pytest.raises(AssertionError):
        rmatrix.get_path(asn, asn)


def test_dump_and_load(tmp_path):
    rels = random_rels(3)
    rmatrix = simulate(rels)
    rmatrix.dump(tmp_path / "rmatrix.lz4")
    loaded = RMatrix.load(tmp_path / "rmatrix.lz4")
    np.testing.assert_array_equal(loaded.__state__, rmatrix.__state__)
    np.testing.assert_array_equal(loaded.__next_hop__, rmatrix.__next_hop__)
    asns = sorted({asn for rel in rels for asn in rel[:2]})
    for asn1 in asns:
        for asn2 in asns:
            assert loaded.get_path(asn1, asn2) == rmatrix.get_path(asn1, asn2)
