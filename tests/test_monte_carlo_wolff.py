"""
The Wolff cluster algorithm: what a sweep is, and the distribution it
samples, checked against the exact enumeration of a small lattice.
"""

import itertools

import numpy as np
import pytest

from spectrojotometer.monte_carlo import WolffCluster, detect_frustration, run_ising_mc
from spectrojotometer.monte_carlo.lattices import build_square_lattice_2d


def antiferromagnet(L=4):
    """Square lattice with J = -1: unfrustrated (bipartite), so Wolff applies."""
    spins, bonds, J, E0 = build_square_lattice_2d(L=L, J=-1.0)
    frustrated, gauge, _ = detect_frustration(bonds, J, len(spins))
    assert not frustrated
    return spins, bonds, J, E0, gauge


def exact_averages(bonds, J, E0, gauge, T):
    """
    Exact thermal averages of H = E0 - sum_b J_b s_i s_j by enumeration:
    energy per site, specific heat per site, and <m^2>, with m the
    magnetization per site in the gauge basis (here, the staggered one).
    """
    n = int(np.max(bonds)) + 1
    states = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)
    energy = E0 - (states[:, bonds[:, 0]] * states[:, bonds[:, 1]]) @ J
    m = states @ np.asarray(gauge, dtype=float) / n
    w = np.exp(-(energy - energy.min()) / T)
    w /= w.sum()
    e_avg, e2_avg = w @ energy, w @ energy**2
    return {"E": e_avg / n, "C": (e2_avg - e_avg**2) / (T**2 * n), "M2": w @ m**2}


@pytest.mark.parametrize("T", [0.2, 2.3, 100.0])
def test_a_sweep_flips_about_n_spins(T):
    """
    A sweep is a fixed number of clusters, which flips N spins on
    average: its cost does not grow with the size of the clusters, as it
    did when a sweep was N clusters (O(N^2) in the ordered phase, where
    each cluster covers the whole lattice).
    """
    spins, bonds, J, E0, gauge = antiferromagnet(L=8)
    n = len(spins)
    sim = WolffCluster(spins, bonds, J, E0, seed=3, gauge=gauge)
    sim.set_temperature(T)
    sim.equilibrate(50)
    clusters, flipped = sim.n_clusters, sim.n_flipped
    sweeps = 200
    for _ in range(sweeps):
        sim.metropolis_sweep()
    assert sim.n_clusters - clusters == sweeps * sim.clusters_per_sweep
    assert 0.5 * n < (sim.n_flipped - flipped) / sweeps < 2 * n
    if T == 0.2:
        # Ordered phase: one cluster, the whole lattice.
        assert sim.clusters_per_sweep == 1
    if T == 100.0:
        # Almost no bond is activated: clusters of about one site.
        assert sim.clusters_per_sweep > 0.8 * n


def test_the_number_of_clusters_is_estimated_at_each_temperature():
    spins, bonds, J, E0, gauge = antiferromagnet(L=8)
    sim = WolffCluster(spins, bonds, J, E0, seed=4, gauge=gauge)
    sim.set_temperature(0.2)
    sim.metropolis_sweep()  # without equilibrate: estimated first
    assert sim.clusters_per_sweep == 1
    sim.set_temperature(100.0)
    assert sim.clusters_per_sweep is None
    sim.equilibrate(20)
    assert sim.clusters_per_sweep > 1


def test_wolff_step_flips_one_cluster():
    spins, bonds, J, E0, gauge = antiferromagnet(L=4)
    sim = WolffCluster(spins, bonds, J, E0, seed=5, gauge=gauge)
    sim.set_temperature(2.0)
    before = sim.spins.copy()
    size = sim.wolff_step()
    assert 1 <= size <= len(spins)
    assert int(np.sum(before != sim.spins)) == size
    assert (sim.n_clusters, sim.n_flipped) == (1, size)


@pytest.mark.parametrize("T", [1.5, 2.3, 3.5])
def test_wolff_samples_the_boltzmann_distribution(T):
    """Energy, specific heat and staggered <m^2> of a 4x4 antiferromagnet."""
    spins, bonds, J, E0, gauge = antiferromagnet(L=4)
    exact = exact_averages(bonds, J, E0, gauge, T)
    sim = WolffCluster(spins, bonds, J, E0, seed=7, gauge=gauge)
    sim.set_temperature(T)
    sim.equilibrate(500)
    obs = sim.sample(40000)
    assert obs["E"] == pytest.approx(exact["E"], abs=0.01)
    assert obs["C"] == pytest.approx(exact["C"], rel=0.06, abs=0.01)
    assert obs["M2"] == pytest.approx(exact["M2"], abs=0.01)


def test_runner_with_wolff_agrees_with_the_exact_energy():
    spins, bonds, J, E0, gauge = antiferromagnet(L=4)
    res = run_ising_mc(spins, bonds, J, E0, 1.5, 3.5, 3,
                       n_equil=500, n_sweeps=40000, algorithm="auto", seed=1)
    assert res["algorithm"] == "wolff"
    for T, E in zip(res["T"], res["E"]):
        assert E == pytest.approx(exact_averages(bonds, J, E0, gauge, T)["E"], abs=0.01)
