"""Numba kernels for the Monte Carlo simulation of the Ising model.

Every kernel works on flat arrays (CSR format for the lists of
neighbours). The random number generator is a SplitMix64 of its own,
with its state in an np.uint64 array of length 1, so that each instance
has its own stream, independent of the global generators of numba and
NumPy.

Without numba, a no-op decorator is used and the kernels work the same
in pure Python (much more slowly).
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(func):
            return func

        return deco


# ---------------------------------------------------------------------------
# RNG (SplitMix64)
# ---------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_TWO53 = 1.0 / 9007199254740992.0  # 2**-53


@njit(cache=True, inline="always")
def _splitmix64(state):
    """Advance the state and return a pseudo-random uint64."""
    state[0] = state[0] + _GOLDEN
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def _rand_double(state):
    """Uniform in [0, 1), from the 53 highest bits."""
    return float(_splitmix64(state) >> np.uint64(11)) * _TWO53


@njit(cache=True, inline="always")
def _rand_int(state, n):
    """Uniform integer in [0, n)."""
    return int(_splitmix64(state) % np.uint64(n))


# ---------------------------------------------------------------------------
# Energy and local field
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _interaction_energy_kernel(spins, bonds, J_vals):
    """Return sum_{<i,j>} J_ij s_i s_j (without E0)."""
    e = 0.0
    for k in range(bonds.shape[0]):
        i = bonds[k, 0]
        j = bonds[k, 1]
        e += J_vals[k] * float(spins[i]) * float(spins[j])
    return e


@njit(cache=True, fastmath=True, inline="always")
def _local_field(spins, nbr_idx, nbr_J, nbr_start, i):
    """Molecular field h_i = sum_j J_ij s_j."""
    h = 0.0
    a = nbr_start[i]
    b = nbr_start[i + 1]
    for k in range(a, b):
        h += nbr_J[k] * float(spins[nbr_idx[k]])
    return h


# ---------------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------------

@njit(cache=True)
def _metropolis_sweep_kernel(spins, nbr_idx, nbr_J, nbr_start, beta, state):
    """A full sweep: N flip attempts at random sites."""
    N = spins.shape[0]
    for _ in range(N):
        i = _rand_int(state, N)
        h = _local_field(spins, nbr_idx, nbr_J, nbr_start, i)
        si = spins[i]
        dE = 2.0 * si * h
        if dE <= 0.0:
            spins[i] = -si
        else:
            if _rand_double(state) < np.exp(-beta * dE):
                spins[i] = -si


# ---------------------------------------------------------------------------
# Wolff
# ---------------------------------------------------------------------------

@njit(cache=True)
def _wolff_clusters_kernel(spins, nbr_idx, nbr_p_add, nbr_start, state, stack,
                           min_flipped, max_clusters):
    """Wolff clusters, until min_flipped spins have been flipped in total
    or max_clusters clusters have been flipped, whatever comes first.

    Each cluster grows from a random seed. A neighbour v of a site of the
    cluster is added with probability nbr_p_add (1 - exp(-2 beta J), the
    couplings being those of the gauge basis, all of them ferromagnetic;
    see WolffCluster) only if spins[v] still has the sign of the cluster
    before the flip: this is the alignment condition of the algorithm,
    without which the cluster crosses domain walls. Every site is flipped
    as soon as it is added, so the flip itself marks it as visited, and
    the cost of a cluster is proportional to its size (each site is
    pushed at most once onto stack, of length N).

    Returns (number of clusters, number of flipped spins).
    """
    N = spins.shape[0]
    clusters = 0
    flipped = 0
    while flipped < min_flipped and clusters < max_clusters:
        seed_site = _rand_int(state, N)
        cluster_spin = spins[seed_site]
        spins[seed_site] = -cluster_spin
        stack[0] = seed_site
        top = 1
        size = 1
        while top > 0:
            top -= 1
            u = stack[top]
            for k in range(nbr_start[u], nbr_start[u + 1]):
                v = nbr_idx[k]
                if spins[v] == cluster_spin and _rand_double(state) < nbr_p_add[k]:
                    spins[v] = -cluster_spin
                    stack[top] = v
                    top += 1
                    size += 1
        clusters += 1
        flipped += size
    return clusters, flipped

# ---------------------------------------------------------------------------
# Warmup (optional: call it before benchmarks)
# ---------------------------------------------------------------------------

def warmup():
    """Force the compilation of every kernel, with trivial data."""
    if not HAS_NUMBA:
        return
    spins = np.ones(4, dtype=np.int8)
    bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    J_vals = np.ones(4, dtype=np.float64)
    nbr_idx = np.array([1, 3, 0, 2, 1, 3, 0, 2], dtype=np.int64)
    nbr_J = np.ones(8, dtype=np.float64)
    nbr_start = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    state = np.array([_GOLDEN], dtype=np.uint64)
    stack = np.zeros(4, dtype=np.int64)

    _interaction_energy_kernel(spins, bonds, J_vals)
    _metropolis_sweep_kernel(spins, nbr_idx, nbr_J, nbr_start, 1.0, state)
    _wolff_clusters_kernel(spins, nbr_idx, nbr_J * 0.5, nbr_start, state, stack, 4, 4)
