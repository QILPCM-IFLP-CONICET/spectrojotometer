"""
equivalent_model(..., align=True): the source model moved rigidly onto the
target, each site at the image closest to its site of the target; and
generate_bonds with a reference, which gives the couplings of the target,
with the same names and bonds, in the same order.
"""

import itertools
import logging

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel, equivalent_model, site_maps
from spectrojotometer.model_io import magnetic_model_from_file

from .test_model_mapping import (
    EXAMPLES,
    chain,
    check_equivalent,
    check_mapped,
    swapped,
)
from .test_cli import run_command


def rotation_matrix(angles):
    """Rotation by the Euler angles (z, y, x), in degrees."""
    a, b, c = np.radians(angles)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def distorted_rotated_copy(model, seed=3, noise=0.003):
    """The structure of model with a small strain and random displacements
    of the sites (a distorted phase), its sites in another order, rotated
    and with another origin: StructureMatcher is needed to align it."""
    rng = np.random.default_rng(seed)
    lattice = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    frac = np.linalg.solve(
        lattice.T, np.array(model.site_properties["coord_atomos"], dtype=float).T
    ).T
    order = rng.permutation(len(frac))
    new_lattice = (lattice * np.array([1.004, 0.998, 1.002])[:, None]) @ rotation_matrix(
        (25, -10, 40)
    ).T
    new_frac = (frac[order] + rng.normal(0, noise, frac.shape) + [0.21, 0.37, 0.11]) % 1
    copy = MagneticModel(
        list(new_frac @ new_lattice),
        list(new_lattice),
        magnetic_species=[model.site_properties["magnetic_species"][i] for i in order],
    )
    copy.generate_bonds(0.02, [[0, 3.5]])
    return copy


@pytest.fixture(scope="module")
def spinel():
    model = magnetic_model_from_file(str(EXAMPLES / "cromita_ortogonal.cif"))
    model.generate_bonds(0.02, [[0, 3.5]])
    return model


@pytest.fixture(scope="module")
def distorted(spinel):
    return distorted_rotated_copy(spinel)


def site_distances(model, target):
    a = np.array(model.site_properties["coord_atomos"], dtype=float)
    b = np.array(target.site_properties["coord_atomos"], dtype=float)
    return np.linalg.norm(a - b, axis=1)


def bond_table(model):
    return [(name, coupling["bonds"]) for name, coupling in model.bonds.items()]


def test_the_sites_are_moved_onto_the_target(distorted, spinel):
    assert site_maps(distorted, spinel)[2] == "structure"
    # Without align, the result is in the frame of the source.
    assert site_distances(equivalent_model(distorted, spinel), spinel).max() > 1
    eq = equivalent_model(distorted, spinel, align=True)
    # Only the distortion (random displacements of ~0.025 and the strain)
    # separates them.
    assert site_distances(eq, spinel).max() < 0.15
    # Each site is the image closest to its site of the target.
    lattice = np.array(eq.lattice_properties["bravais_vectors"], dtype=float)
    positions = np.array(eq.site_properties["coord_atomos"], dtype=float)
    targets = np.array(spinel.site_properties["coord_atomos"], dtype=float)
    for cell in itertools.product((-1, 0, 1), repeat=3):
        if any(cell):
            moved = positions + np.array(cell) @ lattice
            assert np.all(
                np.linalg.norm(moved - targets, axis=1)
                > np.linalg.norm(positions - targets, axis=1)
            )


def test_the_aligned_model_is_the_source_model(distorted, spinel):
    """A rigid motion: the same cell (up to a rotation), the same
    couplings, the same coefficients as without align."""
    plain = equivalent_model(distorted, spinel)
    eq = equivalent_model(distorted, spinel, align=True)
    a = np.array(plain.lattice_properties["bravais_vectors"], dtype=float)
    b = np.array(eq.lattice_properties["bravais_vectors"], dtype=float)
    np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-9)
    # The Bravais vectors point as those of the target.
    target = np.array(spinel.lattice_properties["bravais_vectors"], dtype=float)
    cosines = np.sum(b * target, axis=1) / np.linalg.norm(b, axis=1) / np.linalg.norm(target, axis=1)
    assert np.all(cosines > 0.9999)
    rng = np.random.default_rng(0)
    check_equivalent(eq, distorted, spinel, [list(rng.integers(0, 2, 16)) for _ in range(200)])
    assert eq.reference_model is not None
    assert plain.reference_model is None


def test_generate_bonds_follows_the_reference(distorted, spinel, caplog):
    eq = equivalent_model(distorted, spinel, align=True)
    eq.bonds = {}
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        eq.generate_bonds(0.02, [[0, 3.5]])
    assert bond_table(eq) == bond_table(spinel)
    # The distortion changes the lengths by more than the discretization:
    # it is said, and the bonds are kept in the coupling of the reference.
    assert "differ in length from those of the reference" in caplog.text
    # The distance is the mean length in this model.
    lengths = []
    positions = np.array(eq.site_properties["coord_atomos"], dtype=float)
    lattice = np.array(eq.lattice_properties["bravais_vectors"], dtype=float)
    from spectrojotometer.magnetic_model import _bond_offset

    for i, j, tag in eq.bonds["J1"]["bonds"]:
        lengths.append(np.linalg.norm(positions[j] + _bond_offset(tag) @ lattice - positions[i]))
    assert abs(eq.bonds["J1"]["distance"] - np.mean(lengths)) <= 0.01
    # Without the reference, the distortion splits the coupling.
    eq.bonds = {}
    eq.reference_model = None
    eq.generate_bonds(0.02, [[0, 3.5]])
    assert len(eq.bonds) > 1


def test_an_explicit_reference(distorted, spinel):
    eq = equivalent_model(distorted, spinel, align=True)
    eq.bonds, eq.reference_model = {}, None
    eq.generate_bonds(0.02, [[0, 3.5]], reference=spinel)
    assert bond_table(eq) == bond_table(spinel)
    with pytest.raises(ValueError, match="the same sites"):
        eq.generate_bonds(0.02, [[0, 3.5]], reference=chain([0, 1], 2))


def test_an_edited_map_with_align(spinel):
    eq_source = distorted_rotated_copy(spinel, noise=0.0)
    forward = site_maps(eq_source, spinel)[0]
    edited = swapped(forward, forward[0], forward[5])
    eq = equivalent_model(eq_source, spinel, mapping=edited, align=True)
    check_mapped(eq, eq_source, spinel, edited)
    # Every site at the image closest to its site of the target; the two
    # exchanged ones are farther.
    distances = site_distances(eq, spinel)
    assert np.all(np.delete(distances, [0, 5]) < 0.05)


def test_a_supercell_with_another_origin():
    cell = chain([0, 1], 2)
    cell.generate_bonds(0.02, [[0, 1.5]])
    double = chain([10.3, 11.3, 12.3, 13.3], 4)  # far, and shifted
    eq = equivalent_model(cell, double, align=True)
    np.testing.assert_allclose(site_distances(eq, double), 0, atol=1e-9)
    double.generate_bonds(0.02, [[0, 1.5]])
    eq.bonds = {}
    eq.generate_bonds(0.02, [[0, 1.5]])
    assert bond_table(eq) == bond_table(double)


def test_couplings_missing_or_new_in_the_model(caplog):
    """In a chain with gaps 1.0 and 1.2 (couplings J1, J2 of the
    reference), a model with the 1.0 gaps stretched beyond the range loses
    J1 (J2 keeps its name); a model where a longer pair enters the range
    gets a new coupling, after those of the reference."""
    reference = chain([0, 1.0], 2.2)
    reference.generate_bonds(0.02, [[0, 1.25]])
    assert [(n, round(c["distance"], 2)) for n, c in reference.bonds.items()] == [
        ("J1", 1.0),
        ("J2", 1.2),
    ]
    stretched = chain([0, 1.1], 2.2)  # gaps 1.1 and 1.1: both in range
    stretched.reference_model = reference
    stretched.generate_bonds(0.02, [[0, 1.25]])
    assert bond_table(stretched) == bond_table(reference)

    lost = chain([0, 1.3], 2.5)  # gaps 1.3 (out of range) and 1.2
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        lost.generate_bonds(0.02, [[0, 1.25]], reference=reference)
    assert list(lost.bonds) == ["J2"]
    assert lost.bonds["J2"]["bonds"] == reference.bonds["J2"]["bonds"]
    assert "left out" in caplog.text

    caplog.clear()
    reference3 = chain([0, 1.0, 2.0], 3.3)  # gaps 1.0, 1.0, 1.3 (out)
    reference3.generate_bonds(0.02, [[0, 1.25]])
    shrunk = chain([0, 1.0, 2.0], 3.2)  # gaps 1.0, 1.0, 1.2 (in)
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        shrunk.generate_bonds(0.02, [[0, 1.25]], reference=reference3)
    assert list(shrunk.bonds) == ["J1", "J2"]
    assert shrunk.bonds["J1"]["bonds"] == reference3.bonds["J1"]["bonds"]
    assert len(shrunk.bonds["J2"]["bonds"]) == 1
    assert "not in the reference" in caplog.text


def test_saved_and_read_back(distorted, spinel, tmp_path):
    """The positions outside the cell, and so the bonds, survive the CIF."""
    eq = equivalent_model(distorted, spinel, align=True)
    eq.save_cif(str(tmp_path / "eq.cif"))
    back = magnetic_model_from_file(str(tmp_path / "eq.cif"))
    np.testing.assert_allclose(site_distances(back, spinel), site_distances(eq, spinel), atol=1e-3)
    rng = np.random.default_rng(2)
    for conf in [list(rng.integers(0, 2, 16)) for _ in range(50)]:
        np.testing.assert_array_equal(back.coefficient_matrix([conf]), eq.coefficient_matrix([conf]))
    back.bonds = {}
    back.generate_bonds(0.02, [[0, 3.5]], reference=spinel)
    assert bond_table(back) == bond_table(spinel)


def test_command_line(distorted, spinel, tmp_path):
    """map_configs --align --model-out, then bond_generator --reference
    --replace: the couplings of the target."""
    run = run_command
    distorted.save_cif(str(tmp_path / "source.cif"))
    (tmp_path / "source.spin").write_text("-1.0\t" + str([0] * 16) + "\t# ferro\n")
    target = EXAMPLES / "cromita_ortogonal.cif"
    res = run(
        "map_configs", tmp_path / "source.cif", target, tmp_path / "source.spin", "-",
        "--align", "--model-out", tmp_path / "aligned.cif", cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    for name, extra in (("ref.cif", ()), ("out.cif", ("--reference", target, "--replace"))):
        source = target if name == "ref.cif" else tmp_path / "aligned.cif"
        res = run(
            "bond_generator", source, tmp_path / name, "-d", 0.02, "--rmax", 3.5, *extra,
            cwd=tmp_path,
        )
        assert res.returncode == 0, res.stderr
    ref = magnetic_model_from_file(str(tmp_path / "ref.cif"))
    out = magnetic_model_from_file(str(tmp_path / "out.cif"))
    assert bond_table(out) == bond_table(ref)


def test_a_mirrored_source(distorted, spinel):
    """A source described with a left-handed cell (its mirror image along
    z): still aligned, with its couplings as those of the target."""
    mirror = np.diag([1.0, 1.0, -1.0])
    source = MagneticModel(
        [mirror @ p for p in distorted.site_properties["coord_atomos"]],
        [mirror @ v for v in distorted.lattice_properties["bravais_vectors"]],
        magnetic_species=list(distorted.site_properties["magnetic_species"]),
    )
    eq = equivalent_model(source, spinel, align=True)
    assert site_distances(eq, spinel).max() < 0.15
    eq.generate_bonds(0.02, [[0, 3.5]])
    assert bond_table(eq) == bond_table(spinel)
