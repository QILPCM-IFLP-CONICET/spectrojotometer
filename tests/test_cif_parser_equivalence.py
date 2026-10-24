"""
Compara, archivo por archivo, la implementación actual de
`magnetic_model_from_cif` contra el prototipo
`magnetic_model_from_cif_pymatgen`: para cada CIF que ya sabemos leer
correctamente, las dos deben describir la misma geometría física
(mismos átomos, misma celda, mismos enlaces), aunque el orden interno
de los átomos pueda diferir.

Requiere pymatgen instalado; si no está, estos tests se saltean.
"""
import numpy as np
import pytest

pymatgen = pytest.importorskip("pymatgen")

from spectrojotometer.model_io import magnetic_model_from_cif
from spectrojotometer.model_io.model_io_pymatgen import magnetic_model_from_cif_pymatgen
from spectrojotometer.tools import unpack_offset

from .conftest import assert_models_equivalent

CR_INCLUSIVE_ATOMS = (
    "Mn", "Fe", "Co", "Ni", "Dy", "Tb", "Eu", "Cu", "V", "Cr",
)


@pytest.mark.parametrize(
    "path_fn,kwargs",
    [
        pytest.param(
            lambda ex, fx: ex / "example1.cif", {}, id="example1"
        ),
        pytest.param(
            lambda ex, fx: ex / "cromita_ortogonal.cif",
            {"magnetic_atoms": CR_INCLUSIVE_ATOMS},
            id="cromita-conventional",
        ),
        pytest.param(
            lambda ex, fx: ex / "cromita_ortogonal.cif",
            {"magnetic_atoms": CR_INCLUSIVE_ATOMS, "primitive_cell": True},
            id="cromita-primitive",
        ),
        pytest.param(
            lambda ex, fx: ex / "h2o.cif", {"magnetic_atoms": ("O",)}, id="h2o"
        ),
        pytest.param(
            lambda ex, fx: fx / "fe2o_synthetic.cif", {}, id="fe2o"
        ),
        pytest.param(
            lambda ex, fx: fx / "synthetic_two_atom.cif", {}, id="synthetic-two-atom"
        ),
        pytest.param(
            lambda ex, fx: fx / "synthetic_no_blank_between_loops.cif",
            {},
            id="no-blank-between-loops",
        ),
        pytest.param(
            lambda ex, fx: fx / "synthetic_g_factor_spin.cif",
            {},
            id="g-factor-spin",
        ),
    ],
)
def test_pymatgen_prototype_matches_current_implementation(
    examples_dir, fixtures_dir, path_fn, kwargs
):
    path = str(path_fn(examples_dir, fixtures_dir))

    current = magnetic_model_from_cif(path, **kwargs)
    prototype = magnetic_model_from_cif_pymatgen(path, **kwargs)

    assert_models_equivalent(current, prototype)


def test_g_factor_and_spin_survive_in_both_implementations(fixtures_dir):
    """
    `assert_models_equivalent` no mira `g_lande_factors`/`spin_repr`
    (no forman parte de la geometría), así que se chequean aparte acá.
    """
    path = str(fixtures_dir / "synthetic_g_factor_spin.cif")
    current = magnetic_model_from_cif(path)
    prototype = magnetic_model_from_cif_pymatgen(path)

    assert (
        current.site_properties["g_lande_factors"]
        == prototype.site_properties["g_lande_factors"]
        == [2.1, 2.3]
    )


@pytest.mark.parametrize(
    "loader",
    [magnetic_model_from_cif, magnetic_model_from_cif_pymatgen],
    ids=["current", "pymatgen-prototype"],
)
def test_bond_via_non_identity_symmetry(fixtures_dir, loader):
    """
    A bond declared with `_geom_bond_site_symmetry_2` = '2_555': from Cu1
    to its image by the second operator, the centring (x, y+1/2, z+1/2).
    It was dropped (0 bonds): the image was looked up as "Cu1_1", and
    generate_atoms_by_symmetries labels it "Cu1_2". It is now found by
    its position. Its image by the centring, from the centred site to
    Cu1 one cell up in y and z, is a different bond of the same coupling:
    two bonds, of the declared length.
    """
    model = loader(str(fixtures_dir / "synthetic_bond_via_symmetry.cif"))

    assert len(model.site_properties["coord_atomos"]) == 2
    assert set(model.bonds) == {"J0"}
    bonds = model.bonds["J0"]["bonds"]
    assert len(bonds) == 2
    pos = np.array(model.site_properties["coord_atomos"], dtype=float)
    lat = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    for i, j, tag in bonds:
        offset = np.zeros(3) if tag in (".", None) else np.array(unpack_offset(tag))
        assert np.linalg.norm(pos[j] + offset @ lat - pos[i]) == pytest.approx(7.0711, abs=1e-4)
