"""
Reading spin configurations: parse_spin_configurations, the reader of
the library, the main window of visualbond and visualbondweb.
"""

import logging
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from spectrojotometer.model_io import (
    SpinConfigurationError,
    magnetic_model_from_file,
    parse_spin_configurations,
    read_spin_configurations_file,
)

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def test_configurations_and_comments():
    text = "# header\n\n1.5\t[0, 1]\t#  a label  \nnan 011\n2 [1, 0, 1]#7\n"
    energies, confs, comments = parse_spin_configurations(text, 3)
    assert confs == [[0, 1, 0], [0, 1, 1], [1, 0, 1]]  # the first one, padded
    assert comments == ["a label", "", "7"]
    assert energies[0] == 1.5 and np.isnan(energies[1]) and energies[2] == 2.0


@pytest.mark.parametrize(
    "text, line, reason",
    [
        ("# header\n\nabc [0, 1, 0]\n", 3, "invalid energy 'abc'"),
        ("1.0 [0, 1, 0]\n2.0\n", 2, "expected '<energy> <config>'"),
        ("1.0 # only a comment\n", 1, "expected '<energy> <config>'"),
        ("1.0 [0, 1, 0]\n\n2.0 [0, 1, 0, 1]\n", 3, "4 values for 3 sites"),
    ],
)
def test_errors_give_the_line(text, line, reason):
    with pytest.raises(SpinConfigurationError) as info:
        parse_spin_configurations(text, 3)
    assert (info.value.line, info.value.reason) == (line, reason)
    assert str(info.value) == f"line {line}: {reason}"
    assert isinstance(info.value, ValueError)


def test_without_the_number_of_sites_any_length_is_accepted():
    _, confs, _ = parse_spin_configurations("1 [0, 1]\n2 [0, 1, 1, 1]\n")
    assert confs == [[0, 1], [0, 1, 1, 1]]


def test_the_file_reader_pads_to_the_number_of_sites(tmp_path):
    """It padded to model.cell_size, which is always 0: not at all."""
    model = magnetic_model_from_file(str(EXAMPLES / "example1.cif"))
    assert model.lattice_properties["cell_size"] == 3
    path = tmp_path / "short.spin"
    path.write_text("1.0 [1]\n")
    assert read_spin_configurations_file(str(path), model)[1] == [[1, 0, 0]]
    path.write_text("1.0 [1, 0, 0, 1]\n")
    with pytest.raises(SpinConfigurationError, match="line 1: 4 values for 3 sites"):
        read_spin_configurations_file(str(path), model)


def test_optimize_configurations_writes_a_space_after_the_hash(tmp_path):
    """The comments of the known configurations are read without the space."""
    model_file = tmp_path / "example1.cif"
    model_file.write_text((EXAMPLES / "example1.cif").read_text())
    out = tmp_path / "out.spin"
    res = subprocess.run(
        [sys.executable, "-m", "spectrojotometer.cli.optimize_configurations",
         str(model_file), str(out), "--it", "3", "--known", str(EXAMPLES / "example1.spin")],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    rows = [line for line in out.read_text().splitlines() if line and not line.startswith("#")]
    assert rows and all("\t\t# " in line for line in rows), rows


# The main window of visualbond.

tkinter = pytest.importorskip("tkinter")


@pytest.fixture
def app(monkeypatch):
    from spectrojotometer.gui import main as gui_main

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no display available")
    root.withdraw()
    application = gui_main.ApplicationGUI(root)
    monkeypatch.setattr(
        gui_main.fdlg, "askopenfilename", lambda **k: str(EXAMPLES / "example1.cif")
    )
    application.open_model()
    yield application
    application.log_panel.close()
    root.destroy()


def reload(app, text):
    app.spinconfigs.delete("1.0", tkinter.END)
    app.spinconfigs.insert("1.0", text)
    app.reload_configs()


def test_gui_reads_with_the_library(app):
    reload(app, "# header\n1.0 [1]  #  first \nnan [0, 1, 1]\n")
    energies, confs, labels = app.configurations
    assert confs == [[1, 0, 0], [0, 1, 1]]
    assert labels == ["first", "6"]  # without a comment: the index


def test_gui_reports_a_line_without_configuration(app, caplog):
    """It raised IndexError (the line has no field after the energy)."""
    reload(app, "1.0 [1, 0, 0]\n")
    before = app.configurations
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        reload(app, "1.0 [1, 0, 0]\n\n2.5\n")
    assert "Error at line 3: expected '<energy> <config>'" in caplog.text
    assert app.configurations == before  # unchanged
