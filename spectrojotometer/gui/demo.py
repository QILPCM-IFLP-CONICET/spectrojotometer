"""Demo standalone del panel Monte Carlo.

Uso:
    python -m spectrojotometer.gui.demo
"""

import tkinter as tk
from tkinter import ttk

from .monte_carlo_panel import MonteCarloPanel
from .mock_model import MockMagneticModel
from ..monte_carlo.lattices import build_square_lattice_2d


def main():
    root = tk.Tk()
    root.title("Spectrojotometer - Monte Carlo (demo)")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Panel 1: modelo mock (via adapter)
    model = MockMagneticModel()
    panel1 = MonteCarloPanel(notebook, get_model=lambda: model)
    notebook.add(panel1, text="Modelo mock")

    # Panel 2: datos directos (red cuadrada 2D L=8)
    panel2 = MonteCarloPanel(notebook)
    spins, bonds, J, E0 = build_square_lattice_2d(L=8, J=1.0)
    panel2.load_data(spins, bonds, J, E0)
    # Ajustar defaults para este caso
    panel2.var_Tmin.set(1.5)
    panel2.var_Tmax.set(3.5)
    notebook.add(panel2, text="Ising 2D L=8")

    root.geometry("1000x700")
    root.mainloop()


if __name__ == "__main__":
    main()