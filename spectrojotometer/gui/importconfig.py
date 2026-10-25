#!/usr/bin/env python3
#  from tkmessagebox import *
"""
Import dialog for spin configurations from another model.

"""
#  from visualbond.dialogs.finddialog import FindDialog
import logging
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    DISABLED,
    END,
    HORIZONTAL,
    LEFT,
    NORMAL,
    RIGHT,
    TOP,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    LabelFrame,
    OptionMenu,
    PanedWindow,
    StringVar,
    Toplevel,
    X,
    messagebox,
)
from tkinter import filedialog as fdlg
from tkinter.scrolledtext import ScrolledText


from spectrojotometer.magnetic_model import (
    compatible_configurations,
    equivalent_model,
    map_configurations,
    site_maps,
)
from spectrojotometer.model_io import magnetic_model_from_file, parse_spin_configurations

logger = logging.getLogger(__name__)


class ImportConfigWindow(Toplevel):
    """
    Window for importing configurations.
    """

    def __init__(self, app):
        self.app = app
        Toplevel.__init__(self, app.root)
        #  self.root = Toplevel(app.root)
        self.dictatoms = []
        self.transient(app.root)
        self.model = app.model
        self.configurations = ([], [], [])
        # The other models, by (file, primitive cell), and their files.
        self.models = {}
        self.filenames = []
        self.parameters = {}
        self.energy_scale = 1.0

        controls1 = LabelFrame(self, text="Parameters", padx=5, pady=5)

        row = Frame(controls1)
        Label(row, text="model file:").pack(side=LEFT)
        self.selected_model = StringVar()
        self.selected_model.trace_add("write", self.onmodelselect)
        self.optmodels = OptionMenu(row, self.selected_model, "[other model]")
        self.optmodels.pack(side=LEFT, fill=X)
        # How to read the other model, independently of the current one:
        # e.g. its conventional cell, for a current model in the primitive
        # cell. By default, as the main window.
        self.primitive_var = BooleanVar(value=app.primitive_cell_var.get())
        self._reverting = False
        Checkbutton(row, text="Primitive cell", variable=self.primitive_var).pack(side=LEFT)
        self.primitive_var.trace_add("write", self.onprimitivechange)
        self.sitemap = Entry(row)
        self.sitemap.pack(side=RIGHT)
        Label(row, text="site map").pack(side=RIGHT)
        row.pack(side=TOP, fill=X)

        row = Frame(controls1)
        Label(row, text="Length tolerance:").pack(side=LEFT)
        self.tol = Entry(row, validate="key", validatecommand=self.app.vcmdf)
        self.tol.insert(10, 0.1)
        self.tol.pack(side=LEFT)
        row.pack(side=TOP, fill=X)
        controls1.pack(side=TOP, fill=X)

        #  controls2 = Frame(self, padx=5, pady=5)
        controls2 = PanedWindow(self, orient=HORIZONTAL)
        controls2.pack(side=TOP, fill=BOTH, expand=1)
        controls2l = LabelFrame(controls2, text="inputs", padx=5, pady=5)
        self.inputconfs = ScrolledText(controls2l, height=10, width=80, undo="True")
        self.inputconfs.pack()
        self.inputconfs.tag_configure("sel", background="black", foreground="gray")
        buttons = Frame(controls2l)
        Button(
            buttons,
            text="Load Configuration from File",
            command=self.configs_from_file,
        ).pack(side=RIGHT)
        Button(buttons, text="Import", command=self.map_confs).pack(side=LEFT)
        buttons.pack(side=BOTTOM, fill=X)

        #  controls2l.pack(side=LEFT, fill=Y)
        controls2.add(controls2l)
        controls2r = LabelFrame(controls2, text="in main model", padx=5, pady=5)
        self.outputconfs = ScrolledText(controls2r, height=10, width=80)
        self.outputconfs.config(state=DISABLED)
        self.outputconfs.pack()
        self.outputconfs.tag_configure("sel", background="black", foreground="gray")
        #  controls2r.pack(side=RIGHT, fill=Y)
        controls2.add(controls2r)
        #  controls2.pack(side=TOP, fill=X)

        framebts = Frame(self)
        Button(
            framebts,
            text="Send to main application",
            command=self.send_to_application,
        ).pack(side=LEFT)
        Button(framebts, text="Close", command=self.close_window).pack(side=RIGHT)
        Button(
            framebts,
            text="Export transformed model...",
            command=self.export_model,
        ).pack(side=RIGHT)
        # Export the model moved rigidly onto the current one, each site at
        # the image closest to its site (equivalent_model(align=True)).
        self.align_var = BooleanVar(value=False)
        Checkbutton(
            framebts, text="Align onto the current model", variable=self.align_var
        ).pack(side=RIGHT)
        framebts.pack(side=BOTTOM, fill=X)
        self.grab_set()
        app.root.wait_window(self)

    def onmodelselect(self, *args):
        """On model select: ask for the file of "[other model]", or show
        the site map of the selected one."""
        name = self.selected_model.get()
        if name == "[other model]":
            filename = fdlg.askopenfilename(
                initialdir=self.app.datafolder + "/",
                title="Select file to open",
                filetypes=(
                    ("cif files", "*.cif"),
                    ("Wien2k struct files", "*.struct"),
                    ("all files", "*.*"),
                ),
            )
            logger.debug("model file: %s", filename)
            if filename == "":
                return
            self.app.datafolder = str(Path(filename).parent)
            model = self._load_model(filename)
            if model is None:
                return
            if filename not in self.filenames:
                self.filenames.append(filename)
                self._update_menu()
            # Tcl disables the traces of a variable while one of them runs:
            # this does not call onmodelselect again, nor update the text of
            # the menu button (it kept showing "[other model]"), which is
            # read again when its variable is assigned.
            self.selected_model.set(filename)
            self.optmodels.configure(textvariable=self.selected_model)
            self._show_site_map(filename, model)
            return
        model = self._load_model(name)
        if model is not None:
            self._show_site_map(name, model)

    def onprimitivechange(self, *args):
        """Read the selected model again, with the new cell option."""
        name = self.selected_model.get()
        if self._reverting or name not in self.filenames:
            return
        model = self._load_model(name)
        if model is None:
            # Keep the previous option, and the model read with it.
            self._reverting = True
            try:
                self.primitive_var.set(not self.primitive_var.get())
            finally:
                self._reverting = False
            return
        self._show_site_map(name, model)

    def _load_model(self, filename):
        """
        The model of filename, read with the cell option of the window
        (cached). If it cannot be read so, show why and return None.
        """
        primitive = self.primitive_var.get()
        key = (filename, primitive)
        if key not in self.models:
            try:
                self.models[key] = magnetic_model_from_file(
                    filename=filename, primitive_cell=primitive
                )
            except ValueError as error:
                hint = (
                    "\n\nUncheck \"Primitive cell\" to read it in its "
                    "conventional cell." if primitive else ""
                )
                messagebox.showerror("Import", f"{error}{hint}", parent=self)
                return None
        return self.models[key]

    def _current_model(self):
        """The selected model, read with the current cell option, if any."""
        return self.models.get((self.selected_model.get(), self.primitive_var.get()))

    def _update_menu(self):
        menu = self.optmodels["menu"]
        menu.delete(0, "end")
        menu.add_command(
            label="[other model]",
            command=lambda: self.selected_model.set("[other model]"),
        )
        logger.debug("updating menu")
        for key in self.filenames:
            menu.add_command(
                label=key,
                command=lambda value=key: self.selected_model.set(value),
            )

    def _show_site_map(self, name, model1):
        """The site map from the current model to model1, and the scale of
        the energies."""
        tol = float(self.tol.get())
        model2 = self.app.model
        size1 = len(model1.site_properties["coord_atomos"])
        size2 = len(model2.site_properties["coord_atomos"])
        self.energy_scale = float(size2) / float(size1)
        logger.info(
            "%s: %d sites (%s cell); energies scaled by %g.",
            name,
            size1,
            "primitive" if self.primitive_var.get() else "conventional",
            self.energy_scale,
        )
        if self.energy_scale < 1.0:
            # Only the configurations periodic with the current cell can be
            # imported; map_confs checks each one.
            logger.info(
                "The cell of the current model is smaller than the one of %s: "
                "only its configurations periodic with the current cell can "
                "be imported.",
                name,
            )
        # For each site of the current model, the site of the other model at
        # the same position: comparing positions or, if that does not find
        # every site, aligning the structures (see site_map).
        dictatoms, _, method = site_maps(model1, model2, tol)
        if -1 in dictatoms:
            logger.warning(
                "Sites %s of the current model have no counterpart in %s: "
                "edit the site map, or increase the length tolerance.",
                [i for i, j in enumerate(dictatoms) if j < 0],
                name,
            )
        elif method == "structure":
            logger.info(
                "The positions of %s do not match the current model directly "
                "(different origin, orientation or cell); the site map comes "
                "from aligning both structures with pymatgen's "
                "StructureMatcher.",
                name,
            )
        else:
            logger.info("Site map of %s: %s", name, dictatoms)
        self.dictatoms = list(dictatoms)
        self.sitemap.delete(0, END)
        self.sitemap.insert(0, dictatoms.__str__()[1:-1])

    def export_model(self):
        """
        Save, as a CIF file, the model of the configurations rewritten in
        the cell and order of sites of the current model: its own
        positions and couplings, with the sites of the current model (see
        magnetic_model.equivalent_model). If the site map was edited, its
        sites follow the edited map, as the imported configurations.

        With "Align onto the current model" checked, the model is moved
        rigidly onto the current one (rotated and translated) and each site
        is the image closest to its site of the current model, instead of
        being in the frame of the other model and wrapped into the cell
        (equivalent_model(..., align=True)): each bond (i, j, cell shift)
        of the current model is then the same bond in the exported one.
        """
        if not self._ensure_model():
            return
        name = self.selected_model.get()
        source, target = self._current_model(), self.app.model
        try:
            tol = float(self.tol.get())
            shown = [int(x) for x in self.sitemap.get().replace(",", " ").split()]
            automatic = site_maps(source, target, tol)[0]
            edited = bool(shown) and shown != list(automatic)
            align = self.align_var.get()
            model = equivalent_model(
                source, target, tol, mapping=shown if edited else None, align=align
            )
        except ValueError as error:
            messagebox.showerror(
                "Export transformed model",
                f"{error}." + ("\n\nCheck the site map." if self.sitemap.get() else ""),
                parent=self,
            )
            return
        filename = fdlg.asksaveasfilename(
            initialdir=self.app.datafolder,
            title="Save the transformed model",
            defaultextension=".cif",
            filetypes=(("cif files", "*.cif"), ("all files", "*.*")),
        )
        if not filename:
            return
        model.save_cif(filename)
        logger.info(
            "%s, in the cell and order of sites of the current model%s%s, "
            "saved to %s (%d sites, couplings %s).",
            name,
            " (with the edited site map)" if edited else "",
            ", aligned onto it" if align else "",
            filename,
            len(model.site_properties["coord_atomos"]),
            list(model.bonds),
        )

    def _ensure_model(self):
        """
        Make sure the model of the configurations has been chosen: if not,
        ask for it now (before, loading configurations or importing
        without choosing it first failed). Returns whether there is one.
        """
        if self._current_model() is None:
            # Triggers onmodelselect, which asks for the file.
            self.selected_model.set("[other model]")
        return self._current_model() is not None

    def close_window(self):
        """close the window"""
        self.destroy()

    def send_to_application(self):
        """Sends the new configs to the application."""
        logger.debug("send_to_application")
        self.app.spinconfigs.insert(
            END, "\n\n#  From " + self.selected_model.get() + "\n"
        )
        self.app.spinconfigs.insert(END, self.outputconfs.get(1.0, END))
        self.app.spinconfigs.insert(END, "\n" + 20 * "# " + "\n\n")
        self.app.reload_configs(src_widget=self.app.spinconfigs)
        logger.debug("configurations sent to the application")

    def configs_from_file(self):
        """
        Read configs from a model
        """
        logger.debug("configs from file")
        if not self._ensure_model():
            return
        filename = fdlg.askopenfilename(
            initialdir=self.app.datafolder + "/",
            title="Select file",
            filetypes=(("spin list", "*.spin"), ("all files", "*.*")),
        )
        if filename == "":
            return
        self.app.datafolder = str(Path(filename).parent)
        self.inputconfs.delete(1.0, END)
        with open(filename, "r") as filecfg:
            self.inputconfs.insert(END, "\n" + filecfg.read())
        logger.info("Configurations read from %s", filename)

    def map_confs(self):
        """
        Map the configurations of the other model (inputs) to the current
        one (outputs), with the site map shown (it can be edited).

        Each site of the current model takes the spin of its site in the
        other model, and the energies are scaled by the ratio of the
        numbers of sites. Configurations that the current cell cannot
        represent (not periodic with its lattice) are not imported: they
        are kept as comments, with the reason.
        """
        if not self._ensure_model():
            return
        name = self.selected_model.get()
        source, target = self._current_model(), self.app.model
        try:
            mapping = [int(x) for x in self.sitemap.get().replace(",", " ").split()]
            tol = float(self.tol.get())
            # Checks the site map once (sites without counterpart, size).
            map_configurations(source, [], target, tol, mapping=mapping)
        except ValueError as error:
            messagebox.showerror(
                "Import",
                f"{error}.\n\nEdit the site map, or increase the length tolerance.",
            )
            return
        num_sites = len(source.site_properties["coord_atomos"])
        lines, skipped = [], 0
        for line in self.inputconfs.get(1.0, END).split("\n"):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                lines.append(line)
                continue
            try:
                energies, confs, comments = parse_spin_configurations(stripped)
            except ValueError:
                lines.append("# not a configuration: " + stripped)
                skipped += 1
                continue
            conf, comment = confs[0], comments[0].strip()
            if len(conf) != num_sites:
                lines.append(
                    f"# skipped, {len(conf)} values for {num_sites} sites: {stripped}"
                )
                skipped += 1
                continue
            if not compatible_configurations(source, [conf], target, tol)[0]:
                lines.append(
                    "# skipped, not periodic with the current cell: " + stripped
                )
                skipped += 1
                continue
            (mapped,) = map_configurations(source, [conf], target, tol, mapping=mapping)
            energy = energies[0] * self.energy_scale
            lines.append(f"{energy}\t{mapped}\t# {comment}".rstrip())
        if skipped:
            logger.warning(
                "%d configurations of %s were not imported (see the comments "
                "in the imported list).",
                skipped,
                name,
            )
        self.outputconfs.config(state=NORMAL)
        self.outputconfs.delete(1.0, END)
        self.outputconfs.insert(END, "\n".join(lines).strip("\n") + "\n")
        self.outputconfs.config(state=DISABLED)
