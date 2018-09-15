
from .magnetic_model import *


def magnetic_model_from_file(filename,
                             magnetic_atoms=["Mn", "Fe", "Co",
                                             "Ni", "Dy", "Tb", "Eu"],
                             bond_names=None):
    if filename[-4:] == ".cif" or filename[-4:] == ".CIF":
        return magnetic_model_from_cif(filename, magnetic_atoms, bond_names)
    if filename[-7:] == ".struct" or filename[-7:] == ".STRUCT":
        return magnetic_model_from_wk2_struct(filename, magnetic_atoms,
                                              bond_names)
    eprint("unknown file format")
    return -1


def magnetic_model_from_cif(filename,
                            magnetic_atoms=["Mn", "Fe", "Co",
                                            "Ni", "Dy", "Tb", "Eu"],
                            bond_names=None):
    bravais_params = {}
    magnetic_positions = None
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []
    with open(filename, "r") as src:
        for line in src:
            listrip = line.strip()
            if listrip[:13] == '_cell_length_':
                varvals = listrip[13:].split()
                bravais_params[varvals[0]] = float(varvals[1])
            elif listrip[:12] == '_cell_angle_':
                varvals = line[12:].strip().split()
                bravais_params[varvals[0]] = float(varvals[1]) *\
                3.1415926 / 180.
            elif listrip[:5] == 'loop_':
                labels = []
                entries = []
                l = src.readline()
                listrip = l.strip()
                if listrip == '':
                    break
                if listrip != "" and line[0] == "#":
                    continue
                while listrip[0] == '_' or listrip[0] == '#':
                    if listrip != "" and line[0] == "#":
                        continue
                    labels.append(listrip.split()[0])
                    line = src.readline()
                    listrip = line.strip()
                while listrip != '':
                    entries.append(listrip.split())
                    l = src.readline()
                    listrip = l.strip()
                # if the block contains the set of atoms
                if '_atom_site_fract_x' in labels:
                    magnetic_positions = []
                    atomlabels = {}
                    for i, l in enumerate(labels):
                        if l == "_atom_site_fract_x":
                            xcol = i
                        elif l == "_atom_site_fract_y":
                            ycol = i
                        elif l == "_atom_site_fract_z":
                            zcol = i
                        elif l == "_atom_site_type_symbol":
                            tcol = i
                        elif l == "_atom_site_label":
                            labelcol = i

                    for i, entry in enumerate(entries):
                        if entry[tcol] in magnetic_atoms:
                            newpos = np.array([float(entry[xcol]),
                                               float(entry[ycol]),
                                               float(entry[zcol])])
                            magnetic_positions.append(newpos)
                            magnetic_species.append(entry[tcol])
                            atomlabels[entry[labelcol]] = i

                # If the block contains the set of bonds
                if '_geom_bond_atom_site_label_1' in labels:
                    jlabelcol = None
                    bondlists = []
                    bond_distances = []
                    bond_labels = {}
                    for i, l in enumerate(labels):
                        if l == "_geom_bond_atom_site_label_1":
                            at1col = i
                        if l == "_geom_bond_atom_site_label_2":
                            at2col = i
                        if l == "_geom_bond_distance":
                            distcol = i
                        if l == "_geom_bond_label":
                            jlabelcol = i
                    if jlabelcol is None:
                        for en in entries:
                            newbond = (atomlabels[en[at1col]],
                                       atomlabels[en[at2col]])

                            if en[distcol] not in bond_distances:
                                bond_distances.append(en[distcol])
                                lbl = "J" + str(len(bond_distances))
                                bond_labels[lbl] = len(bond_labels)
                                bondlists.append([])
                            bs = bond_distances.item(en[distcol])
                            bondlists[bs].append(newbond)
                    else:
                        for en in entries:
                            newbond = (atomlabels[en[at1col]],
                                       atomlabels[en[at2col]])
                            if bond_labels.get(en[jlabelcol]) is None:
                                bond_labels[en[jlabelcol]] = len(bondlists)
                                bond_distances.append(en[distcol])
                                bondlists.append([])
                            bondlists[
                                bond_labels[en[jlabelcol]]].append(newbond)
                    bond_labels = sorted([(bond_labels[la], la)
                                          for la in bond_labels],
                                         key=lambda x: x[0])
                    bond_labels = [x[1] for x in bond_labels]
            # Build the Bravai's vectors

        bravais_vectors = []
        if bravais_params.get('a') is not None:
            bravais_vectors.append(np.array([bravais_params.get('a'), 0, 0]))

        if bravais_params.get('b') is not None:
            gamma = bravais_params.get('gamma')
            if gamma is None:
                gamma = 3.1415926 * .5
            bravais_vectors.append(np.array([bravais_params.get('b') *
                                             np.cos(gamma),
                                             bravais_params.get('b') *
                                             np.sin(gamma), 0]))

        if bravais_params.get('c') is not None:
            alpha = bravais_params.get('alpha')
            beta = bravais_params.get('beta')
            if alpha is None:
                alpha = 3.1415926 * .5
            if beta is None:
                beta = 3.1415926 * .5
            x = np.cos(alpha)
            y = np.cos(beta) - x * np.cos(gamma)
            y = y / np.sin(gamma)
            z = bravais_params.get('c') * np.sqrt(1 - x * x - y * y)
            x = bravais_params.get('c') * x
            y = bravais_params.get('c') * y
            bravais_vectors.append(np.array([x, y, z]))

    magnetic_positions = np.array(magnetic_positions).dot(
        np.array(bravais_vectors))
    model = MagneticModel(magnetic_positions, bravais_vectors,
                          bond_lists=bondlists,
                          bond_names=bond_labels,
                          magnetic_species=magnetic_species)
    model.bond_distances = [float(d) for d in bond_distances]
    return model


def magnetic_model_from_wk2_struct(filename,
                                   magnetic_atoms=["Mn", "Fe", "Co",
                                                   "Ni", "Dy", "Tb", "Eu"],
                                   bond_names=None):
    bravais_params = {}
    magnetic_positions = []
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []

    with open(filename) as fin:
        title = fin.readline()
        fin.readline()         # size
        fin.readline()         # not any clue
        bravais = fin.readline()
        for l in fin:
            sl = l.strip()
            if sl[:4] == "ATOM":
                positions = []
                if sl[4] == " ":
                    sl = list(sl)
                    sl[4] = "-"
                    sl = "".join(sl)
                if sl[5] == " ":
                    sl = list(sl)
                    sl[5] = "-"
                    sl = "".join(sl)
                fields = sl.split()
                idxatom = fields[0][4:-1]
                positions.append([float(fields[1][3:]),
                                  float(fields[2][3:]),
                                  float(fields[3][3:])])
                mult = int(fin.readline().strip().split()[1])
                mult = mult - 1
                for k in range(mult):
                    sl = fin.readline()
                    fields = sl.split()
                    positions.append([float(fields[1][3:]),
                                      float(fields[2][3:]),
                                      float(fields[2][3:])])

                atomlabelfield = fin.readline().strip()
                if atomlabelfield[1] == " ":
                    atomspecies = atomlabelfield[0]
                else:
                    atomspecies = atomlabelfield[:2]
                atomlabel = atomspecies + idxatom
                lrm = fin.readline()              # Rotation matrix
                lrm = lrm + fin.readline()
                lrm = lrm + fin.readline()

                if atomspecies not in magnetic_atoms:
                    continue
                for p in positions:
                    magnetic_positions.append(p)
                    magnetic_species.append(atomspecies)

        bravais_fields = bravais.strip().split()
        bravais_params["a"] = float(bravais_fields[0])
        bravais_params["b"] = float(bravais_fields[1])
        bravais_params["c"] = float(bravais_fields[2])
        bravais_params["alpha"] = float(bravais_fields[3]) * 3.1415926 / 180
        bravais_params["beta"] = float(bravais_fields[4]) * 3.1415926 / 180
        bravais_params["gamma"] = float(bravais_fields[5]) * 3.1415926 / 180

        bravais_vectors = []
        if bravais_params.get('a') is not None:
            bravais_vectors.append(np.array([bravais_params.get('a'), 0, 0]))

        if bravais_params.get('b') is not None:
            gamma = bravais_params.get('gamma')
            if gamma is None:
                gamma = 3.1415926 * .5
            bravais_vectors.append(np.array([bravais_params.get('b') *
                                             np.cos(gamma),
                                             bravais_params.get('b') *
                                             np.sin(gamma), 0]))

        if bravais_params.get('c') is not None:
            alpha = bravais_params.get('alpha')
            beta = bravais_params.get('beta')
            if alpha is None:
                alpha = 3.1415926 * .5
            if beta is None:
                beta = 3.1415926 * .5
            x = np.cos(alpha)
            y = np.cos(beta) - x * np.cos(gamma)
            y = y / np.sin(gamma)
            z = bravais_params.get('c') * np.sqrt(1 - x * x - y * y)
            x = bravais_params.get('c') * x
            y = bravais_params.get('c') * y
            bravais_vectors.append(np.array([x, y, z]))

    magnetic_positions = np.array(magnetic_positions).dot(
        np.array(bravais_vectors))
    model = MagneticModel(magnetic_positions, bravais_vectors,
                          bond_lists=None,
                          bond_names=None,
                          magnetic_species=magnetic_species,
                          model_label=title)
    model.bond_distances = [float(d) for d in bond_distances]
    return model


def confindex(c):
    return sum([i * 2**n for n, i in enumerate(c)])


def read_spin_configurations_file(filename, model):
    configuration_list = []
    energy_list = []
    comments = []
    with open(filename, "r") as f:
        for l in f:
            ls = l.strip()
            if ls == "" or ls[0] == "#":
                continue
            fields = ls.split(maxsplit=1)
            energy = float(fields[0])
            ls = fields[1]
            newconf = []
            comment = ""
            for pos, c in enumerate(ls):
                if c == "#":
                    comment = ls[(pos + 1):]
                    break
                elif c == "0":
                    newconf.append(0)
                elif c == "1":
                    newconf.append(1)
            while len(newconf) < model.cell_size:
                eprint("filling empty places")
                newconf.append(0)
            comments.append(comment)
            configuration_list.append(newconf)
            energy_list.append(energy)
    return (energy_list, configuration_list, comments)
