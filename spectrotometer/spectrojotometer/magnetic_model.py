
# coding: utf-8


from __future__ import print_function
import sys

import numpy as np
from .tools import *
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3
#from matplotlib.lines import Line2D





########################################################################33





class MagneticModel:
    def __init__(self,atomic_pos, bravais_lat, 
                 bond_lists=None,
                 bond_names=None,
                 ranges=None,
                 supercell_size=2,
                 discretization=0.02,
                 magnetic_species=None,
                 onfly=True, model_label="default"):
        self.model_label = model_label
        self.cell_size = len(atomic_pos)
        self.coord_atomos = atomic_pos
        self.bravais_vectors = bravais_lat
        self.onfly = onfly

        if magnetic_species is None:
            self.magnetic_species = ["X" for i in atomic_pos]
        else:
            self.magnetic_species = []
            self.magnetic_species[:] = magnetic_species
        if len(bravais_lat) == 1:
            self.supercell = np.array([[site[idx] + i * self.bravais_vectors[0][idx]
                                       for idx in range(3)]
                            for i in range(-supercell_size,supercell_size+1)
                            for site in self.coord_atomos ])

        elif len(bravais_lat)== 2:
            self.supercell = np.array([[site[idx] + i * self.bravais_vectors[0][idx]
                                        + j * self.bravais_vectors[1][idx] 
                                       for idx in range(3)]
                            for i in range(-supercell_size,supercell_size+1)
                            for j in range(-supercell_size,supercell_size+1)
                            for site in self.coord_atomos ])
        elif len(bravais_lat) == 3:
            self.supercell = np.array([[site[idx] + i * self.bravais_vectors[0][idx]
                                        + j * self.bravais_vectors[1][idx] 
                                        + k * self.bravais_vectors[2][idx] 
                                       for idx in range(3)]
                            for i in range(-supercell_size,supercell_size+1)
                            for j in range(-supercell_size,supercell_size+1)
                            for k in range(-supercell_size,supercell_size+1)
                            for site in self.coord_atomos ])
        else:
            self.supercell = self.coord_atomos
        
        if ranges is None:
            maxdist = 0 
            for p in self.coord_atomos:
                for q in self.coord_atomos:
                    d = np.linalg.norm(p-q)
                    if d > maxdist:
                        maxdist = d
            ranges = [[0,d]]
        elif type(ranges) is float:
            ranges = [[0,ranges]]
        elif type(ranges) is list and type(ranges[0]) is float:
            ranges = [ranges]

        if bond_lists is not None:            
            self.bond_lists = bond_lists
            bond_distances = [0 for i in bond_lists]
            if bond_names is not None:
                self.bond_names = bond_names
            else:
                self.bond_names =[ "J"+str(i) for i in range(len(self.bond_lists))]
        else:
            self.bond_names = []
            if bond_names is not None:
                self.bond_names[:] = bond_names            
            self.bond_lists =[]
            self.bond_distances = []
            self.discretization = discretization
            self.generate_bonds(discretization,ranges)
            

            
    def remover_bond(self,idx):
        """
        Elimina el idx-esimo bond
        """
        self.bond_distances.pop(idx)
        self.bond_names.pop(idx)
        self.bond_lists.pop(idx)
        
    def remover_bond_by_name(self,name):
        """
        Elimina el bond de nombre <name>
        """
        try:
            idx = self.bond_names.index(name)
        except ValueError:
            eprint("bond "+ name + " not found")
            return
        
        self.remover_bond(idx)
        
        
    def generate_bonds(self, discretization, ranges, bond_names=None):
        """
        En base a la celda unidad, la red de Bravais y el tamaño de la 
        "super-celda" (cuantas copias de la celda unidad se consideran 
        como entorno) calcula los bonds,
        esto es, los pares de sitios magnéticos que interactúan según cada 
        una de las constantes de acoplamiento a determinar.
        """
        cell_size = self.cell_size 
        coord_atomos = self.coord_atomos
        bravais_vectors = self.bravais_vectors
        supercell = self.supercell
        old_bond_lists = self.bond_lists 
        old_bond_distances = self.bond_distances 
        old_bond_names = self.bond_names
        
        bond_lists = []
        bond_distances = []
        bond_type = []
        bond_names = []
        atom_type = self.magnetic_species
        
        def is_in_range(val):
            for r in ranges:
                if r[0]<val<r[1]:
                    return True
            return False
        
        for d, bt in sorted([(np.linalg.norm(q-p), set([atom_type[i],atom_type[j]]))
                             for i,q in enumerate(self.coord_atomos) 
                                             for j,p in enumerate(self.coord_atomos)]):
            dr = round(d/discretization)*discretization
            if not is_in_range(dr): 
                continue
            if dr!=0 and (dr not in bond_distances or bt not in bond_type ):
                bond_distances.append(dr)
                bond_lists.append([])
                bond_type.append(bt)
                    
        #supercell=np.array([p for p in supercell if p[2]>0])
        for p,x in enumerate(coord_atomos):        
            for q,y in enumerate(self.supercell):
                qred = q%cell_size            
                if(p<qred):                
                    d = x-y
                    d = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
                    if not is_in_range(d):
                        continue
                    bt = set([atom_type[p],atom_type[qred]])
                    for i in range(len(bond_distances)):
                        if np.abs(d-bond_distances[i])<discretization and bt == bond_type[i]:
                            bond_lists[i].append((p,qred))
                            
        self.bond_lists =  old_bond_lists +  bond_lists
        self.bond_distances =  old_bond_distances +  bond_distances

        
        nnames = len(self.bond_names)
        while len(self.bond_names)<len(self.bond_lists):
            while "J"+str(nnames+1) in self.bond_names:
                nnames = nnames + 1
            self.bond_names.append("J"+str(nnames+1))            
        return
            
    def formatted_equations(self,cm,ensname=None, comments=None,format="plain"):
        res = ""
        if format == "latex":
            times_symbol = ""
        else:
            times_symbol = "*"
        if format == "plain" or format == "latex":
            equal_symbol = "="
        else:
            equal_symbol = "=="

        if format == "plain":
            open_comment = "# "
            close_comment = ""
        elif format == "latex":
            open_comment = "% "
            close_comment = ""
        elif format == "wolfram":
            open_comment =  "(*"
            close_comment = "*)"
        else:
            open_comment = "#"
            close_comment = ""

        if format == "latex":
            sub_symb = "_"
        else:
            sub_symb = ""

        jsname = []
        jsname[:] = self.bond_names
        jsname.append("E" +  sub_symb +"0")
        if ensname is None:
            ensname = [ "E"+ sub_symb +str(i+1) for i in range(len(cm))]
        for i,row in enumerate(cm):
            eq=""
            for k,c in enumerate(row):
                cr = round(c*100)/100.
                if c>0:
                    if eq != "":
                        eq = eq + " + "
                    else:
                        eq = "  "
                    if cr != 1:
                        eq = eq +  str(cr)  +  " " + times_symbol
                    eq = eq + " " + jsname[k]
                elif c<0:
                    if eq != "":
                        eq = eq + " "
                    eq = eq + "- "
                    if cr != -1:
                        eq = eq +  str(-cr)  +  " "+ times_symbol
                    eq = eq + " " + jsname[k]
            if comments is not None:
                res = res + eq + equal_symbol + ensname[i]  + "  " + open_comment + comments[i] +  close_comment + "\n\n"
            else:
                res = res + eq + equal_symbol + ensname[i]   + "\n\n"
        return res



        
    def print_equations(self,cm,ensname=None, comments=None,format="plain"):
        print("\n\n# Equations: \n============\n\n")
        print(self.formatted_eqnations(self,cm,ensname, comments,format))

    def coefficient_matrix(self,configs,normalizar=True):
        """
        Devuelve la matriz que define el sistema de ecuaciones que vincula a las constantes
        de acoplamiento con las energías correspondientes a las configuraciones dadas.
        """
        rawcm = [np.array([
                         -sum([(-1)**sc[b[0]] * (-1)**sc[b[1]] for b in bondfamily ]) 
                         for sc in configs]) 
                        for bondfamily in self.bond_lists]
        if normalizar:
            cm = [ v-np.average(v) for v in rawcm]
        else:
            cm = rawcm
        cm = np.array(cm + 
                      [np.array([1. for sc in configs]) # Energía no magnética
                      ]).transpose()
        return cm

    def inv_min_sv_from_config(self,confs,boxsize=True):
        """
        Dada una lista de configuraciones de espin, construye las ecuaciones y calcula la inversa del mínimo valor singular.
        Las ecuaciones son de la forma
        \Sum_{(ij)} J_{ij} S_i[c] S_j[c] = E_c)
        Aprovechamos que en la construcción de las ecuaciones garantizamos que las ecuaciones para
        la E0 y los js queden desacopladas (al restar en cada coeficiente el promedio sobre las configuraciones)
        """
        if len(confs) == 0 :
            return 1e80
        eqarray = self.coefficient_matrix(confs)
        eqarray = eqarray[:,0:-1]
        if boxsize:
            return max(box_ellipse(eqarray,1 ))
        singularvalues = np.linalg.svd(eqarray)[1]
        if min(singularvalues) == 0 or len(singularvalues) < len(eqarray[0]):
            return 1e80
        cond_number = 1./min(singularvalues)
        return cond_number
    

    def compute_couplings(self, confs, energs, err_energs=.01,printeqs=False):
        """
        compute_couplings
        Given a set of configurations, and the energies calculated from 
        the ab-initio tools, estimates the values of the coupling 
        constants from the proposed model.
        
        confs: list of magnetic configurations
        energs: energies evaluated for each magnetic configuration
        err_energs: estimation of the maximum convergence error in energies
        printeqs: print the corresponding equations.
        """

        if printeqs:
            coeffs = self.coefficient_matrix(confs,normalizar=False)
            eprint("\n# Configurations:\n=================\n\n")
            for c in confs:
                print(c)
            self.print_equations(coeffs)

        coeffs = self.coefficient_matrix(confs,normalizar=True)

        # The choice in the way the equation is written allows to decouple
        # the determination of the coupling constants from the base
        # energy. This implies that the condition number associated to the
        # coupling constants should be evaluated from the reduced set of 
        # equations.
        rcoeffs = coeffs[:,0:-1]
        singularvalues = np.linalg.svd(rcoeffs)[1]
        cond_number = 1./min(singularvalues)
        if printeqs:
            eprint("\nInverse of the minimum singular value: ", cond_number,"\n\n")

        # js = (A^t A )^{-1} A^t En, i.e. least squares solution
        resolvent =  np.linalg.inv(coeffs.transpose().dot(coeffs)).dot(coeffs.transpose())
        js = resolvent.dot(energs)
        model_chi = abs(coeffs.dot(js)-energs)/err_energs
        rr = (len(model_chi) - sum(model_chi**2))
        if rr <0:
            deltaJ = [-1 for i in js]
        else:
            rr = np.sqrt(rr) * err_energs
            deltaJ = box_ellipse(2.*np.sqrt(len(coeffs))*coeffs, rr)
        return (js, deltaJ, model_chi)

    # def show_config(self, config, sp):
    #     """
    #     Esta función dibuja las redes con las correspondientes configuraciones de 
    #     espines.

    #        config es una lista con los estados de los átomos
    #        sp es un subplot donde hacer el gráfico
    #        si showbonds = True, dibuja lineas entre los sitios interactuantes
    #     """
    #     coord_atomos = self.coord_atomos
    #     fig = plt.figure()
    #     colors = ['r' if config[i] >0 else 'b' for i in range(self.cell_size)]
    #     ax = fig.add_subplot(sp, projection='3d')
    #     idx=0
    #     for i in range(len(coord_atomos)):
    #         p = coord_atomos[i]
    #         ax.text(p[0], p[1], p[2], s=i+1)
    #     ax.scatter(coord_atomos[:,0],coord_atomos[:,1],coord_atomos[:,2],c=colors)
    #     ax.set_ylim((-1,1))
    #     ax.set_xlim((-1,1))       
    #     return (fig,ax)

    # def show_superlattice_bonds(self, nd=1.,bt=None,discretization=.02):
    #     """
    #     Muestra los sitios que interactúan entre sí a una cierta distancia.
    #     """
    #     coord_atomos = self.coord_atomos
    #     supercell = self.supercell
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)


    #     if bt is not None:
    #         nd=bond_distances[bt]
    #     colors=["r" for p in supercell]
    #     ax.scatter(supercell[:,0],supercell[:,1],c=colors)
    #     colors=["b" for p in coord_atomos]
    #     ax.scatter(coord_atomos[:,0],coord_atomos[:,1],c=colors)
    #     ax.set_ylim((-3,3))

    #     for p,x in enumerate(coord_atomos):
    #         for q,y in enumerate(supercell):
    #             qred = q%cell_size            
    #             if(p<qred):                
    #                 d = x-y
    #                 d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
    #                 #print([(p,q),[np.abs(d-z) for z in [d0,d1p,d1x,d2p,d2x,d3p,d3x]]])
    #                 if np.abs(d-nd)<.01:
    #                     ax.add_artist(Line2D([x[0],y[0]],[x[1],y[1]]))
    #             elif(p>qred):
    #                 d = x-y
    #                 d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
    #                 if d > 4.5:
    #                     continue
    #                 if np.abs(d-nd)<discretization:
    #                     ax.add_artist(Line2D([x[0],y[0]],[x[1],y[1]],linestyle=":"))
    #             elif(p==qred):
    #                 d = x-y
    #                 d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
    #                 if np.abs(d-nd)<discretization:
    #                     ax.add_artist(Line2D([x[0],y[0]],[x[1],y[1]],linestyle="-."))
    #     return 


    # def check_superlattice_bonds(self, nd=1.,bt=None,discretization=.02):
    #     """
    #     Muestra los sitios que interactúan entre sí a una cierta distancia.
    #     """
    #     coord_atomos = self.coord_atomos
    #     supercell = self.supercell
    #     if bt is not None:
    #         nd = self.bond_distances[bt]
    #     else:
    #         for k,di in self.bond_distances:
    #             if np.abs(d-nd)<discretization:
    #                 bt=k
    #                 break
    #     if bt is None:
    #         return 

    #     countbond=0
    #     for p,x in enumerate(coord_atomos):
    #         for q,y in enumerate(supercell):
    #             qred = q%cell_size            
    #             if(p<qred):                
    #                 d = x-y
    #                 d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
    #                 if np.abs(d-nd)<discretization:
    #                     countbond = countbond+1
    #                     if (p,qred) not in self.bond_lists[bt]:
    #                         eprint ((p,qred), "missing")
    #             elif(p>qred):
    #                 d = x-y
    #                 d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
    #                 if 9.9 > d > .7:
    #                     continue
    #                 if np.abs(d-nd)<discretization:
    #                     countbond = countbond + 1
    #                     if (qred,p) not in self.bond_lists[bt]:
    #                         eprint ((qred,p), "missing")
    #     if 2*len(bond_lists[bt]) != countbond:
    #         eprint (2*len(bond_lists[bt])," != ", countbond)
    #     return 

    def generate_configurations_onfly(self):
        size = self.cell_size
        for c in range(2**((size-1))):
            yield [c >> i & 1 for i in range(size-1,-1,-1)]

    def generate_random_configurations(self,t=10):
        size = self.cell_size
        for c in np.random.random_integers(0,2**((size-1)),t):
            yield [c >> i & 1 for i in range(size-1,-1,-1)]
    
    
    def find_optimal_configurations(self, start=None, num_new_confs=None, known=None, its=100, update_size=1):        
        if known is None:
            known = []
        
        if num_new_confs is None or num_new_confs + len(known) < len(self.bond_lists) + 1:
            num_new_confs = max(len(self.bond_lists)-len(known) +1 ,1)
        if start is None:
            repres = self.generate_random_configurations(2*num_new_confs)
            last_better = normalize_configurations([q for q in repres])                                    
        else:            
            last_better = start

        num_confs = num_new_confs + len(known)
        inequiv_confs = last_better
        last_better_cn = self.inv_min_sv_from_config(last_better)
        cn = last_better_cn
        
        
        for it in range(its):

            #Dadas las configuraciones equivalentes, busca un subconjunto que optimize la dependencia 
            # de la energía con los parámetros

            #inequiv_confs=[ocho_a_diezyseis(c) for c in repres]

            #print("generando matriz de coeficientes para las configuraciones inequivalentes")
            #inequiv_confs=repres

            # Aquí se calculan los coeficientes del sistema de ecuaciones asociado al conjunto completo de configuraciones 
            # no equivalentes

            repres = self.generate_random_configurations(update_size)
            inequiv_confs =  known + [q for q in repres] + inequiv_confs
            inequiv_confs = normalize_configurations(inequiv_confs)
            coefs = self.coefficient_matrix(inequiv_confs)
            
            #Este es el algoritmo que uso para buscar las óptimas:
            # Descompongo coefs como SVD, y me quedo sólo con los vectores asociados a los valores singulares no nulos.
            # Luego, busco cuales configuraciones definen el soporte efectivo de los vectores singulares.
            v=((np.linalg.svd(coefs)[0])[0:len(coefs[0])]).transpose()
            v = v[len(known):]
            # Busco definir un soporte efectivo sobre los vectores singulares (que dan 
            # los índices en la lista de configuraciones).
            threshold=sorted([np.linalg.norm(z) for z in v])[-num_new_confs]

            # relevant guarda las configuraciones que lucen relevantes (porque tienen mayor peso en los 
            # vectores singulares)
            relevant=[]
            for j,val in enumerate(v):
                if np.linalg.norm(val)>=threshold:
                    if j not in relevant:
                        relevant.append(j)

            # Ordeno los índices de configuraciones relevantes y elimino duplicados.
            relevant = sorted(relevant)[:num_new_confs]
            relevant =  [k + len(known) for k in relevant]
            inequiv_confs = [inequiv_confs[k] for k in relevant]
            cn = self.inv_min_sv_from_config( normalize_configurations(known + inequiv_confs))
            # a partir de los índices, fabrico una lista más corta de configuraciones relevantes
            #print("Número de condición para el conjunto reducido de ",len(inequiv_confs)," elementos:",cn)
            if cn < last_better_cn:
                last_better_cn = cn
                last_better = inequiv_confs 
                eprint("it", it,"new 1/|A^{+}|=", cn)
                eprint(inequiv_confs)
            else:
                inequiv_confs = last_better

        cn = last_better_cn
        inequiv_confs = last_better        

        return(last_better_cn, last_better)



    def save_cif(self, filename, bond_names=None):
        bravais_vectors = self.bravais_vectors
        
        with open(filename,"w") as fileout:
            head = """
#======================================================================
            
# CRYSTAL DATA

#----------------------------------------------------------------------

data_magnetic_model_1
            
_chemical_name_common                  """ + self.model_label + "\n"
            fileout.write(head)
            bbn = ["a","b","c"]
            bbnang = ["alpha","beta","gamma"]
            
            if len(bravais_vectors) == 1:
                fileout.write("_cell_length_a \t\t\t" + str(np.linalg.norm(bravais_vectors[0]))+"\n\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz\t\t\t\n\'z\'\n")
            elif len(bravais_vectors) == 2:
                a = np.linalg.norm(bravais_vectors[0])
                b = np.linalg.norm(bravais_vectors[1])
                gamma = round(180/3.1415926 * bravais_vectors[0].dot(bravais_vectors[1])/(a*b))
                fileout.write("_cell_length_a \t\t\t" + str(a)+"\n")
                fileout.write("_cell_length_b \t\t\t" + str(b)+"\n")
                fileout.write("_cell_length_gamma \t\t\t" + str(gamma)+"\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz\n\'x, y\'\n\n")
                
            elif len(bravais_vectors) == 3:
                a = np.linalg.norm(bravais_vectors[0])
                b = np.linalg.norm(bravais_vectors[1])
                c = np.linalg.norm(bravais_vectors[2])
                gamma =round(180/3.1415926 * np.arccos(bravais_vectors[0].dot(bravais_vectors[1])/(a*b)))
                alpha =round(180/3.1415926 * np.arccos(bravais_vectors[0].dot(bravais_vectors[2])/(a*c)))
                beta = round(180/3.1415926 * np.arccos(bravais_vectors[0].dot(bravais_vectors[2])/(c*b)))
                fileout.write("_cell_length_a \t\t\t" + str(a)+"\n")
                fileout.write("_cell_length_b \t\t\t" + str(b)+"\n")
                fileout.write("_cell_length_c \t\t\t" + str(c)+"\n")
                fileout.write("_cell_angle_alpha \t\t\t" + str(alpha)+"\n")
                fileout.write("_cell_angle_beta \t\t\t" + str(beta)+"\n")
                fileout.write("_cell_angle_gamma \t\t\t" + str(gamma)+"\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz\t\t\t\n\'x, y, z\'\n\n")
            
            fileout.write("# Atom positions \n\n")
            fileout.write("loop_\n   _atom_site_label\n" + \
                          "_atom_site_occupancy\n" +  \
                          "_atom_site_fract_x\n" +  \
                          "_atom_site_fract_y\n" + \
                          "_atom_site_fract_z\n" + \
                          "_atom_site_adp_type\n" + \
                          "_atom_site_B_iso_or_equiv\n" + \
                          "_atom_site_type_symbol\n")

            bravaiscoords = self.coord_atomos.dot(np.linalg.inv( 
                                                    np.array(bravais_vectors)))
                                      
            for i,pos in enumerate(bravaiscoords):
                fileout.write(self.magnetic_species[i] + str(i+1) + "\t 1.\t" + \
                               str(round(1E5*pos[0])*1E-5) + "\t" +  \
                               str(round(1E5*pos[1])*1E-5) + "\t" + \
                               str(round(1E5*pos[2])*1E-5) + "\t" + \
                               "Biso \t" + \
                               "1 \t" + \
                               self.magnetic_species[i] + " \n")
            fileout.write("   \n")
            
            if len(self.bond_lists)>0 :
                fileout.write("# Bonds  \n")
                fileout.write("loop_\n")
                fileout.write("_geom_bond_atom_site_label_1\n")
                fileout.write("_geom_bond_atom_site_label_2\n")
                fileout.write("_geom_bond_distance\n")
                fileout.write("_geom_bond_label\n")                                                
                for k,bl in enumerate(self.bond_names):
                    for a,b in self.bond_lists[k]:
                        fileout.write(self.magnetic_species[a]+str(a+1) +"\t" +\
                                      self.magnetic_species[b]+str(b+1) +"\t" +\
                                      str(self.bond_distances[k]) +"\t" +\
                                      str(bl) +"\n")
                fileout.write("   \n")
        return True
            
            
            
            
