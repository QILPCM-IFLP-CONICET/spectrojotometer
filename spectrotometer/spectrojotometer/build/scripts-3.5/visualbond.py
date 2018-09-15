from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
#from tkmessagebox import *
import os
import tempfile
import sys
import numpy as np

from spectrojotometer.magnetic_model import  MagneticModel
from spectrojotometer.model_io import  magnetic_model_from_file, read_spin_configurations_file, confindex





#----------------------------------------------------

quote = """# BONDS GENERATOR 0.0: 
# 1-Please open a .CIF file with the site positions to start....
# 2- Enter the parameters and Press Generate model to define effective couplings...
# 3- Press optimize configurations in order to determine the optimal configurations for the ab initio calculations
#4- With the ab-initio energies press "Calculate model parameters" ..
"""

textmarkers = {}

textmarkers["separator_symbol"]={"latex":"","plain":"","wolfram":", ",}
textmarkers["Delta_symbol"]={"latex":"\Delta","plain":"Delta","wolfram":"\[Delta]",}
textmarkers["times_symbol"]={"latex":"","plain":"*","wolfram":"*",}
textmarkers["equal_symbol"]={"latex":"=","plain":"=","wolfram":"==",}
textmarkers["open_comment"]={"latex":"% ","plain":"# ","wolfram":"(*",}
textmarkers["close_comment"]={"latex":"","plain":"","wolfram":"*)",}
textmarkers["sub_symbol"]={"latex":"_","plain":"","wolfram":"",}
textmarkers["plusminus_symbol"]={"latex":"\pm","plain":"+/-","wolfram":"\[PlusMinus]",}





def validate_pinteger(action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if text in '0123456789':
            return True
        else:
            return False


def validate_float(action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if text in '0123456789.-+':
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False



class ImportConfigWindow(Toplevel):
    def __init__(self,app):
        self.app = app
        Toplevel.__init__(self,app.root)
        #self.root = Toplevel(app.root)
        self.transient(app.root)
        self.model = app.model
        self.configurations = ([],[],[])
        self.models = {}
        self.parameters = {}
        
        controls1 = LabelFrame(self,text="Parameters", padx=5, pady=5)

        row = Frame(controls1)
        Label(row,text="model file:").pack(side=LEFT)
        self.selected_model = StringVar()
        self.selected_model.trace('w', self.onmodelselect)        
        self.optmodels = OptionMenu(row, self.selected_model,"[other model]")
        self.optmodels.pack(side=LEFT,fill=X)        
        row.pack(side=TOP,fill=X)
        
        row = Frame(controls1)
        Label(row,text="Length tolerance:").pack(side=LEFT)
        self.tol = Entry(row, validate='key', validatecommand=self.app.vcmdf)
        self.tol.insert(10,.1)
        self.tol.pack(side=LEFT)        
        row.pack(side=TOP,fill=X)
        controls1.pack(side=TOP,fill=X)
        
        #controls2 = Frame(self, padx=5, pady=5)
        controls2 = PanedWindow(self,orient=HORIZONTAL)
        controls2.pack(side=TOP,fill=BOTH,expand=1)
        controls2l = LabelFrame(controls2, text="inputs", padx=5, pady=5)
        self.inputconfs = ScrolledText(controls2l, height=10, width=80)
        self.inputconfs.pack()
        buttons = Frame(controls2l)
        Button(buttons,text="Load Configuration from File",command=self.configs_from_file).pack(side=RIGHT)
        Button(buttons,text="Import",command=self.map_confs).pack(side=LEFT)
        buttons.pack(side=BOTTOM,fill=X)
        
        #controls2l.pack(side=LEFT,fill=Y)
        controls2.add(controls2l)
        controls2r = LabelFrame(controls2, text="in main model", padx=5, pady=5)
        self.outputconfs = ScrolledText(controls2r, height=10, width=80)
        self.outputconfs.config(state=DISABLED)
        self.outputconfs.pack()

#        controls2r.pack(side=RIGHT,fill=Y)
        controls2.add(controls2r)
        #controls2.pack(side=TOP,fill=X)
        
        framebts =  Frame(self)
        Button(framebts,text="Send to main application",command=self.send_to_application).pack(side=LEFT)
        Button(framebts,text="Close",command=self.close_window).pack(side=RIGHT)
        framebts.pack(side=BOTTOM,fill=X)
        self.grab_set()
        app.root.wait_window(self)
        
    def onmodelselect(self,*args):
        if self.selected_model.get()=="[other model]":
            filename = filedialog.askopenfilename(initialdir = self.app.datafolder+"/",
                                                  title = "Select file to open",
                                                  filetypes = (("cif files","*.cif"),
                                                               ("Wien2k struct files","*.struct"),
                                                               ("all files","*.*"))
            )
            print(filename)
            newmodel = magnetic_model_from_file(filename=filename)
            modellabel = filename
            self.models[filename] = newmodel
            menu = self.optmodels["menu"]
            menu.delete(0, "end")
            menu.add_command(label="[other model]",
                             command=lambda value="[other model]":self.selected_model.set("[other model]"))
            print("Updating menu")
            for key in self.models:
                menu.add_command(label=key,
                                 command=lambda value=key:self.selected_model.set(value))
            self.selected_model.set(filename)
            #self.optmodels.set(filename)
        
    def close_window(self):
        self.destroy()

    def send_to_application(self):        
        self.app.spinconfigs.insert(END,"\n\n# From " + self.selected_model.get()+ "\n")
        self.app.spinconfigs.insert(END, self.outputconfs.get(1.0,END))
        self.app.spinconfigs.insert(END,"\n#################\n\n")        
        self.app.reload_configs(src_widget=self.app.spinconfigs)

    def configs_from_file(self):
        if self.selected_model.get() ==  "[other model]":
            self.onmodelselect()
        filename = filedialog.askopenfilename(initialdir = self.app.datafolder+"/",
                                          title = "Select file",
                                          filetypes = (("spin list","*.spin"),("all files","*.*"))
        )
        self.inputconfs.delete(1.0,END)
        with open(filename,"r") as filecfg:            
            self.inputconfs.insert(END,"\n"+ filecfg.read())

    def map_confs(self):
        tol = float(self.tol.get())
        if self.selected_model.get() ==  "[other model]":
            self.onmodelselect()
        model1 = self.models[self.selected_model.get()]
        model2 = self.app.model
        size1 = len(model1.coord_atomos)
        scale_energy = float(len(model2.coord_atomos))/float(size1)        
        if scale_energy < 1. :
            messagebox.showinfo("Different sizes","# alert: unit cell in model2 is smaller than in model1.")
            
        dictatoms = [-1 for p in model2.coord_atomos]
        for i,p in enumerate(model2.coord_atomos):
            for j,q in enumerate(model1.supercell):
                if np.linalg.norm(p-q)<tol:
                    dictatoms[i] = j % size1
                    break
        self.outputconfs.config(state=NORMAL)         
        lines = self.inputconfs.get(1.0,END).split("\n")
        for line in lines:
            linestrip = line.strip()
            if linestrip == "" or linestrip[0] == "#" :
                self.outputconfs.insert(END,line+"\n")
                continue

            fields = line.strip().split(maxsplit=1)
            if len(fields)==1:
                self.outputconfs.insert(END, "#" +line+"\n")
                continue                
            fields = [str(float(fields[0])*scale_energy), fields[1].strip().split(sep="#",maxsplit=1)]
            if len(fields[1])==1:
                fields = [fields[0],fields[1][0], ""]
            else:
                fields = [fields[0],fields[1][0], fields[1][1]]

            configtxt = fields[1]
            config = []
            for c in configtxt:
                if c == '0':
                    config.append(0)
                if c == '1':
                    config.append(1)
            if len(config)< len(dictatoms):
                config = config + (len(dictatoms) - len(config))*[0]
            config = [ config[i] if i>=0 else 0 for i in dictatoms]
            newline = str(fields[0])  + "\t" + str(config) + "  # " + fields[2] + "\n"
            self.outputconfs.insert(END,newline)
        self.outputconfs.config(state=DISABLED)         
    

    


        


class ApplicationGUI:
    def __init__(self):        
        sys.stdout = self
        sys.stderr = self
        self.model = None
        self.configurations = ([],[],[])
        self.root = Tk()
        self.vcmdi = (self.root.register(validate_pinteger),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.vcmdf = (self.root.register(validate_float),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.root.title("Bonds generator 0.0")
        self.datafolder=os.getcwd()
        self.tmpmodel = tempfile.NamedTemporaryFile(mode="w",suffix=".cif")
        self.tmpmodel.close()
        self.tmpconfig = tempfile.NamedTemporaryFile(mode="w",suffix=".spin")
        self.tmpconfig.close()
        self.buildmenus()
        Frame(height=5, bd=1, relief=SUNKEN).pack(fill=X, padx=5, pady=5)
        self.status = ScrolledText(self.root, height=10, width=170)
        Frame(height=5, bd=1, relief=SUNKEN).pack(fill=X, padx=5, pady=5)
        self.status.config(background="black",foreground="white")
        self.status.pack()
        self.nb=ttk.Notebook(self.root)
        self.parameters = {}
        self.outputformat =  StringVar()
        self.outputformat.set("plain")
        self.build_page1()
        self.build_page2()
        self.build_page3()
#        self.build_page4()
        self.nb.pack(expand=1,fill="both")
        self.statusbar = Label(self.root, text="No model loaded.", bd=1,relief=SUNKEN,anchor=W)
        self.statusbar.pack(side=BOTTOM, fill=X)
        self.root.mainloop()

    def buildmenus(self):
        self.menu = Menu(self.root) #defino un menu desplegable
        self.root.config(menu=self.menu) # ??
        filemenu = Menu(self.menu)
        self.menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open model file ...", command=self.import_model)
        filemenu.add_command(label="Open config file ...", command=self.import_configs)
        filemenu.add_command(label="Save model as ...", command=self.save_model)
        filemenu.add_command(label="Save configurations as ...", command=self.save_configs)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        helpmenu = Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.about)

    def build_page1(self):
        self.parameters["page1"] = {}
        self.page1 = Frame(self.nb)
        controls = Frame(self.page1,width=50)
        # Controls for bonds
        controls2 = LabelFrame(controls,text="Add bonds", padx=5, pady=5)
        fields = ['Discretization', 'rmin', 'rmax']
        defaultfields = ["0.02", "0.0", "4.9"]
        for i,field in enumerate(fields):
            row = Frame(controls2)
            lab = Label(row, width=12, text=field + ": ", anchor='w')
            ent = Entry(row, validate='key', validatecommand=self.vcmdf)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            print(defaultfields[i])
            ent.insert(0,defaultfields[i])
            self.parameters["page1"][field] = ent
            print(ent.get())
        btn = Button(controls2,text="add bonds",command=self.add_bonds)
        btn.pack(side=BOTTOM)
        controls2.pack(side=TOP)

        controls1 = LabelFrame(controls,text="Grow lattice", padx=5, pady=5)
        fields = ['Lx', 'Ly', 'Lz']
        defaultfields = ("1", "1","1")
        valorentry=[]
        for i,field in enumerate(fields):
            row = Frame(controls1)
            lab = Label(row, width=12, text=field + ": ", anchor='w')
            ent = Entry(row, validate='key', validatecommand=self.vcmdi)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(10,defaultfields[i])
            self.parameters["page1"][field] = ent
        btn = Button(controls1,state=DISABLED,text="Grow unit cell",command=self.grow_unit_cell)
        btn.pack(side=BOTTOM)
        controls1.pack()

        controls.pack(side=LEFT,fill=Y)

        self.modelcif = ScrolledText(self.page1,width=150)
        self.modelcif.bind("<FocusOut>", self.reload_model  )
        self.modelcif.pack(side=RIGHT,fill=Y)
        self.modelcif.insert(END, quote)
        
        page1tools = Frame(self.page1)
        self.nb.add(self.page1,text="1. Define Model")

    def build_page2(self):
        self.parameters["page2"] = {}
        self.page2 = Frame(self.nb)

        controls = Frame(self.page2)
        
        # Controls for loading configurations
        controls1 = LabelFrame(controls,text="Load", padx=5, pady=5)
        btn = Button(controls1,text="Load configs from other model",
                     command=self.configs_from_other_model)
        btn.pack()
        controls1.pack(side=TOP,fill=X)
        # Controls for Optimize configurations
        controls2 = LabelFrame(controls,text="Optimize", padx=5, pady=5)
        fields = ['Number of configurations', 'Bunch size', 'Iterations']
        defaultfields = ("10", "10", "100")
        for i,field in enumerate(fields):
            row = Frame(controls2)
            lab = Label(row, width=12, text=field + ": ", anchor='w')
            ent = Entry(row,validate='key', validatecommand=self.vcmdi)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(10,defaultfields[i])
#            ents[field] = ent
            self.parameters["page2"][field] = ent
        btn = Button(controls2,text="Optimize",
                     command=self.optimize_configs)
        btn.pack()
        controls2.pack(side=TOP,fill=X)
        controls3 = LabelFrame(controls,text="Format", padx=5,pady=5)
        row = Frame(controls3)
        lab = Label(row,width=12,text="Format", anchor="w")
        lab.pack(side=LEFT)
        optformat = OptionMenu(row, self.outputformat, "plain", "latex", "wolfram", command=self.print_full_equations)
        optformat.pack(side=LEFT,fill=X)
        row.pack(side=TOP,fill=X)
        controls3.pack(side=TOP,fill=X)
        controls.pack(side=LEFT,fill=Y)

        panels=PanedWindow(self.page2,orient=HORIZONTAL)
        panels.pack(fill=BOTH,expand=1)
        frameconfs = LabelFrame(panels, text="Configuration File",relief=SUNKEN,padx=5,pady=5)
        self.spinconfigs = ScrolledText(frameconfs,width=100)
        self.spinconfigs.bind("<FocusOut>", self.reload_configs)
        self.spinconfigs.pack(side=LEFT,fill=Y)
        panels.add(frameconfs)
        self.spinconfigs.insert(END, 
                                "# Spin configurations definition file\n" + \
                                "# Energy\t [config]\t\t # label / comment\n")
        
        

        results = LabelFrame(panels,text="Results",padx=5,pady=5)
        eqpanel = LabelFrame(results, text="Equations",relief=SUNKEN,padx=5,pady=5)
        self.equationpanel = ScrolledText(eqpanel,width=20)    #Label(eqpanel,text=50*(80*" "+ "\n "))
        self.equationpanel.config(state=DISABLED)
        self.equationpanel.pack(side=TOP,fill=BOTH)
        eqpanel.pack(side=TOP,fill=BOTH)
        
        #results.pack(side=TOP,fill=BOTH)
        panels.add(results)
        
        self.nb.add(self.page2,text="2. Spin Configurations and Couplings ")

    def build_page3(self):
        self.parameters["page3"] = {}
        self.page3 = Frame(self.nb)
#        ents = {}
        # ents = makeform(self.page1, fields)
        ##########################
        controls = Frame(self.page3)
        # Controls for loading configurations
        controls1 = LabelFrame(controls,text="Settings", width=10,padx=5, pady=5)
        fields = ["Energy tolerance"]
        defaultfields = ("0.1",)
        for i,field in enumerate(fields):
            row = Frame(controls1)
            lab = Label(row, width=14, text=field + ": ", anchor='w')
            ent = Entry(row,width=5,validate='key', validatecommand=self.vcmdf)
            lab.pack(side=LEFT)
            ent.pack(side=LEFT)
            row.pack(side=TOP, padx=5, pady=5)
            ent.insert(0,defaultfields[i])
#            ents[field] = ent
            self.parameters["page3"][field] = ent
        row = Frame(controls1)
        lab = Label(row,width=14,text="Format", anchor="w")
        lab.pack(side=LEFT)
        optformat = OptionMenu(row, self.outputformat, "plain", "latex", "wolfram", command=self.print_full_equations)
        optformat.pack(side=LEFT,fill=X)
        row.pack(side=TOP,fill=X)
        btn = Button(controls1,text="Estimate Parameters",
                     command=self.evaluate_couplings)
        btn.pack()
        controls1.pack(side=TOP,fill=X)
        controls.pack(side=LEFT,fill=Y)

        panels=PanedWindow(self.page3,orient=HORIZONTAL)
        panels.pack(fill=BOTH,expand=1)
        #######  ScrolledText  #############
        
        self.spinconfigsenerg = ScrolledText(panels)
        self.spinconfigsenerg.bind("<FocusOut>", self.reload_configs)
        #self.spinconfigsenerg.pack(side=LEFT,fill=Y)
        panels.add(self.spinconfigsenerg)
        self.spinconfigsenerg.insert(END, 
                                "# Spin configurations definition file\n" + \
                                "# Energy\t [config]\t\t # label / comment\n")
        ######  Results
        results = LabelFrame(panels,text="Results",padx=5,pady=5)
        panelsr = PanedWindow(results,orient=VERTICAL)
        panelsr.pack(side=LEFT,fill=BOTH)

        eqpanel = LabelFrame(panelsr, text="Equations",relief=SUNKEN)
        self.equationpanel2 = ScrolledText(eqpanel,state=DISABLED,height=10,width=200)
        self.equationpanel2.pack(side=LEFT,fill=BOTH,expand=1)
        eqpanel.pack(fill=X,expand=1)
        panelsr.add(eqpanel)


        respanel = LabelFrame(panelsr, text="Determined Parameters",relief=SUNKEN,padx=5,pady=5,width=80)
        self.resparam = ScrolledText(respanel,state=DISABLED,height=10,width=80)
        self.resparam.pack(fill=BOTH,expand=1)
        panelsr.add(respanel)
        
        
        chipanel = LabelFrame(panelsr, text="Energy Errors",relief=SUNKEN)
        chibuttons=Frame(chipanel)
        chibuttons.pack(side=RIGHT)
        Button(chibuttons,text="plot",state=DISABLED).pack(side=TOP)
        self.chis = ScrolledText(chipanel,state=DISABLED,height=10)
        self.chis.pack(side=LEFT,fill=BOTH)
        panelsr.add(chipanel)
        panels.add(results)

        
        self.nb.add(self.page3,text="3. Set energies and evaluate.")
        

        

    def build_page4(self):
        self.page4 = Frame(self.nb)
        self.nb.add(self.page4,text="4. Evaluate parameters")
        self.parameters["page4"]={}        
        self.outputformat =  StringVar()
        self.outputformat.set("Plain")
        ctrl = LabelFrame(self.page4,text="Evaluate and Show parameters",padx=5,pady=5)        
        OptionMenu(ctrl, self.outputformat, "Plain","Latex","Wolfram").pack(side=TOP,fill=X)
        btn = Button(ctrl,text="Evaluate couplings")
        btn.pack(side=TOP)
        ctrl.pack(side=RIGHT,fill=X)
        leftpanel = LabelFrame(self.page4,relief=SUNKEN,text="Configurations",padx=5,pady=5)
        ScrolledText(leftpanel).pack(side=LEFT,fill=Y)
        leftpanel.pack(side=LEFT,fill=Y)
        rightpanel = LabelFrame(self.page4,relief=SUNKEN,text="Results",padx=5,pady=5)
        respanel = LabelFrame(rightpanel, text="Determined Parameters",relief=SUNKEN,padx=5,pady=5)
        Label(respanel,text=20*(80*" "+ "\n ")).pack(fill=BOTH)
        respanel.pack(side=BOTTOM,fill=X)
        eqpanel = LabelFrame(rightpanel, text="Equations",relief=SUNKEN)
        Label(eqpanel,text=20*(80*" "+ "\n ")).pack(fill=BOTH)
        eqpanel.pack(side=BOTTOM,fill=X)
        rightpanel.pack(side=RIGHT,fill=BOTH)

    def about(self):
        messagebox.showinfo("About", '''Version 0.0 - 2017''')

        
    def print_status(self,msg):
        print(msg)
        #self.status.delete(1.0,END)
        #self.status.insert(END, msg+"\n")
        
    def import_model(self):
        filename = filedialog.askopenfilename(initialdir = self.datafolder+"/",
                                          title = "Select file",
                                          filetypes = (("cif files","*.cif"),("Wien2k struct files","*.struct"),("all files","*.*"))
        )        
        self.model=magnetic_model_from_file(filename=filename)
        self.model.save_cif(self.tmpmodel.name)
        self.statusbar.config(text="model loaded")
        with open(self.tmpmodel.name,"r") as tmpf:
            modeltxt=tmpf.read()
            self.modelcif.delete("1.0",END)
            self.modelcif.insert(INSERT,modeltxt)
        self.nb.select(0)
        
    def import_configs(self,clean=True):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        filename = filedialog.askopenfilename(initialdir = self.datafolder+"/",
                                          title = "Select file",
                                          filetypes = (("spin list","*.spin"),("all files","*.*"))
        )        
        self.configurations = read_spin_configurations_file(filename=filename, model=self.model)
        confs = self.configurations[1]
        energies = self.configurations[0]
        labels = self.configurations[2]
        if clean:
            self.spinconfigs.delete(1.0,END)
        with open(self.tmpconfig.name,"w") as of:
            for idx, nc in enumerate(confs):
                row = str(energies[idx]) +"\t"+ str(nc) + "\t\t #" + labels[idx] + "\n"
                of.write(row)
                self.spinconfigs.insert(INSERT,row)
                
        self.statusbar.config(text="config loaded")
        self.nb.select(1) 


        
        
    def save_model(self):
        datafolder = os.getcwd()
        filename = filedialog.asksaveasfilename(initialdir = datafolder+"/",title = "Select file",filetypes = (("cif files","*.cif"),("all files","*.*")))
        print(filename.__repr__())
        if filename == "":
            return
        with open(filename,"w") as tmpf:
            tmpf.write(self.modelcif.get(1.0,END))
        self.print_status(filename)
        self.statusbar.config(text="model saved.")


    def save_configs(self):
        datafolder=os.getcwd()
        filename=filedialog.asksaveasfilename(initialdir = datafolder+"/",title = "Select file",filetypes = (("spin files","*.spin"),("all files","*.*")))
        if filename == "":
            return
        if self.nb.selecter() == self.page3:
            self.reload_configs(src_widget=self.spinconfigsenerg)
        else:
            self.reload_configs()
        with open(filename,"w") as tmpf:
            tmpf.write(self.spinconfigsenerg.get(1.0,END))
        self.print_status(filename)
        self.statusbar.config(text="configurations saved.")


    def print_full_equations(self, ev=None):
        eqformat = self.outputformat.get()
        confs = self.configurations[1]
        labels = self.configurations[2]
        if len(confs) == 0 or len(self.model.bond_lists) == 0:
            return
        cm = self.model.coefficient_matrix(confs,False)
        equations = self.model.formatted_equations(cm,
                                                   ensname=None,
                                                   comments=labels,
                                                   format=eqformat)
        equations = equations + "\n\n |A^+|_{\infty}^{-1} = " +\
                    str(self.model.inv_min_sv_from_config(confs))
        self.equationpanel.config(state=NORMAL)
        self.equationpanel.delete(1.0,END)
        self.equationpanel.insert(END,equations)
        self.equationpanel.config(state=DISABLED)
        # In the tab3, just those configurations with known energies.
        fullconfs = confs
        fulllabels = labels
        confs = []
        labels = []
        energs = self.configurations[0]
        for i,energ in enumerate(energs):
            if energ == energ:
                confs.append(fullconfs[i])
                labels.append(fulllabels[i])
        if len(confs) == 0:
            return        
        cm = self.model.coefficient_matrix(confs,False)
        equations = self.model.formatted_equations(cm,
                                                   ensname=None,
                                                   comments=labels,
                                                   format=eqformat)
        equations = equations + "\n\n |A^+|_{\infty}^{-1} = " +\
                    str(self.model.inv_min_sv_from_config(confs))
        self.equationpanel2.config(state=NORMAL)
        self.equationpanel2.delete(1.0,END)
        self.equationpanel2.insert(END,equations)
        self.equationpanel2.config(state=DISABLED)

        
    def reload_configs(self,ev=None,src_widget=None):        
        self.print_status("updating configs")
        if ev is not None:
            spinconfigs = ev.widget
        else:
            if src_widget is not None:
                spinconfigs = src_widget
            else:
                spinconfigs = self.spinconfigs
        confs = []
        labels = []
        energies = []

        conftxt = spinconfigs.get(1.0,END)
        for linnum,l in enumerate(conftxt.split(sep="\n")):
            ls = l.strip()
            if ls == "" or ls[0]=="#":
                continue
            fields = ls.split(maxsplit=1)
            try:
                energy = float(fields[0])
            except ValueError:
                self.print_status("Error at line " + str(linnum+1))
                return
            ls = fields[1]
            newconf = []
            comment = ""
            for pos, c in enumerate(ls):
                if c == "#":
                    comment = ls[(pos+1):]
                    break
                elif c == "0":
                    newconf.append(0)
                elif c == "1":
                    newconf.append(1)
            while len(newconf)<self.model.cell_size:
                newconf.append(0)
            if comment == "":
                comment = str(confindex(newconf))
            labels.append(comment)
            confs.append(newconf)
            energies.append(energy)
        self.configurations = (energies,confs,labels)
        with open(self.tmpconfig.name,"w") as of:
            for idx, nc in enumerate(confs):
                row = str(energies[idx]) +"\t"+ str(nc) + "\t\t #" + labels[idx] + "\n"
        self.print_full_equations()
        if spinconfigs == self.spinconfigs:
            self.spinconfigsenerg.delete(1.0,END)
            self.spinconfigsenerg.insert(END,conftxt)
        else:
            self.spinconfigs.delete(1.0,END)
            self.spinconfigs.insert(END,conftxt)



        
    def reload_model(self,ev):
        self.print_status("reload model")
        if self.model is None:
            return
        current_model = self.modelcif.get(1.0,END)
        newtmpfile = tempfile.NamedTemporaryFile(mode="w",suffix=".cif")
        newtmpfile.close()
        with open(newtmpfile.name,"w") as ff:
            ff.write(current_model)        
        try: 
            model=magnetic_model_from_file(filename=newtmpfile.name)
        except:            
            self.print_status("the model can not be loaded. Check the syntax.")
            self.statusbar.config(text="the model can not be loaded. Check the syntax.")
            os.remove(newtmpfile.name)
            return            
        # if everything works, 
        self.model=model
        os.remove(self.tmpmodel.name)
        self.tmpmodel = newtmpfile
        self.statusbar.config(text="model updated")
            
#            modeltxt=tmpf.read()
        
    def grow_unit_cell(self):
        if self.model is None:
            self.print_status("Model is not defined. Please load a model")
            return
        self.print_status("growing cell...")
        messagebox.showinfo("Not implemented...","Grow unit cell is not implemented.")
        parms = self.parameters["page1"]
        lx = int(parms["Lx"].get())
        ly = int(parms["Ly"].get())
        lz = int(parms["Lz"].get())
#        self.model.generate_bonds(ranges=[[rmin,rmax]],discretization=discr)
#        self.model.save_cif(self.tmpmodel.name)
#        with open(self.tmpmodel.name,"r") as tmpf:
#            modeltxt=tmpf.read()
#            self.modelcif.delete("1.0",END)
#            self.modelcif.insert(INSERT,modeltxt)

    def optimize_configs(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        if len(self.model.bond_lists) == 0:
            self.print_status("Bonds must be defined before run optimization.")
            return
        parms = self.parameters["page2"]
        n = int(parms['Number of configurations'].get())
        its = int(parms["Iterations"].get())
        us = max(int(parms["Bunch size"].get()),n)
        known = [] #self.configs[1]
        cn,newconfs=self.model.find_optimal_configurations(
                num_new_confs=n,
                start=[],
                known=known,
                its=its, update_size=us
        )
        labels = [str(confindex(c))  for c in newconfs]
        #self.configs=([float("nan") for i in newconfs],newconfs,labels)

        self.spinconfigs.insert(END,"\n# New configurations. ")
        self.spinconfigs.insert(END," |A^{+}|^{-1}:" +  str(cn)  +" :\n")
        for idx, nc in enumerate(newconfs):
            row ="nan \t"+ str(nc) + "\t\t #" + labels[idx] + "\n"
            self.spinconfigs.insert(END,row)
        self.reload_configs(src_widget=self.spinconfigs)
        
        
    def add_bonds(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded. Please load a model first.\n")
            return
        self.print_status("adding bonds...")
        parms = self.parameters["page1"]
        rmin = float(parms["rmin"].get())
        rmax =  float(parms["rmax"].get())
        discr =  float(parms["Discretization"].get())
        self.model.generate_bonds(ranges=[[rmin,rmax]],discretization=discr)
        self.model.save_cif(self.tmpmodel.name)
        with open(self.tmpmodel.name,"r") as tmpf:
            modeltxt=tmpf.read()
            self.modelcif.delete("1.0",END)
            self.modelcif.insert(INSERT,modeltxt)


    def configs_from_other_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded. Please load a model first.")
            return

        #filename = filedialog.askopenfilename(initialdir = self.datafolder+"/",
        #                                  title = "Select model file",
        #                                  filetypes = (("cif files","*.cif"),("Wien2k struct files","*.struct"),("all files","*.*"))
        #)        
        #external_model = magnetic_model_from_file(filename=filename)
        self.print_status("importing configurations...")
        self.reload_configs(src_widget=self.spinconfigs)
        icw = ImportConfigWindow(self)
        self.print_status("donde importing configurations")
        
        # cfilename = filedialog.askopenfilename(initialdir = self.datafolder+"/",
        #                                   title = "Select configuration file",
        #                                   filetypes = (("cif files","*.cif"),("Wien2k struct files","*.struct"),("all files","*.*"))
        # )        
        # exens, ext_configs , exlabels = read_spin_configurations_file(filename=cfilename, model=external_model)
        # map_confs = self.map_config_model1_model2(external_model,ext_configs,self.model)
        
        # self.spinconfigs.insert(END,"\n# Imported configurations from " + cfilename +" according to the model in "+ filename + ".")
        # for idx, nc in enumerate(ext_configs):
        #     row = str(exens[idx]) + "\t"+ str(nc) + "\t\t #" + exlabels[idx] + "\n"
        #     self.spinconfigs.insert(END,row)

        # self.reload_configs(src_widget=self.spinconfigs)



    def evaluate_couplings(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return

        self.reload_configs(src_widget=self.spinconfigsenerg)
        
        tolerance = float(self.parameters["page3"]["Energy tolerance"].get())
        confs = []
        energs = []
        fmt = self.outputformat.get() 
        print("\n**Evaluating couplings")
        for it, c in enumerate(self.configurations[1]):
            en = self.configurations[0][it]
            if en == en:
                energs.append(en)
                confs.append(c)
        if len(confs) < len(self.model.bond_lists)+1:
            self.print_status("Number of known energies is not enough to determine all the couplings\n")
            messagebox.showerror("Error","Number of known energies is not enough to determine all the couplings.")
            return
        js,jerr,chis = self.model.compute_couplings(confs,
                                                    energs,
                                                    err_energs=tolerance)

        offset_energy = js[-1]
        js.resize(js.size-1)
        jmax = max(abs(js))
            
        resparmtxt = "E" + textmarkers["sub_symbol"][fmt] + "0"  +\
                     textmarkers["equal_symbol"][fmt] + str(offset_energy) + "\n\n"
        if min(jerr)<0:
            self.print_status("Warning: error bounds suggest that  the model is not compatible with the data. Try increasing the tolerance by means of the parameter --tolerance [tol].")
            incopatibletxt = textmarkers["open_comment"][fmt] +\
                             " incompatible " +\
                             textmarkers["close_comment"][fmt] +\
                             textmarkers["separator_symbol"][fmt]  + "\n"
            for i,val in enumerate(js):
                if jerr[i]<0:
                    resparmtxt = resparmtxt + self.model.bond_names[i] + " " + \
                                 textmarkers["equal_symbol"][fmt] +  "(" + \
                                 str(val/jmax) +  ") " + \
                                 textmarkers["times_symbol"][fmt] + " " +\
                                 str(jmax) + incopatibletxt
                else:
                    resparmtxt = resparmtxt +  self.model.bond_names[i] + " " +\
                                 textmarkers["equal_symbol"][fmt] +  "(" +\
                                 str(val/jmax) + textmarkers["plusminus_symbol"][fmt] + str(jerr[i]/jmax) +\
                                 ") " + textmarkers["times_symbol"][fmt] +  " " + str(jmax) +\
                                 textmarkers["separator_symbol"][fmt] + "\n" 
        else:
            for i,val in enumerate(js):
                resparmtxt = resparmtxt +  self.model.bond_names[i] + " " + \
                             textmarkers["equal_symbol"][fmt] +  \
                             "(" + str(val/jmax) + " " + \
                             textmarkers["plusminus_symbol"][fmt] +  " " + \
                             str( jerr[i]/jmax) + ") " + \
                             textmarkers["times_symbol"][fmt] + \
                             str(jmax) +"\n"
        self.resparam.config(state=NORMAL)
        self.resparam.delete(1.0,END)
        self.resparam.insert(END,resparmtxt)
        self.resparam.config(state=DISABLED)

        # Update chi panel
        chitext = ""
        labels = self.configurations[2]
        for j,chi in enumerate(chis):
            chitext = chitext  + textmarkers["Delta_symbol"][fmt] + \
                      "E" +  textmarkers["sub_symbol"][fmt]  + str(j + 1) + \
                       textmarkers["equal_symbol"][fmt] + \
                      str(chi) + textmarkers["open_comment"][fmt] + labels[j] + \
                      textmarkers["separator_symbol"][fmt] + \
                      textmarkers["close_comment"][fmt] + "\n"
        self.chis.config(state=NORMAL)
        self.chis.delete(1.0,END)
        self.chis.insert(END,chitext)
        self.chis.config(state=DISABLED)
        
        
    def write(self, txt):
        self.status.insert(INSERT, txt)
                        
            
 
ApplicationGUI()














