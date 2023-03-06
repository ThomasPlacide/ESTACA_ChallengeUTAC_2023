import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent 


#############
#BLOC RTMAPS#
#############
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) 

    def Dynamic(self):
        #Entree
        self.add_input("Objets", rtmaps.types.REAL_OBJECT)

        #Sortie
        self.add_output("PietonInROI", rtmaps.types.AUTO) 

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

# Core() is called every time you have a new input
    def Core(self):

        #Recuperation des infos objets
        objects = self.inputs["Objets"].ioelt.data
        print("objets", objects)
        

        #Recuperation du timestamp
        ts = self.inputs["Objets"].ioelt.ts

        #Creation de l'element de sortie
        Out = rtmaps.types.Ioelt()

        #Ecriture du timestamps
        Out.ts = ts

        #Initialisation a false (pas de pietion)
        Out.data = False

        #On parcour pour chaque objet, si il y a au moins un pieton, la sortie = true
        for obj in objects:
            if obj.misc1 == 0:
                Out.data = True

        #Ecriture sur la sortie
        self.outputs["PietonInROI"].write(Out) 

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
