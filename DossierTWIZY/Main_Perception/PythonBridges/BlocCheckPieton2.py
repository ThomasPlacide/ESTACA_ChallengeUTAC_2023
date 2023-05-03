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
        self.add_input("Objects", rtmaps.types.REAL_OBJECT, 50)

        #Sortie
        self.add_output("PietonInROI", rtmaps.types.AUTO) 

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

# Core() is called every time you have a new input
    def Core(self):

        #Recuperation des infos objets
        objects = self.inputs["Objects"].ioelt.data

        #Recuperation du timestamp
        ts = self.inputs["ClassIDs"].ioelt.ts

        #Creation de l'element de sortie
        Out = rtmaps.types.Ioelt()

        #Ecriture du timestamps
        Out.ts = ts            

        #Initialisation a false (pas de pietion)
        
        if objects!=[]:
            for obj in objects:
                if obj == 0: 
                    Out.data = 1
                else: 
                    Out.data = 0
        else :
            Out.data = 0
            
        self.outputs["PietonInROI"].write(Out) 

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
