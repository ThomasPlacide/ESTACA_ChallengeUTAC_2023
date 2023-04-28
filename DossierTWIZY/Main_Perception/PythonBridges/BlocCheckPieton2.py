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
        self.add_input("ClassIDs", rtmaps.types.ANY)

        #Sortie
        self.add_output("PietonInROI", rtmaps.types.AUTO) 

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

# Core() is called every time you have a new input
    def Core(self):

        #Recuperation des infos objets
        objects = np.array(self.inputs["ClassIDs"].ioelt.data)
        #print("objects",objects)
        #Recuperation du timestamp
        ts = self.inputs["ClassIDs"].ioelt.ts
        
        EmptyIoelt1 = rtmaps.types.Ioelt()
        EmptyIoelt1.data = np.array([],dtype=np.uint64)
        EmptyIoelt1.ts = ts

        #Creation de l'element de sortie
        Out = rtmaps.types.Ioelt()

        #Ecriture du timestamps
        Out.ts = ts

        #Initialisation a false (pas de pietion)
        
        if objects!=[]:
        #On parcour pour chaque objet, si il y a au moins un pieton, la sortie = true
            for obj in objects:
                if obj == 0: # L'attribut misc1 est initialisé avec class_ids dans BlocTrackingImage2Objects
                    Out.data = 1
                else: # Si class_ids != 0 OU pas d'objets détectés
                    Out.data = 0
            self.outputs["PietonInROI"].write(Out) 
        else :
            Out.data = 0
            self.outputs["PietonInROI"].write(Out) 

        # print('sortie : ', Out)
        # print('class Id :', objects.misc1)

        #Ecriture sur la sortie
        

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
