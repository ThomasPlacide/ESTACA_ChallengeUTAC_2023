import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent 
import sys

#################
#FONCTIONS PERSO#
#################

class DistanceArret: 
    def __init__(self) -> None:
        pass

    def DistanceAParcourir(self, speed):
        """
        Formule empirique utilisé lors de l'apprentissage du code de la route.
        Vitesse à renseigner en km/h ! 
        """
        return np.power( self.ConversionMpS2KMpH(speed)/10, 2)
            
    
    def ComparaisonDistances(self, coord_obj: rtmaps.types.REAL_OBJECT, speed) -> bool: 
        """
        Compare la distance à parcourir jusqu'à l'arrêt avec les coordonées (X) du piéton détecté.
        Renvoi TRUE si distance arrêt ≤ distance du piéton.
        """

        threshold = 20

        DistEGO_Obj = self.DistanceAParcourir - coord_obj

        if DistEGO_Obj <= threshold:

            return True
        else: 
            
            return False

#############
#BLOC RTMAPS#
#############

class rtmaps_python(BaseComponent): 
    def __init__(self) -> None:
        BaseComponent.__init__(self) # call base class constructor
        #self.force_reading_policy(rtmaps.reading_policy.SYNCHRO) #Pour lire les 3 entrees en même temps.
        

    def Dynamic(self): 
        # Inputs
        
        #self.add_input("VehicleSpeed", rtmaps.types.FLOAT64)
        self.add_input("Objects", rtmaps.types.REAL_OBJECT) 

        # Ouputs
        self.add_output("CommandeActionneurFrein", rtmaps.types.AUTO)
        

    def Birth(self): 
        pass
        
    def Core(self): 
        Obj = self.inputs["Objects"].ioelt.data
        #Speed = self.inputs["VehicleSpeed"].ioelt.data
        Speed=25

        #print(Obj[0].x)

        if DistanceArret().ComparaisonDistances(Obj, Speed): 
            print('Distance violée')
            self.outputs["CommandeActionneurFrein"].write(True)
        else: 
            self.outputs["CommandeActionneurFrein"].write(False)

    def Death(self): 
        pass

      

