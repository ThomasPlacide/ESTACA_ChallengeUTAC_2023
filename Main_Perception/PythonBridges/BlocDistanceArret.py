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

    def ConversionMpS2KMpH(self, speed): 
        """
        Conversion d'une vitesse m/s en km/h.
        """

        return speed*1e3/3600
    
    def EstimationEnM(self, coord_obj: rtmaps.types.REAL_OBJECT): 
        """
        Estime la valeur en mètres du float renvoyé par la composante X des coordonées de l'objet détecté par la caméra.
        Essais approximatifs avec une webcam Logitech HD PRo Webcam C920
        """

        if coord_obj[0].x < 0: 
            return 0  
        else: 
            return coord_obj[0].x # Distance réelle déjà calculée
            
    
    def ComparaisonDistances(self, coord_obj: rtmaps.types.REAL_OBJECT, speed) -> bool: 
        """
        Compare la distance à parcourir jusqu'à l'arrêt avec les coordonées (X) du piéton détecté.
        Renvoi TRUE si distance arrêt ≤ distance du piéton.
        """

        if self.DistanceAParcourir(speed) <= self.EstimationEnM(coord_obj): 

            return True
        else: 
            
            return False

class CameraCalibration: 
    '''
    Utiliser calibration camera à l'aide de la bibliothèque CV2
    '''
    def __init__(self) -> None:
        pass
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

        #print('BlocDistanceArret', DistanceArret().ComparaisonDistances(Obj, Speed))
        #print(Obj[0].x)

        if DistanceArret().ComparaisonDistances(Obj, Speed): 
            print('Distance violée')
            self.outputs["CommandeActionneurFrein"].write(True)
        else: 
            self.outputs["CommandeActionneurFrein"].write(False)

    def Death(self): 
        pass

      

