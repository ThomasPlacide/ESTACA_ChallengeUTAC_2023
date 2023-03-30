import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent 

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
    
    def EstimationEnM(self, coord_obj: float): 
        """
        Estime la valeur en mètres du float renvoyé par la composante X des coordonées de l'objet détecté par la caméra.
        Essais approximatifs avec une webcam Logitech HD PRo Webcam C920
        """
        coef_conv = 10*1/12

        if coord_obj.x == -100: 
            return 0 
        elif coord_obj.x < 5: 
            return 0 
        else: 
            return (10-coord_obj.x)*coef_conv
            
    
    def ComparaisonDistances(self, coord_obj: rtmaps.types.REAL_OBJECT, speed) -> bool: 
        """
        Compare la distance à parcourir jusqu'à l'arrêt avec les coordonées (X) du piéton détecté.
        Renvoi TRUE si distance arrêt ≤ distance du piéton.
        """

        if self.DistanceAParcourir(speed) <= self.EstimationEnM(coord_obj): 

            return True
        else: 
            
            return False

#############
#BLOC RTMAPS#
#############

class rtmaps_python(BaseComponent): 
    def __init__(self) -> None:
        BaseComponent.__init__(self) # call base class constructor
        self.force_reading_policy(rtmaps.reading_policy.SYNCHRO) #Pour lire les 3 entrees en même temps.
        

    def Dynamic(self): 
        # Inputs
        self.add_input("VehicleSpeed", rtmaps.types.FLOAT64)
        self.add_input("Objects", rtmaps.types.REAL_OBJECT) 

        # Ouputs
        self.add_output("CommandeActionneurFrein", rtmaps.types.ANY)

    def Birth(self): 
        pass
    
    def Core(self): 
        Speed = self.inputs["VehicleSpeed"].ioelt.data
        Obj = self.inputs["Objects"].ioelt.data

        print('BlocDistanceArret', DistanceArret().ComparaisonDistances(Obj, Speed))

        if DistanceArret().ComparaisonDistances(Obj, Speed): 
            self.outputs["CommandeActionneurFrein"].write("1")
        else: 
            self.outputs["CommandeActionneurFrein"].write("0")

    def Death(self): 
        pass

      

