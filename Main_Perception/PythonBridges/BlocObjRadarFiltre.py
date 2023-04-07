import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent 

#################
#FONCTIONS PERSO#
#################

class Filtrage():
    def __init__(self) -> None:

        self.mask = np.array([x1, y1, x2, y2]) # Coordonnées Radar à déterminer
        pass

    def WipeObjFarAway(self, list_obj): 
        """
        Remove real_objects too far from radar
        NOTE: "list_obj" must be the complete list returned from CAN decoder
        """

        for each in list_obj: 
            if all([(each.x > self.mask[0]), (each.x < self.mask[2])\
                    (each.y > self.mask[1]), (each.y < self.mask[3])]): 
                list_obj.remove(list_obj == each) 
        


    

#############
#BLOC RTMAPS#
#############

class rtmaps_python(BaseComponent): 
    def __init__(self) -> None:
        BaseComponent.__init__(self) # call base class constructor
        
    def Dynamic(self):
        """
        NOTE: "RadarObj" must come from a CAN decoder -> To_RealObject
        """
        # Inputs
        self.add_input("RadarObj", rtmaps.types.REAL_OBJECT) 

        # Ouputs
        self.add_output("ObjFiltered", rtmaps.types.REAL_OBJECT)
         
        pass
        
    def Birth(self): 
        pass
        
    def Core(self): 
        pass

    def Death(self): 
        pass

      

