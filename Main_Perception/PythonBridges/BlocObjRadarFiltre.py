import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent 

#################
#FONCTIONS PERSO#
#################

class Filtrage():
    def __init__(self, ZoneDetection) -> None:

        self.mask = np.array([ZoneDetection.x1, ZoneDetection.y1,\
                              ZoneDetection.x2, ZoneDetection.y2]) # Coordonnées Radar à déterminer
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


    def WipeObjWOSpeed(self, list_obj): 
        """
        Return an object list with objects that has a speed. 
        """
        NonNulObj = []       
        for obj in list_obj.data: 
            if obj.speed != 0:
                NonNulObj.append(obj)
    

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
        self.add_input("ROI", rtmaps.types.MATRIX)

        # Ouputs
        self.add_output("ObjFiltered", rtmaps.types.REAL_OBJECT)
         
        pass
        
    def Birth(self): 
        pass
        
    def Core(self): 
        ObjRadar = self.inputs["RadarObj"].ioelt
        ZoneDetection = self.inputs["ROI"].ioelt

        NonNulObj = Filtrage(ZoneDetection).WipeObjWOSpeed(ObjRadar)

        self.outputs["ObjFiltered"].write(NonNulObj)

    def Death(self): 
        pass

      

