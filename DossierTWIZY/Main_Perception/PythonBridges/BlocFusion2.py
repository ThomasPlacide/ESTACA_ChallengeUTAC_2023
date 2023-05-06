import rtmaps.types, rtmaps.real_objects
import numpy as np
import pandas as pd
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent

class ObjectsTreatment(): 
    def __init__(self) -> None:
        pass
    
    def CompareCoordRWCoordC(self, CameraObject, RadarObject): 

        
        pass

    def CompareCoordWSpeed(self, R_Obj, C_Obj):

        list_obj = []
        for ind in C_Obj: 
            if ind.misc1 == 0: 
                for ind, rad in enumerate(R_Obj.data):
                    if rad.data.speed != 0:
                        list_obj.append(rad)
        return list_obj


     
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        self.add_input("ObjetsCamera", rtmaps.types.REAL_OBJECT) #0
        self.add_input("ObjetsRadar", rtmaps.types.REAL_OBJECT) #1
        self.add_output("ObjOutput", rtmaps.types.REAL_OBJECT, 100)

    def Birth(self): 
        pass

    def Core(self):

        Clustered_Objects = rtmaps.types.Ioelt()
        Clustered_Objects.data = rtmaps.real_objects.RealObject()

        R_Obj = self.inputs["ObjetsRadar"].ioelt
    
        C_Obj = self.inputs["ObjetsCamera"].ioelt.data    

        list_obj = ObjectsTreatment().CompareCoordWSpeed(R_Obj, C_Obj)
                    
        if list_obj: 
            # print(list_obj)
            Clustered_Objects.data = list_obj
            self.outputs["ObjOutput"].write(Clustered_Objects)


    def Death(self): 
        pass