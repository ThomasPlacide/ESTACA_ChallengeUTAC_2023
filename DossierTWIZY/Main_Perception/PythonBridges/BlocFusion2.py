import rtmaps.types, rtmaps.real_objects
import numpy as np
import pandas as pd
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent

class ObjectsTreatment(): 
    def __init__(self) -> None:

        self.SpaceROI = np.array[0, 50, -10, 10]
        pass

    def FindNearest(self, array, value): 
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()

        return idx
    
    def CompareCoordRWCoordC(self, CameraObject, RadarObject): 

        
        pass

    def CompareCoordWSpeed(self, CameraObject, RadarObject):
        print("PREMIER_BIS")
        for ind in CameraObject:
            print("DEUXIEME") 
            if CameraObject[ind].misc1 == 0:
                print("TROISIEME") 
                for ind_rad in RadarObject:
                    print("QUATRIEME") 
                    if ind_rad.speed != 0:
                        print('CINQUIEME') 

        return True


     
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        self.add_input("ObjetsCamera", rtmaps.types.REAL_OBJECT) #0
        self.add_input("ObjetsRadar", rtmaps.types.REAL_OBJECT) #1
        self.add_output("Clustered_Objects", rtmaps.types.REAL_OBJECT)

    def Birth(self): 
        print("Birth")
        pass

    def Core(self):

        Clustered_Objects = rtmaps.types.Ioelt()
        Clustered_Objects.data = rtmaps.real_objects.RealObject()

        R_Obj = self.inputs["ObjetsRadar"].ioelt.data
        try:
            
            C_Obj = self.inputs["ObjetsCamera"].ioelt.data    
            # ObjectsTreatment().CompareCoord(C_Obj.data, R_Obj.data)
            for ind in C_Obj: 
                if ind.misc1 == 0: 
                    for ind_rad in R_Obj:
                        if ind_rad.data.speed != 0:
                            pass

            self.outputs["Clustered_Objects"].write(Clustered_Objects)
        except: 
            pass

        
    

    def Death(self): 
        pass