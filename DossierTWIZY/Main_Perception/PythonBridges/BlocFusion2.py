import rtmaps.types
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

    def CompareCoord(self, ObjectsIn):
        dCoords=[]
        for i,Obj in enumerate(ObjectsIn):
            coord = [Obj.x,Obj.y,Obj.z]
            dCoords.append(coord)

        R_DF = pd.DataFrame(data = dCoords, columns=["x", "y", "z"])
        
        a=R_DF[(R_DF["x"] < 50) & (R_DF["x"] > 0) & (R_DF["y"] > -10) & (R_DF["y"] < 10)]["x"]

        b=R_DF[(R_DF["x"] < 50) & (R_DF["x"] > 0) & (R_DF["y"] > -10) & (R_DF["y"] < 10)]["y"]

        return a, b
        

     
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        # self.add_input("Camera_Objects", rtmaps.types.REAL_OBJECT)
        self.add_input("Radar_Objects", rtmaps.types.REAL_OBJECT)

        self.add_output("Clustered_Objects", rtmaps.types.REAL_OBJECT)

    def Birth(self): 
        print("Birth")
        pass

    def Core(self):
        print('Core')

        C_obj = self.inputs["Camera_Objects"].ioelt
        R_obj = self.inputs["Radar_Objects"].ioelt

        Clustered_Objects = rtmaps.types.Ioelt()
        
        ObjectsTreatment().CompareCoord(R_obj.data)

        # Clustered_Objects.data.x, Clustered_Objects.data.y = ObjectsTreatment().CompareCoord(C_obj, R_obj)

        self.outputs["Clustered_Objects"].write(R_obj)
        R_obj = None

    

    def Death(self): 
        pass