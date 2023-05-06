import rtmaps.types, rtmaps.real_objects
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
import numpy as np
import pandas as pd

class ObjectsFilterer(): 
    def __init__(self):        
        self.SpaceROI = [0, 50, -5, 5]
        

    def CompareCoordWithROI(self, Objects): 
        list_Obj=[]
        for i,Obj in enumerate(Objects):
            coord = [Obj.x,Obj.y,Obj.z]
            list_Obj.append(coord)

        R_DF = pd.DataFrame(data = list_Obj, columns=["x", "y", "z"])
        
        filtre = R_DF[    (R_DF["x"] < self.SpaceROI[1]) &\
                            (R_DF["x"] > self.SpaceROI[0]) &\
                            (R_DF["y"] > self.SpaceROI[2]) &\
                            (R_DF["y"] < self.SpaceROI[3])]["x"]

        newObj = []
        for i in filtre.index: 
            newObj.append(Objects[i])

        return newObj
        

class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        self.add_input("Radar_Objects", rtmaps.types.REAL_OBJECT)

        self.add_output("FilteredRadarObjects", rtmaps.types.REAL_OBJECT, 100)

    def Birth(self):
        pass

    def Core(self): 
        
        RawObj = self.inputs["Radar_Objects"].ioelt
        
        if RawObj.data:

            FilteredObj = rtmaps.types.Ioelt()
            FilteredObj.data = rtmaps.real_objects.RealObject()

            NewObj = ObjectsFilterer().CompareCoordWithROI(RawObj.data)
            
            FilteredObj.data = NewObj
           
            self.outputs["FilteredRadarObjects"].write(FilteredObj)

    def Death(self): 
        pass
