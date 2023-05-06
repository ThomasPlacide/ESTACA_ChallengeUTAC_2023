import rtmaps.types
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
import numpy as np
import pandas as pd

class ObjectsFilterer(): 
    def __init__(self):        
        self.SpaceROI = [0, 50, -10, 10]
        

    def CompareCoordWithROI(self, Objects): 
        dCoords=[]
        for i,Obj in enumerate(Objects):
            coord = [Obj.x,Obj.y,Obj.z]
            dCoords.append(coord)

        R_DF = pd.DataFrame(data = dCoords, columns=["x", "y", "z"])
        
        X_filtre = R_DF[    (R_DF["x"] < self.SpaceROI[1]) &\
                            (R_DF["x"] > self.SpaceROI[0]) &\
                            (R_DF["y"] > self.SpaceROI[2]) &\
                            (R_DF["y"] < self.SpaceROI[3])]["x"]

        Y_filtre = R_DF[    (R_DF["x"] < self.SpaceROI[1]) &\
                            (R_DF["x"] > self.SpaceROI[0]) &\
                            (R_DF["y"] > self.SpaceROI[2]) &\
                            (R_DF["y"] < self.SpaceROI[3])]["y"]

        return X_filtre, Y_filtre
        

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

        FilteredObj = rtmaps.types.Ioelt()

        Filtered_x, Filtered_y = ObjectsFilterer().CompareCoordWithROI(RawObj.data)
        
        try:
            FilteredObj.x = Filtered_x
            FilteredObj.y = Filtered_y
            self.outputs["FilteredRadarObjects"].write(FilteredObj)

        except: 
            FilteredObj.x = [0]
            FilteredObj.y = [0]
            self.outputs["FilteredRadarObjects"].write(FilteredObj)

    def Death(self): 
        pass
