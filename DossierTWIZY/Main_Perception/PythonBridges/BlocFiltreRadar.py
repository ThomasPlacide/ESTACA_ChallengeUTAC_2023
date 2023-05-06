import rtmaps.types, rtmaps.real_objects
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
import numpy as np
import pandas as pd

class ObjectsFilterer(): 
    def __init__(self):        
        self.ROIs = {   0: [0, 100, -10, 10], 
                            1: [0, 100, -20, 20], 
                            2: [0, 40, -10, 10], 
                            3: [0, 40, -20, 20] }
        

    def CompareCoordWithROI(self, indROI, Objects): 
        """
        First cleaning on radar objects. 
        """

        SpaceROI = self.ROIs[indROI]
        list_Obj=[]
        for i,Obj in enumerate(Objects):
            coord = [Obj.x,Obj.y,Obj.z]
            list_Obj.append(coord)

        R_DF = pd.DataFrame(data = list_Obj, columns=["x", "y", "z"])
        
        filtre = R_DF[      (R_DF["x"] < SpaceROI[1]) &\
                            (R_DF["x"] > SpaceROI[0]) &\
                            (R_DF["y"] > SpaceROI[2]) &\
                            (R_DF["y"] < SpaceROI[3])]["x"]

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

        self.add_property("Radar ROI", \
                          "|0|Long & tight [0, 100, -5, 5]|Long and wide [0, 100, -20, 20]|Short and tight [0, 40, -5, 5]|Short and wide [0, 40, -20, 20]",\
                            rtmaps.types.ENUM)
    def Birth(self):
        self.ROIChoice = self.properties["Radar ROI"].data
        pass

    def Core(self): 
        
        RawObj = self.inputs["Radar_Objects"].ioelt
        
        if RawObj.data:

            FilteredObj = rtmaps.types.Ioelt()
            FilteredObj.data = rtmaps.real_objects.RealObject()

            NewObj = ObjectsFilterer().CompareCoordWithROI(self.ROIChoice, RawObj.data)
            
            FilteredObj.data = NewObj
           
            self.outputs["FilteredRadarObjects"].write(FilteredObj)

    def Death(self): 
        pass
