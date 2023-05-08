import rtmaps.types, rtmaps.real_objects
import numpy as np
import pandas as pd
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
from BlocFiltreRadar import ObjectsFilterer as ObjF

class ObjectsTreatment(): 
    def __init__(self) -> None:
        pass
    
    def CompareCoordRWCoordC(self, CameraObject: rtmaps.types.REAL_OBJECT, RadarObject: rtmaps.types.Ioelt(), tolerance=0.1):
        """
        Clustering objects based on their coordinates.
        """ 
        R_DF = ObjF().RealObjects2DF(RadarObject.data)
        C_DF = ObjF().RealObjects2DF(CameraObject)

        Similar = []
        for nb in range(RadarObject.vector_size): 
            sim = C_DF[     (R_DF["x"][nb] >= (1.0-tolerance)*C_DF["x"]) &\
                            (R_DF["x"][nb] <= (1.0+tolerance)*C_DF["x"]) &\
                            (R_DF["y"][nb] >= (1.0-tolerance)*C_DF["y"]) &\
                            (R_DF["y"][nb] <= (1.0+tolerance)*C_DF["y"])]

            Similar= sim.append(sim, ignore_index=True)

        return ObjF().ApplyFilterToRealObjects(CameraObject, Similar)
        
        

    def CompareCoordWSpeed(self, R_Obj, C_Obj):
        """
        Cluster any object as one intersting objects if its seen as a pedestrian by the camera and if its speed is greater than zero.
        """

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
        self.add_output("Clustered_Objects", rtmaps.types.REAL_OBJECT, 100)

    def Birth(self): 
        pass

    def Core(self):

        ObjOutput = rtmaps.types.Ioelt()
        ObjOutput.data = rtmaps.real_objects.RealObject()

        R_Obj = self.inputs["ObjetsRadar"].ioelt
    
        C_Obj = self.inputs["ObjetsCamera"].ioelt.data    

        list_obj = ObjectsTreatment().CompareCoordWSpeed(R_Obj, C_Obj)
                    
        if list_obj: 
            # print(list_obj)
            ObjOutput.data = list_obj
            self.outputs["Clustered_Objects"].write(ObjOutput)


    def Death(self): 
        pass