import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent

class ObjectsTreatment(): 
    def __init__(self) -> None:

        self.SpaceROI = np.array[0, 50, ]
        pass

    def CompareCoord(self, camera_object: rtmaps.types.REAL_OBJECT, radar_object: rtmaps.types.REAL_OBJECT): 
        
        pass

     
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        self.add_input("Camera_Objects", rtmaps.types.REAL_OBJECT)
        self.add_input("Radar_Objects", rtmaps.types.REAL_OBJECT)

        self.add_output("Clustered_Objects", rtmaps.types.REAL_OBJECT)

    def Core(self):

        C_obj = self.inputs["Camera_Objects"].ioelt
        R_obj = self.inputs["Radar_Objects"].ioelt

        Clustered_Objects = rtmaps.types.Ioelt()