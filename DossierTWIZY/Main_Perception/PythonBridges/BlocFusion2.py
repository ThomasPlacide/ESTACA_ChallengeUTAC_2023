import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent

class ObjectsTreatment(): 
    def __init__(self) -> None:

        self.SpaceROI = np.array[0, 50, -5, 5]
        pass

    def FindNearest(self, array, value): 
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()

        return idx

    def CompareCoord(self, camera_object: rtmaps.types.REAL_OBJECT, radar_object: rtmaps.types.REAL_OBJECT):

        
        for cOBJ in camera_object:
                xID_nearest = [self.FindNearest(radar_array.x, cOBJ.y) for radar_array in radar_object]
                yID_nearest = [self.FindNearest(radar_array.y, cOBJ.y) for radar_array in radar_object]


        return xID_nearest, yID_nearest

     
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

        Clustered_Objects.data.x, Clustered_Objects.data.y = ObjectsTreatment().CompareCoord(C_obj, R_obj)

        self.outputs["Clustered_Objects"].write(Clustered_Objects)