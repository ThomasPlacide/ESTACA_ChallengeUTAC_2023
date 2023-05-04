import rtmaps.types
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
import numpy as np

class ObjectsFilterer(): 
    def __init__(self) -> None:        
        self.SpaceROI = np.array[0, 50, -5, 5]

    def CompareCoordWithROI(self, Objects): 

        ind_x = np.where(Objects.data.x > self.SpaceROI[1])

        Objects.data.x = np.delete(Objects.data.x, ind_x)
        Objects.data.y = np.delete(Objects.data.y, ind_x)

        ind_y = np.where(Objects.data.y < self.SpaceROI[2] \
                         or Objects.data.y > self.SpaceROI[3])

        Objects.data.x = np.delete(Objects.data.x, ind_y)
        Objects.data.y = np.delete(Objects.data.y, ind_y)

        


class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        self.add_input("Radar_Objects", rtmaps.types.REAL_OBJECT)

        self.add_output("FilteredRadarObjects", rtmaps.types.REAL_OBJECT)

    def Core(self): 

        RawObj = self.inputs["Radar_Objects"].ioelt

        FilteredObj = rtmaps.types.Ioelt()

        FilteredObj = ObjectsFilterer().CompareCoordWithROI(RawObj)

        self.outputs["FilteredRadarObjects"].write(FilteredObj)
