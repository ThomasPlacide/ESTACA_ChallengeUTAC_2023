import rtmaps.types
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
import numpy as np

class ObjectsFilterer(): 
    def __init__(self):        
        self.SpaceROI = [0, 50, -5, 5]
        

    def CompareCoordWithROI(self, Objects): 
        ind_x = []
        ind_y = []

        ind_x = np.where(Objects.x > self.SpaceROI[1])   
        ind_y = np.where(Objects.y > self.SpaceROI[3]    \
                                or Objects.y < self.SpaceROI[2])

        if ind_x and ind_y: 
            Objects.x = np.delete(Objects.x, ind_x)
            Objects.y = np.delete(Objects.y, ind_x)
            
            Objects.x = np.delete(Objects.x, ind_y)
            Objects.y = np.delete(Objects.y, ind_y)

        return Objects

        


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
        print(RawObj.data)
        FilteredObj = rtmaps.types.Ioelt()

        # FilteredObj = ObjectsFilterer().CompareCoordWithROI(RawObj.data)

        self.outputs["FilteredRadarObjects"].write(FilteredObj)

    def Death(self): 
        pass
