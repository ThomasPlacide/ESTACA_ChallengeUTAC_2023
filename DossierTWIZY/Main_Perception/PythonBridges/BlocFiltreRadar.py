import rtmaps.types
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
import numpy as np

class ObjectsFilterer(): 
    def __init__(self) -> None:        
        self.SpaceROI = np.array[0, 50, -5, 5]

