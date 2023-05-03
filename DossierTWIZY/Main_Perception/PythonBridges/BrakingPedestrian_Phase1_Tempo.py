#This is a template code. Please save it in a proper .py file.
import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent # base class 

class SensorCalculation(): 
    def __init__(self): 
        pass

    def CalculateDistance(self, VehicleSpeed: float, Timestamps: float) -> float: 

        return VehicleSpeed*(Timestamps*60**2)/1e3
    
    def CheckPieton(self, objects) -> bool: 
        
        if objects!=[]:
            for obj in objects:
                if obj == 0: 
                    return 1
                else: 
                    return 0
        else :
            return 0



# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        
        self.add_input("VehicleSpeed", rtmaps.types.FLOAT64)
        self.add_input("Objects", rtmaps.types.REAL_OBJECT)

        self.add_output("PietonInROI", rtmaps.types.AUTO)
        self.add_output("target_speed_km_h", rtmaps.types.FLOAT64,0)
        self.add_output("target_gear", rtmaps.types.INTEGER32)
        self.add_output("alert", rtmaps.types.INTEGER32)
        

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")
        

# Core() is called every time you have a new input
    def Core(self):
        DistanceAParcourir = 35 # Longueur de la piste
        Current_time = self.inputs["Objects"].ioelt.ts
        Current_time = Current_time/1e6
        Current_Speed = self.inputs["VehicleSpeed"].ioelt.data
        DistanceCovered = SensorCalculation().CalculateDistance(Current_Speed, Current_time)

        Speed = 0
        Alert = 0

        if Current_time > 5: 
            Speed = 3 
            if SensorCalculation().CheckPieton(self.inputs["Objects"].ioelt.data): 
                self.outputs["PietonInROI"].write(True)
                Speed = 0
                Alert = 1
            else: 
                self.outputs["PietonInROI"].write(False)

        if DistanceCovered > DistanceAParcourir:
            Speed = 0
          
        self.outputs["target_speed_km_h"].write(np.float64(Speed))
        self.outputs["target_gear"].write(np.int32(1))
        self.outputs["alert"].write(np.int32(Alert))
       
# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
