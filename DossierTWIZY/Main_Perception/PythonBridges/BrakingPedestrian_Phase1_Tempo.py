#This is a template code. Please save it in a proper .py file.
import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent # base class 

class SensorCalculation(): 
    def __init__(self): 
        self.CriticalROI = {    0: [-10, 10],
                                1: [-5, 5] }

    def CalculateDistance(self, VehicleSpeed: float, Current_time: float) -> float: 
        """
        Odometer function.
        """
        global tabFreq
        tabFreq.append(Current_time)

        frequenceEch = np.mean(np.diff(tabFreq))
        Distance = (VehicleSpeed*frequenceEch)/3.6

        return Distance
    
    def CheckPieton(self, objects) -> bool:
        """
        Check if pedestrian is in the critical zone, base on its y location.
        """ 
        
        if objects!=[]:
            for obj in objects:
                if obj == 0: 
                    return 1
                else: 
                    return 0
        else :
            return 0
        
    def SaturateAngle(self, coeff: float, seuil = 0.005) -> float:
        """
        Saturate angle given by LKA bloc. 
        """

        if coeff > seuil: 
            coeff = seuil
        
        if coeff < -seuil: 
            coeff = -seuil
        
        return coeff




# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        self.add_input("MemoryDistance", rtmaps.types.FLOAT64)
        self.add_input("Objects", rtmaps.types.REAL_OBJECT)
        self.add_input("VehicleSpeed", rtmaps.types.FLOAT64)

        # self.add_input()

        self.add_output("PietonInROI", rtmaps.types.AUTO)
        self.add_output("target_speed_km_h", rtmaps.types.FLOAT64,0)
        self.add_output("target_gear", rtmaps.types.INTEGER32)
        self.add_output("alert", rtmaps.types.INTEGER32)

        self.add_output("ControleDistance", rtmaps.types.FLOAT64)
        self.add_output("ControleTemps", rtmaps.types.FLOAT64)
        

# Birth() will be called once at diagram execution startup
    def Birth(self):
        # self.DistanceCovered = 0
        # DistanceCovered = 0
        
        global tabFreq
        tabFreq = []
        

# Core() is called every time you have a new input
    def Core(self):
        DistanceAParcourir = 35 # Longueur de la piste
        Current_time = rt.current_time()
        Current_time = Current_time/1e6
        Current_Speed = self.inputs["VehicleSpeed"].ioelt.data
        

        DistanceCovered = rtmaps.types.FLOAT64

        # print(self.input_that_answered)
        if self.input_that_answered != 0:
            DistanceCovered = SensorCalculation().CalculateDistance(Current_Speed, Current_time)
            print(self.input_that_answered)
        else:
            PreviousDistance = self.inputs["MemoryDistance"].ioelt.data 
            print(PreviousDistance)
            DistanceCovered = PreviousDistance + SensorCalculation().CalculateDistance(Current_Speed, Current_time)
        

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
        self.outputs["ControleDistance"].write(np.float64(DistanceCovered))
        self.outputs["ControleTemps"].write(Current_time)
        pass
       
# Death() will be called once at diagram execution shutdown
    def Death(self):
        self.DistanceCovered = 0
        # print("Before death %{0:.2f}".format(DistanceCovered))
        # DistanceCovered = 0
        # print("After death %{0:.2f}".format(DistanceCovered))
        pass
