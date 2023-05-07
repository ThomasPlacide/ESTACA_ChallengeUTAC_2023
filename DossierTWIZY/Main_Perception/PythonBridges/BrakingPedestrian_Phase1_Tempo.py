import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
from typing import Union

class SensorCalculation(): 
    def __init__(self): 
        self.CriticalROI = {    0: [-2, 2],
                                1: [-10, 10], 
                                2: [-2, 2], 
                                3: [-10, 10]}
        
        self.PreviousTime = 0
        

    def CalculateSamplingRate(self, Current_time):
        """
        Calculate sampling rate based on rt.Current_Time()
        """

        if self.PreviousTime != 0: 
            time_difference = Current_time - self.PreviousTime
            sampling_rate = 1/time_difference
        self.PreviousTime = Current_time

        return sampling_rate
    
    def CalculateDistance(self, VehicleSpeed: float, Current_time: float) -> float: 
        """
        Odometer function.
        """
        frequenceEch = self.CalculateSamplingRate(self.PreviousTime, Current_time)

        Distance = (VehicleSpeed*frequenceEch)/3.6

        return Distance
    
    def CheckPieton(self,  objects: rtmaps.types.REAL_OBJECT, ROIind=1) -> Union[bool,rtmaps.types.CUSTOM_STRUCT]:
        """
        Check if pedestrian is in the critical zone, base on its y location.
        """ 
        coordPieton = {"x": None, 
                       "y": None, 
                       "z": None}
        
        if objects!=[]:
            for obj in objects:
                if  (obj.y >= self.CriticalROI[ROIind][0]) &\
                    (obj.y <= self.CriticalROI[ROIind][1]):
                    coordPieton["x"] = obj.x
                    coordPieton["y"] = obj.y
                    coordPieton["z"] = obj.z

                    return 1, coordPieton
                else: 
                    return 0, coordPieton
        else :
            return 0, coordPieton
        
    def SaturateAngle(self, coeff: float, seuil) -> float:
        """
        Saturate angle given by LKA bloc. 
        """

        if coeff > seuil: 
            coeff = seuil
        
        if coeff < -seuil: 
            coeff = -seuil
        
        return coeff




class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) 
		
		
    def Dynamic(self):
        self.add_input("Objects", rtmaps.types.REAL_OBJECT)
        self.add_input("VehicleSpeed", rtmaps.types.FLOAT64)
        self.add_input("LKAcorrection", rtmaps.types.FLOAT64)

        self.add_output("PietonInROI", rtmaps.types.AUTO)
        self.add_output("target_speed_km_h", rtmaps.types.FLOAT64,0)
        self.add_output("target_gear", rtmaps.types.INTEGER32)
        self.add_output("alert", rtmaps.types.INTEGER32)
        self.add_output("CoordoneesPieton", rtmaps.types.CUSTOM_STRUCT, PEDESTRIAN) # Custom_Struct

        self.add_output("ControleDistance", rtmaps.types.FLOAT64)
        self.add_output("ControleTemps", rtmaps.types.FLOAT64)
        
        self.add_property("Démarrage de la demo [s]",   5, rtmaps.types.INTEGER32)
        self.add_property("Longueur de la piste [m]",   35, rtmaps.types.INTEGER32)
        self.add_property("Distance de sécurité [m]",   5, rtmaps.types.INTEGER32)
        self.add_property("Vitesse de consigne [km/h]", 3, rtmaps.typers.INTEGER32)
        self.add_property("Angle saturant [rad]",       0.005, rtmaps.types.FLOAT64)
        self.add_property("Angle offset [rad]",         -0.032, rtmaps.types.FLOAT64)
        
        
    def Birth(self):
        self.DistanceCovered = 0.0
        self.Starter = self.properties["Démarrage de la demo [s]"].data
        self.DistanceAParcourir = self.properties["Longueur de la piste"].data
        self.DistanceDeSecurite = self.properties["Distance de sécurité"].data
        self.Saturation = self.properties["Angle saturant"].data
        self.AngleOffset = self.properties["Angle offset"].data
        self.SpeedCommand = self.properties["Vitesse de consigne"].data
        
    def Core(self):

        ROIChoice = rt.get_property("BlocFiltreRadar.RadarROI") # Que renvoie get_property
        Current_time = rt.current_time()
        Current_time = Current_time/1e6
        Current_Speed = self.inputs["VehicleSpeed"].ioelt.data
        LKAcoeff = self.inputs["LKAcorrection"].ioelt.data
        ClusteredObjects = self.inputs["Objects"].ioelt.data
        
        self.DistanceCovered += SensorCalculation().CalculateDistance(Current_Speed, Current_time)
        self.outputs["ControleTemps"].write(Current_time)

        Speed = 0
        Alert = 0
        
        if Current_time > self.Starter: 
            Speed = self.SpeedCommand
            Angle = self.AngleOffset + LKAcoeff
            PedestrianPresence, PedestrianCoordinates = SensorCalculation().CheckPieton(ROIind=ROIChoice, \
                                                                                        objects=ClusteredObjects)
            if PedestrianPresence: 
                self.outputs["PietonInROI"].write(True)
                self.outputs["CoordoneesPieton"].write(PedestrianCoordinates)

                if ClusteredObjects.x < self.DistanceDeSecurite:
                    Speed = 0
                    Alert = 1
            else: 
                self.outputs["PietonInROI"].write(False)
                self.outputs["CoordonneesPieton"].write(PedestrianCoordinates)

        if self.DistanceCovered > self.DistanceAParcourir:
            Speed = 0
          
        self.outputs["target_speed_km_h"].write(np.float64(Speed))
        self.outputs["target_gear"].write(np.int32(1))
        self.outputs["target_angle"].write(Angle)

        self.outputs["alert"].write(np.int32(Alert))
        self.outputs["ControleDistance"].write(np.float64(self.DistanceCovered))
        
    def Death(self):
        self.previous_time = 0
        
