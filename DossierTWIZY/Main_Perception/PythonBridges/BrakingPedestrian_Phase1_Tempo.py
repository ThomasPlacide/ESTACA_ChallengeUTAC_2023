#This is a template code. Please save it in a proper .py file.
import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent # base class 


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
		
		
    def Dynamic(self):
        
        self.add_input("Objects",rtmaps.types.AUTO)
        self.add_input("LKAcoeff", rtmaps.types.FLOAT64)
        self.add_output("target_steering_angle_rad", rtmaps.types.FLOAT64,0)
        self.add_output("target_speed_km_h", rtmaps.types.FLOAT64,0)
        self.add_output("target_gear", rtmaps.types.INTEGER32)
        self.add_output("alert", rtmaps.types.INTEGER32)
        
        #self.add_property("maxTime",10.0)
        #self.add_property("angle",0.0)
        #self.add_property("speed",0.0)
        #self.add_property("gear",0)


# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")
        

# Core() is called every time you have a new input
    def Core(self):
        ObjDATA = self.inputs["Objects"].ioelt.data 
        Current_time = self.inputs["Objects"].ioelt.ts
        Coeff = self.inputs["LKAcoeff"].ioelt.data
        
        Current_time = self.inputs["LKAcoeff"].ioelt.ts
        
        seuil = 0.0050
        if Coeff > seuil: 
            Coeff = seuil
        if Coeff < -seuil: 
            Coeff = -seuil            
        
        Current_time=Current_time/1e6
        Angle = 0.0
        Speed = 0
        Alert = 0
       


        # Phase de démarrage                
        if (Current_time < 5):
            Speed = 0
            Angle = -0.025
        # Phase LKA
        elif (Current_time < 40):
            Speed = 3
            Angle = -0.032 + Coeff
        # Phase YOLO
        elif (Current_time < 45):
      
            if ( (ObjDATA[0].id == 0) and (ObjDATA[0].x != 0)):
            #if (1):
                Alert = 1
                Speed = 0
                Angle = -0.025
            else:
               Speed = 3
               Angle = -0.026 
        # Fin de scénario
        elif (Current_time < 50):
            Speed = 0
            Angle = -0.028
                
                
        

        
            
#        if (Current_time < 10):
#            Speed = 3
#            
#        elif (Current_time < 15):
#            Speed = 0
#
#        elif (Current_time < 20):
#            Speed = 0
#            Angle = -0.1
#        elif (Current_time < 23):
#            Speed = 3
#            Angle = -0.1
#        else:
#            Speed = 0
#            Angle = 0.01
        # create an ioelt from the input
        #out=self.input["nbOfObjectsInROI"]
        #self.outputs["out"].write(out) # and write it to the output
        #int cpt;
               
        Longi = 0    
        self.outputs["target_steering_angle_rad"].write(np.float64(Angle))         
        self.outputs["target_speed_km_h"].write(np.float64(Speed))
        self.outputs["target_gear"].write(np.int32(1))
        self.outputs["alert"].write(np.int32(Alert))
       
# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
