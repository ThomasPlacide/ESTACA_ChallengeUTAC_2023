
import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent
from LKA_lib import Pipeline 


#############
#BLOC RTMAPS#
#############

class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) 

    def Dynamic(self):
        #Entree
        self.add_input("Image", rtmaps.types.REAL_OBJECT)

        #Sortie
        self.add_output("ImageTraitee", rtmaps.types.AUTO)
        self.add_output("CoordonneesLignes", rtmaps.types.AUTO) 

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        pass

    # Core() is called every time you have a new input
    def Core(self):
        VideoFlux = self.inputs["Image"].ioelt.dat
        TreatedImage, LinesCoor = Pipeline.ApplyPipeline(img=VideoFlux)

        pass

       
    # Death() will be called once at diagram execution shutdown
    def Death(self):
        pass

