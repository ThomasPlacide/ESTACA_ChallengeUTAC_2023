import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent  # base class
#from PIL import Image as im

import copy as cp

import cv2
import numpy as np


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    umat_image = cv2.UMat(image).get().astype('uint8')
     # Convert input image to grayscale
    umat_image = cv2.UMat(umat_image)
    gray = cv2.cvtColor(umat_image, cv2.COLOR_BGR2GRAY)
    return gray



class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)

    def Dynamic(self):
        pass
        # Entree
        self.add_input("image_in", rtmaps.types.MATRIX)

        # Sortie
        self.add_output("image_out", rtmaps.types.IPL_IMAGE)
        #self.add_output("angle_car", rtmaps.types.FLOAT64, 100)

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

    # Core() is called every time you have a new input
    def Core(self):
        
        print("ok")
        frame = self.inputs["image_in"].ioelt.data
        timestamp = self.inputs["image_in"].ioelt.ts
        combined_img=cp.copy(frame)
        combined_img.ts = timestamp
        #numpy_matrix = rtmaps_converter.to_numpy(frame)
        print(type(combined_img))
        arr = np.array(combined_img)
        print(type(arr))
        print(arr.shape)
        img = process_image(arr)
        self.outputs["image_out"].write( img)
        #except:
            #print("here")
            #img = combined_img
            #self.outputs["image_out"].write(img) 

        # Ecriture sur la sortie

    # Death() will be called once at diagram execution shutdown
    def Death(self):
        pass

