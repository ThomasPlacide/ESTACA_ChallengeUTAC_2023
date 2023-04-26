#This is a template code. Please save it in a proper .py file.
import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent # base class
import cv2
from PIL import Image as im
import copy as cp
from yolov7 import YOLOv7

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Filtrer les ids 0:2

class CameraObject:
    def __init__(self):
        
        self.scores = []
        self.boxes = []
        self.class_ids = []
        
    pass
    


def filteredIds(input: CameraObject) -> CameraObject:
    """
    Récupère flux d'information, ne ressort que les IDs intéréssants.
    input = <CameraObject>
    """
    if (input.scores!=[]):
        ind=np.where(input.class_ids > 3)[0]
        print(ind)
        # ind = [indexes > 3 for indexes in input.class_ids ]
        input.scores=np.delete(input.scores,ind)
        input.boxes=np.delete(input.boxes,ind, axis=0)
        input.class_ids=np.delete(input.class_ids,ind)
        
    return input

def prepareBoxes(array,indexes):
    #Fonction permettant de preparer les bonnes boites 
    # dans le bon format pour la sortie sur RTMaps
    array = np.array(array,dtype=np.uint64)
    array = array.flatten()
    
    nArray = np.array([],dtype=np.uint64)
    for i in indexes:
        i = int(i)
        nArray = np.append(nArray,array[i*4:i*4+4])

    return nArray

def prepareIds(array,indexes):
    #Fonction permettant de preparer les bonnes classes 
    # dans le bon format pour la sortie sur RTMaps
    array = np.array(array,dtype=np.uint64)
    array = array.flatten()
    nArray = np.array([],dtype=np.uint64)
   
    for i in indexes:
        i = int(i)
        nArray = np.append(nArray,array[i])

    return nArray

def prepareConfs(array,indexes):
    #Fonction permettant de preparer les bonnes confiances 
    # dans le bon format pour la sortie sur RTMaps
    array = np.array(array)
    array = array.flatten()
    nArray = np.array([],dtype=np.float64)
   
    for i in indexes:
        i = int(i)
        nArray = np.append(nArray,array[i])

    return nArray

# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        

    def Dynamic(self):
        self.add_input("image_in", rtmaps.types.IPL_IMAGE) # define input
        self.add_output("image_out", rtmaps.types.IPL_IMAGE) # define output
        self.add_output("boxes", rtmaps.types.UINTEGER64,100) # define output
        self.add_output("scores", rtmaps.types.FLOAT64,100) # define output
        self.add_output("class_ids", rtmaps.types.UINTEGER64,100) # define output
        self.add_output("label", rtmaps.types.ANY,100) # define output

        self.add_property("Yolo_version", "3|0|yolov7_384x640.onnx|yolov7_480x640.onnx|yolov7-tiny_384x640.onnx", rtmaps.types.ENUM)


# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")
        Choix_yolo=["yolov7_384x640.onnx","yolov7_480x640.onnx","yolov7-tiny_384x640.onnx"]
        ind=self.properties["Yolo_version"].data
        model_path = "models/"+Choix_yolo[ind]
        self.yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5) 
        self.CmrObject = CameraObject()
              
        
        # Initialize YOLOv7 object detector
        #boxes, scores, class_ids = yolov7_detector(image)

        #self.combined_img = self.yolov7_detector.draw_detections(self.inputs["image_in"].ioelt.data.image_data)
              
    

# Core() is called every time you have a new input
    def Core(self):
    
        classIdsOut = rtmaps.types.Ioelt()
        boxesOut = rtmaps.types.Ioelt()
        scoresOut=rtmaps.types.Ioelt()
        EmptyIoelt1 = rtmaps.types.Ioelt()
        EmptyIoelt2 = rtmaps.types.Ioelt()


        #--------------------------------------------------
        frame = self.inputs["image_in"].ioelt.data
        timestamp = self.inputs["image_in"].ioelt.ts
        self.CmrObject.boxes, self.CmrObject.scores, self.CmrObject.class_ids = self.yolov7_detector(frame.image_data)
        
        FilteredObj = filteredIds(self.CmrObject)
        

        for obj in FilteredObj.class_ids:
            if obj > 3: 
                print('alert')
        


        #print(scores)
        EmptyIoelt1.ts = timestamp
        EmptyIoelt2.ts = timestamp

        EmptyIoelt1.data = np.array([],dtype=np.uint64)
        EmptyIoelt2.data = np.array([],dtype=np.float64)

       
        Ids=[]
        if len(FilteredObj.class_ids)>0:
            for i in range(0,len(FilteredObj.class_ids)):
                Ids.append(i)

        
        combined_img=cp.copy(frame)
        combined_img.image_data = self.yolov7_detector.draw_detections(frame.image_data)
        combined_img.ts=timestamp
        self.outputs["image_out"].write(combined_img) 

        boxesOut.data = prepareBoxes(FilteredObj.boxes,Ids)
        classIdsOut.data = prepareIds(FilteredObj.class_ids,Ids)
        boxesOut.ts = timestamp
        classIdsOut.ts = timestamp
        

        scoresOut.data = prepareConfs(FilteredObj.scores,Ids)
        scoresOut.ts = timestamp
       
        #boxes = np.array(boxes,dtype=np.float64)
        #boxes = boxes.flatten()
        #FilteredObj.scores = np.array(FilteredObj.scores,dtype=np.float64)
        #FilteredObj.scores = FilteredObj.scores.flatten()
        #FilteredObj.class_ids = np.array(FilteredObj.class_ids,dtype=np.int64)
        #FilteredObj.class_ids = FilteredObj.class_ids.flatten()
        
        if len(classIdsOut.data)>0:
            self.outputs["boxes"].write(boxesOut) 
            self.outputs["scores"].write(scoresOut) 
            self.outputs["class_ids"].write(classIdsOut) 
            for i in FilteredObj.class_ids:
                self.outputs["label"].write(class_names[i]+"\n") # and write it to the output
        else:
           self.outputs["boxes"].write(EmptyIoelt1)
           self.outputs["scores"].write(EmptyIoelt2)
           self.outputs["class_ids"].write(EmptyIoelt1)


# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
