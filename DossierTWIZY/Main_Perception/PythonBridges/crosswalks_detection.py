# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from matplotlib.patches import Polygon
import rtmaps.core as rt
#384x640

class CrosswalkDetector(): 
    def __init__(self):
        self.l = {0: 384, 
                  1: 480, 
                  2: 384}
        self.L = 640

    def vertices(self):
         
        l_ind = rt.get_property("BlocYolo.Yolo_version")

        return np.array([   (0, self.l[l_ind]),\
                            (0, self.l[l_ind]*0.43),\
                            (0.45*self.L, 0.26*self.l[l_ind]),\
                            (0.67*self.L, 0.26*self.l[l_ind]),\
                            (self.L, 0.35*self.l[l_ind]),\
                            (self.L, self.l[l_ind])], dtype=int).reshape((-1,1,2))
    
    def polygon_area(self):
        vect = self.vertices()
        n = len(vect)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vect[i][0] * vect[j][1] - vect[j][0] * vect[i][1]
        return abs(area / 2.0)

    def area(self): 
        vect = self.vertices()
        return self.polygon_area(vect)
    

    def coef(self, rgb_image, flag=30): 
        vect = self.vertices()   
        mask = np.zeros_like(rgb_image)
        cv2.fillPoly(mask, [vect], color=(255, 255, 255))
        masked_img = cv2.bitwise_and(rgb_image, mask)

        cv2.polylines(img=masked_img, pts=[vect], isClosed=True, color=(0,0,255), thickness=2, lineType=cv2.LINE_8)
        
        img_gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
        seuil, img_seuillee = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
        
        count=cv2.countNonZero(img_seuillee)
        coef = 2 * count * 100 / self.area()
        cv2.putText(masked_img, ('{:.4f} flag'.format(coef) if coef>flag else '{:.4f}'.format(coef)), org=(550,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255))
        cv2.imshow('Crosswalk_detect', masked_img)
        
        if (coef > flag):
            return True
        return False




