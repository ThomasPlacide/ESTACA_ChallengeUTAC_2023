# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from matplotlib.patches import Polygon
#384x640


def polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    return abs(area / 2.0)

def coef(rgb_image, flag=30):    
    mask = np.zeros_like(rgb_image)
    cv2.fillPoly(mask, [vertices], color=(255, 255, 255))
    masked_img = cv2.bitwise_and(rgb_image, mask)

    cv2.polylines(img=masked_img, pts=[vertices], isClosed=True, color=(0,0,255), thickness=2, lineType=cv2.LINE_8)
    
    img_gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
    seuil, img_seuillee = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
    
    count=cv2.countNonZero(img_seuillee)
    coef = 2 * count * 100 / area
    cv2.putText(masked_img, ('{:.4f} flag'.format(coef) if coef>flag else '{:.4f}'.format(coef)), org=(550,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255))
    cv2.imshow('Crosswalk_detect', masked_img)
    
    if (coef > flag):
        return True
    return False

l = 1024 
L = 1280
vertices = np.array([(0, l),(0, l*0.43),(0.45*L, 0.26*l),(0.67*L, 0.26*l),(L, 0.35*l),(L, l)], dtype=int)
area = polygon_area(vertices)
vertices = vertices.reshape((-1, 1, 2))



cap = cv2.VideoCapture('C:/Users/paulg/Downloads/Crosswalks_Detection/video.avi')
if (cap.isOpened()== False):
    print("Error opening video file")


while(cap.isOpened()):
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    print(coef(frame))
cap.release()
cv2.destroyAllWindows()



