import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent  # base class
from PIL import Image

import copy as cp
import cv2
import numpy as np

def find_points_of_color(image, color):
    """
    Extracts a list of points from an image that correspond to pixels of a given color.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        color (tuple): The color to search for, represented as a tuple of (B,G,R) values.

    Returns:
        list: A list of (x,y) coordinates that correspond to pixels of the given color.
    """
    # Convert the image to the HSV color space

    # Define a color range to search for in the image
    lower_color = np.array(color)
    upper_color = np.array(color)

    # Threshold the image to extract pixels within the color range
    mask = cv2.inRange(image, lower_color, upper_color)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract the (x,y) coordinates of the contour points
    points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            points.append((x, y))

    return points


def middle_line_camera(image):
    x_milieu = int(image.shape[1] / 2)
    y_min_milieu = int(image.shape[0] / 2)
    y_max_milieu = image.shape[0]
    return cv2.line(image, (x_milieu, y_min_milieu), (x_milieu, y_max_milieu), (40, 40, 0), 2)


def linear_middle_point(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    coefficients = np.polyfit(x, y, 1)
    return coefficients


import math


def angle_droites_verticale(m):
    if 0 > m:
        return math.atan(-1 / m)#+3.0740588427461506
    if 0 < m:
        return math.atan(1 / m)#+3.0740588427461506


def angle_entre_droites_verticale_et_ax_b(a, b):
    # Calcul de l'angle alpha en radians
    alpha = math.atan(a)

    # Calcul de l'angle beta en radians
    beta = math.atan(1 / math.tan(alpha))

    # Conversion en degrés et renvoi du résultat
    return math.degrees(beta)


def mean_of_points(points1, points2):
    """
    Computes the mean of each corresponding point in two lists of (x,y) coordinates.

    Args:
        points1 (list): The first list of (x,y) coordinates.
        points2 (list): The second list of (x,y) coordinates.

    Returns:
        list: A new list of (x,y) coordinates that contains the mean of each corresponding point.
    """
    # Check that the two input lists have the same length
    if len(points1) != len(points2):
        raise ValueError("Input lists must have the same length")

    # Compute the mean of each corresponding point in the two lists
    mean_points = []
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        mean_x = (x1 + x2) / 2
        mean_y = (y1 + y2) / 2
        mean_points.append((mean_x, mean_y))

    return mean_points


def draw_points(image, points, radius=5, color=(255, 0, 0), thickness=-1):
    """
    Draws a list of (x,y) coordinates as circles on an image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        points (list): A list of (x,y) coordinates to draw as circles.
        radius (int): The radius of the circle to draw (default: 5).
        color (tuple): The color of the circle as a tuple of (B,G,R) values (default: red).
        thickness (int): The thickness of the circle boundary. Use -1 to fill the circle (default: -1).

    Returns:
        numpy.ndarray: The image with circles drawn at the specified points.
    """
    # Convert the image to RGB color space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw a circle at each (x,y) coordinate
    for point in points:
        x, y = point
        cv2.circle(rgb_image, (x, y), radius, color, thickness)

    # Convert the image back to BGR color space
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    return bgr_image


def comparer(liste1, liste2):
    res = []
    res1 = []
    for i in range(len(liste1)):
        for j in range(len(liste2)):
            if liste2[j][1] == liste1[i][1]:
                res.append(liste2[j])
                res1.append(liste1[i])
    return res, res1


def keep_middle_line(image, lines):
    largeur = image.shape[0]
    middle = int(largeur / 2)
    x_value = []
    y_value = []
    positive_value = []
    negative_value = []
    point1 = []
    point2 = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        point1.append((x1, y1))
        point2.append((x2, y2))
        x_value.append(x1)

    for i in range(len(x_value)):
        a = x_value[i] - middle
        if a < 0:
            positive_value.append(x_value[i])
        else:
            negative_value.append(x_value[i])

    Lline = x_value.index(max(negative_value))
    try:
        Rline = x_value.index(min(positive_value))
    except:
        Rline = 1
    return (point1[Lline], point2[Lline]), (point1[Rline], point2[Rline])




def process_image(image):
    camMat = np.matrix(
        '498.608787910998	0	320.214224026595; 0	501.376570341078	239.80564405617;0	0	1.',
        dtype=np.float32)
    distVec = np.array(
        (0.205507749348631, -0.538550196466212, -0.00189031688941969, -0.000292445745341019, 0.40952455096272),
        dtype=np.float32)
    undistorted_img = cv2.undistort(image, camMat, distVec)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.convertScaleAbs(gray)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    height, width = image.shape[:2]
    vertices = np.array([[(0, height*0.80), (width, height*0.80), (width * 0.8, height * 0.35), (width * 0.2, height*0.35)]],
                        dtype=np.int32)
    # Création du masque
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)

    masked_image = cv2.bitwise_and(edges, mask)

    if np.sum(edges) == 0:
        print("No edges found in image")

    else:
        lines_image = np.zeros_like(image)
        lines = cv2.HoughLinesP(masked_image, rho=1, theta=np.pi / 180, threshold=50, minLineLength=15, maxLineGap=25)
        coeff_droite = []
        y_vert = int(image.shape[0] * 0.35)
        xl, xr = keep_middle_line(image, lines)
        # print(xl,xr)
        lines = []
        lines.append(xl)
        lines.append(xr)
        color = (0, 255, 0)
        color1 = (255, 0, 0)
        for i in range(len(lines)):
            try:
                a, b = lines[i]
                x1, y1 = a
                x2, y2 = b
                a = (y2 - y1) / (x2 - x1)
                if -0.5 > a:
                    b = y1 - a * x1
                    x_start = int((y_vert - b) / a)
                    x_end = int((height - b) / a)
                    cv2.line(image, (x_start, y_vert), (x_end, height), color, 2)
                if 0.5 < a:
                    b = y1 - a * x1
                    x_start = int((y_vert - b) / a)
                    x_end = int((height - b) / a)
                    cv2.line(image, (x_start, y_vert), (x_end, height), color1, 2)
            except:
                pass
    image = middle_line_camera(image)
    img = image
    color = (0, 255, 0)
    color1 = (255, 0, 0)
    points = find_points_of_color(img, color)
    points1 = find_points_of_color(img, color1)
    points = sorted(points, key=lambda p: p[0])
    points1 = sorted(points1, key=lambda p: p[0])
    ok, okk = comparer(points, points1)
    mean_point = mean_of_points(ok, okk)
    angle=0
    try:
        a, b = linear_middle_point(mean_point)
        angle = angle_droites_verticale(a)
    except:
        pass
    for point in mean_point:
        x, y = point
        center = (int(x), int(y))
        cv2.circle(img, center, radius=1, color=(255, 255, 0), thickness=-1)

    return img,angle,lines






class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)

    def Dynamic(self):
        pass
        # Entree
        self.add_input("image_in", rtmaps.types.MATRIX)
        self.add_input("image", rtmaps.types.IPL_IMAGE)
        # Sortie
        
        self.add_output("image2", rtmaps.types.IPL_IMAGE)
        self.add_output("lane_view", rtmaps.types.IPL_IMAGE)
        self.add_output("angle", rtmaps.types.FLOAT64,1)
        self.add_output("ext_point", rtmaps.types.MATRIX)
        #self.add_output("angle_car", rtmaps.types.FLOAT64, 100)

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

    # Core() is called every time you have a new input
    def Core(self):
        
        print("ok")
        frame1 = self.inputs["image"].ioelt.data
        timestamp1 = self.inputs["image"].ioelt.ts
        angle_car =  rtmaps.types.Ioelt()
        angle_car.ts = timestamp1
        combined_img=cp.copy(frame1)
        combined_img.ts = timestamp1
        point_mat = rtmaps.types.Ioelt()
        point_mat.ts = timestamp1
        #numpy_matrix = rtmaps_converter.to_numpy(frame)
        agl=0
        try:
            img,agl,lane = process_image(frame1.image_data)
            print(img.shape)
        except:
            img = frame1.image_data
            agl=0
        print(agl)
        angle_car.data=agl
        combined_img.data = img
        point_mat.data.matrix_data = np.array(point_mat)
        self.outputs["image2"].write(frame1)
        self.outputs["lane_view"].write(combined_img)
        if not angle_car.data:
            angle_car.data = 0
        self.outputs["angle"].write(angle_car)
        self.outputs["ext_point"].write(point_mat)
        #except:
            #print("here")
            #img = combined_img
            #self.outputs["image_out"].write(img) 

        # Ecriture sur la sortie

    # Death() will be called once at diagram execution shutdown
    def Death(self):
        pass

