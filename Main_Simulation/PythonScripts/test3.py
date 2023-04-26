# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:29:17 2023

@author: adminvirtu
"""
# Import the CARLA Python API library and some utils

import math 
import random 
import time 
import glob
import os
import sys
import numpy as np
import cv2
import threading
try:
    sys.path.append(glob.glob(r'C:/Users/adminvirtu/Downloads/CARLA_0.9.12/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
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
    print(mask)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract the (x,y) coordinates of the contour points
    points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            points.append((x, y))

    return points


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






def keep_middle_line(image,lines):
    largeur = image.shape[0]
    middle = int(largeur/2)
    x_value = []
    positive_value = []
    negative_value = []
    point1=[]
    point2=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        point1.append((x1,y1))
        point2.append((x2,y2))
        x_value.append(x1)
    
    for i in range(len(x_value)):
        a = x_value[i]-middle
        if a < 0 :
            positive_value.append(x_value[i])
        else : 
            negative_value.append(x_value[i])
    
    
    Lline = x_value.index(max(negative_value))
    try:
        Rline = x_value.index(min(positive_value))
    except:
        Rline = 1
    return (point1[Lline],point2[Lline]),(point1[Rline],point2[Rline])
                
    
        
        


def process_image(image,re):
    image_data = re.raw_data
    image = np.frombuffer(image_data, dtype=np.uint8).reshape(image.height, image.width, 4)

    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    edges = cv2.Canny(blur, 50, 160, apertureSize=3)
    height, width = image.shape[:2]
    roi_vertices = np.array([[(0, height), (width*0.4, height*0.6), (width*0.6, height*0.6), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    masked_image = cv2.bitwise_and(edges, mask)
    if np.sum(edges) == 0:
        print("No edges found in image")
    else:
        lines_image = np.zeros_like(image)
        lines = cv2.HoughLinesP(masked_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=25)
        coeff_droite = []
        y_vert = int(image.shape[0] * 0.6)
        xl,xr = keep_middle_line(image, lines)
        #print(xl,xr)
        lines=[]
        lines.append(xl)
        lines.append(xr)

        for i in range(len(lines)):
            try:
                a,b = lines[i]
                x1, y1 = a
                x2, y2 = b
                a = (y2 - y1) / (x2 - x1)
                if -0.5 > a:
                    b = y1 - a * x1
                    x_start = int((y_vert - b) / a)
                    x_end = int((height - b) / a)
                    cv2.line(image, (x_start, y_vert), (x_end, height), (0, 255, 0), 2)
                if 0.5 < a:
                    b = y1 - a * x1
                    x_start = int((y_vert - b) / a)
                    x_end = int((height - b) / a)
                    cv2.line(image, (x_start, y_vert), (x_end, height), (255, 0, 0), 2)
            except:
                pass
    
    img = image
    color = (0, 255, 0)
    color1 = (255, 0, 0)
    points = find_points_of_color(img, color)
    points1 = find_points_of_color(img, color1)
    print(points1)
    points = sorted(points, key=lambda p: p[0])
    points1 = sorted(points1, key=lambda p: p[0])
    ok, okk = comparer(points, points1)
    mean_point = mean_of_points(ok, okk)
    for point in mean_point:
        x, y = point
        print("yes")
        print(x)
        center = (int(x), int(y))
        cv2.circle(img, center, radius=5, color=(255, 255, 0), thickness=-1)
    return img


def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    

def attach_camera(image, camera, vehicle):
    camera_transform = camera.get_transform()
    vehicle_transform = vehicle.get_transform()
    camera_transform.location += vehicle_transform.location
    camera_transform.rotation += vehicle_transform.rotation
    image.convert(carla.ColorConverter.Raw)
    image.save_to_disk('output/%06d.png' % image.frame)

def pross(image):
    #image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #image = np.reshape(image, (IM_HEIGHT, IM_WIDTH, 4))

    cv2.imshow("", image)
    cv2.waitKey(1)
    return image


def control_cam():
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    camera_data = {'image': np.zeros((image_h, image_w, 4))}

    # Start camera recording
    camera.listen(lambda image: camera_callback(image,camera_data))

    # OpenCV named window for rendering
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB Camera', process_image(camera_data['image'],image))
    cv2.waitKey(1)

    
    # Game loop
    while True:
        
        # Imshow renders sensor data to display
        cv2.imshow('RGB Camera', camera_data['image'])
        # Quit if user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Close OpenCV window when finished
    cv2.destroyAllWindows()
    cv2.stop()

def control_voiture(vehicle,vehicle2):
    
    start_time = time.time() 
    
    while time.time() - start_time < 8.5:
        
        control = carla.VehicleControl(throttle=0.25, steer=0.0, brake=0.0)
        vehicle.apply_control(control)
        control1 = carla.VehicleControl(throttle=0.45, steer=0.0, brake=0.0)
        vehicle2.apply_control(control1)
        
    start_time = time.time()
    while time.time() - start_time < 2:
        
        
    
        control3 = carla.VehicleControl(throttle=0, steer=0, brake=0.5)
        vehicle.apply_control(control3)
        vehicle2.apply_control(control3)
            
      
        
        
        
    start_time = time.time()    
    while time.time() - start_time < 3:
        pass
    
    
    start_time = time.time()
    
    
    
    while time.time() - start_time < 4:
         
        control6 = carla.VehicleControl(throttle=0.4, steer=0, brake=0)
        vehicle.apply_control(control6)
        vehicle2.apply_control(control6)
        



actor_list=[]
# Connect to the CARLA simulator
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Define the location of the pedestrian
    #spawn_point = random.choice(world.get_map().get_spawn_points())
    loc = carla.Location(x=92, y=38.1, z=0.6)
    rot = carla.Rotation(-0.0000746,0.000984,0.0)
    
    spawn_point = carla.Transform(loc,rot)
    
    
    
    #spawn_point.location = world.get_random_location_from_navigation()
    
    
    # Spawn the pedestrian
    blueprint_library = world.get_blueprint_library()
    walker_bp = blueprint_library.find('walker.pedestrian.0003')
    walker_actor = world.spawn_actor(walker_bp, spawn_point)
    actor_list.append(walker_actor)
    
    model3_bp = world.get_blueprint_library().find('vehicle.citroen.c3')
    model3_bp.set_attribute('color', '255,0,0')   # set vehicle's color to red
    #model3_spawn_point = np.random.choice(spawn_points)
    location1=carla.Location(x=109.6, y=84.1, z=0.600000)
    rotation1=carla.Rotation(pitch=0.000000, yaw=-89.609253, roll=0.000000)
    transform1 = carla.Transform(location1,rotation1)
    model3_actor = world.spawn_actor(model3_bp, transform1)
    actor_list.append(model3_actor)
    
    car2 = world.get_blueprint_library().find('vehicle.citroen.c3')
    model3_bp.set_attribute('color', '0,0,255')
    vehicle_location = model3_actor.get_transform().location
    vehicle_direction = model3_actor.get_transform().get_forward_vector()
    vehicle_rotation = model3_actor.get_transform().rotation
    
    new_location = vehicle_location + 10*vehicle_direction
    new_vehicle_location = carla.Location(new_location.x - 4 , new_location.y - 4, new_location.z )
    
    car = world.try_spawn_actor(car2, carla.Transform(new_vehicle_location, vehicle_rotation))
    actor_list.append(car)
    #model3_actor(car2)
    # Passer en mode "Follow" pour suivre l'acteur
    world.get_spectator().set_transform(walker_actor.get_transform())
    #world.get_spectator().set_transform(world.get_spectator().get_transform(), carla.Transform(carla.Location(z=20), carla.Rotation(pitch=15)))
    
    print('succes')
     #Make the pedestrian walk forward
    control1 = carla.WalkerControl()
    control1.speed = 0.75  # m/s
    control1.direction = carla.Vector3D(x=1.0, y=0.0, z=0.0)  # forward
    walker_actor.apply_control(control1)
    
    #Iterate this cell to find desired camera location
    camera_bp = blueprint_library.find('sensor.camera.rgb') 
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_init_trans = carla.Transform(carla.Location(z=2)) #Change this to move camera
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=model3_actor)
    actor_list.append(camera)
    
    # Move spectator behind vehicle to view
    spectator = world.get_spectator() 
    transform = carla.Transform(model3_actor.get_transform().transform(carla.Location(x=-4,z=2.5)),model3_actor.get_transform().rotation) 
    spectator.set_transform(transform)

    time.sleep(0.2)
    spectator.set_transform(camera.get_transform())
    #camera.destroy()

    # Spawn camera
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=car)
    # Get gamera dimensions and initialise dictionary                       
    
    # Create a thread for the vehicle control
    vehicle_thread = threading.Thread(target=control_voiture, args=(car,model3_actor))
    vehicle_thread.start()
    
    #camera.listen(lambda image: threading.Thread(target=pross,args=(image)).start())
    camcam = threading.Thread(target=control_cam)
    camcam.start()

    
     #Wait for a few seconds
    time.sleep(40)
    
    # Définition de la vitesse cible (en m/s)
    #♥target_speed = 1.0
    
    # Définition de la vitesse cible pour le véhicule
    #walker_actor.set_target_velocity(carla.Vector3D(target_speed, 1, 0))
    
    #target_location = carla.Location(x=115, y=84.1, z=0.6)
    #target_speed = 1  # meters per second
    #walker_actor.set_target_location(target_location, target_speed)
    
    #print(walker_actor.velocity)
     #Stop the pedestrian
    #control.speed = 0.0
    #walker_actor.apply_control(control1)

    # Callback stores sensor data in a dictionary for use outside callback
    world.tick()

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
        print('done.')
    sys.exit()