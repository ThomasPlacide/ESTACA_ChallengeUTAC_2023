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
    camera.listen(lambda image: camera_callback(image, camera_data))

    # OpenCV named window for rendering
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB Camera', camera_data['image'])
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