import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent # base class

import cv2
#################
#FONCTIONS PERSO#
#################

#PARAMETRES INTRINSEQUES DES CAMERAS
def camera4K():
    #Definitions paramètres intrinsèques
    camMat = np.matrix('835.7750 0 652.2803; 0  844.0353  344.2802;0  0   1.0000',dtype=np.float32)
    distVec = np.array((-0.320287537148166,	0.0923425722591583 ,0.00120289394268330,	-0.00144428192413888),dtype=np.float32)
    return camMat, distVec

def cameraIntegree():
     #Definitions paramètres intrinsèques
    camMat = np.matrix('941.148348547665	0	631.113126521286; 0	943.626173645712	351.866863348675 ;0	0	1',dtype=np.float32)
    distVec = np.array((0.0868182596096991,	-0.184244451637343 ,-0.00789726238200264,	0.000136073492180380),dtype=np.float32)
    return camMat, distVec

def cameraUSB():
     #Definitions paramètres intrinsèques
    camMat = np.matrix('600.5043 0 266.8958; 0  600.9746  181.4191;0  0   1.0000',dtype=np.float32)
    distVec = np.array((-0.503575932929749,	0.267527768089534,-0.000418809726227716,	-0.00245646730296180),dtype=np.float32)
    return camMat, distVec

def cameraTwizzy():
    camMat = np.matrix('1.2333307880480911e+003 0. 6.5608259821013826e+002; 0. 1.2419473055412493e+003 5.5885526096352805e+002 ;0. 0. 1.',dtype=np.float32)
    distVec = np.array((-1.1903351805055828e-001, 1.9497572973143704e-001,
    5.5272294360454990e-003, 3.8408944506890742e-003,7.9069991990030047e-003),dtype=np.float32)
    return camMat,distVec

#FONCTIONS UTILES AU CALCUL Pixel->Position

def rotationMatrix(Roll,Pitch,Yaw):
    #Retourne la matrice de rotation totale (equivaut à trois rotations sur les axes x,y,z)

    #Rotation sur x
    Cr = np.cos(np.deg2rad(Roll))
    Sr = np.sin(np.deg2rad(Roll))
    R_x = np.matrix([[1, 0, 0],[0,Cr,-Sr],[0, Sr, Cr]])

    #Rotation sur y
    Cp = np.cos(np.deg2rad(Pitch))
    Sp = np.sin(np.deg2rad(Pitch))
    R_y = np.matrix([[Cp, 0, Sp],[0,1,0],[-Sp, 0, Cp]])

    #Rotation sur z
    Cy = np.cos(np.deg2rad(Yaw))
    Sy = np.sin(np.deg2rad(Yaw))
    R_z = np.matrix([[Cy, -Sy, 0],[Sy, Cy, 0],[0, 0, 1]])

    return R_x*R_y*R_z

def intersecLine_Plane(point,vector,plane,d):

    # Calcul l'intersection entre la droite passant par le point, 
    # de vecteur directeur vector et le plan

    A = np.dot(plane,vector)
    B = np.dot(plane,point)

    t = (d - B)/A

    intersec = np.array(point+vector*t)
    return intersec[0:2]

def pixel2World(p,zp,hauteur_cam,inclinaison_cam,camMat,distortion):
    #Creation point
    input_P = np.zeros((1,1,2),dtype=np.float32)
    input_P[0,0,0] = p[0]
    input_P[0,0,1] = p[1]
    
    #Paramètres extrinsèques 
    h = hauteur_cam    # Hauteur camera

    Roll = -90 - inclinaison_cam     #Rot sur x           
    Pitch = 0     #Rot sur y
    Yaw = 0       #Rot sur z
    
    camPose = np.array([0,0,h])

    #Calcul point sans distortion
    up = cv2.undistortPoints(input_P,camMat,distortion)
    x = up[0,0,0]
    y = up[0,0,1]
    
    #Vecteur
    norm = np.sqrt(x*x + y*y + 1) 
    vec = np.matrix((x/norm,y/norm,1/norm))

    
    #Calcul matrice de rotation
    rot = rotationMatrix(Roll,Pitch,Yaw)
    
    #Transfo coordonnées du vecteur dans le monde
    vec = np.matmul(rot,np.transpose(vec))
    #vec = np.reshape(vec,[1,3])
    vec = np.squeeze(np.asarray(vec))
    #print("vec" +str(vec))
    #Calcul du point intersection 
    point = intersecLine_Plane(camPose,vec,[0,0,1],zp)

    #Matrice de rotation plan
    c = np.cos(np.deg2rad(-90))
    s = np.sin(np.deg2rad(-90))
    Rp = np.matrix([[c,-s],[s,c]])

    point = np.matmul(Rp,point)
    return point

#############
#BLOC RTMAPS#
#############

class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        self.force_reading_policy(rtmaps.reading_policy.SYNCHRO) #Pour lire les 3 entrees en même temps.

    def Dynamic(self):
         #ENTREES
        self.add_input("tIds", rtmaps.types.UINTEGER64)
        self.add_input("tBoxes", rtmaps.types.UINTEGER64)
        self.add_input("tClassIds", rtmaps.types.UINTEGER64) 

        #SORTIES
        self.add_output("Objects", rtmaps.types.REAL_OBJECT,50) 

        #PROPRIETES
        self.add_property("Hauteur_camera_m",1.71)
        self.add_property("Inclinaison_deg",0.0)
        self.add_property("Choix_camera","4|0|Camera-USB|Camera-integree|Camera-4K|Camera-Twizzy",rtmaps.types.ENUM)

    def Birth(self):
        
        #Lecture des proprietes camera (Choix de la camera pour paramètre intrinseques ET EXTRINSEQUES)
        self.h_cam = self.properties["Hauteur_camera_m"].data
        self.incli_cam = self.properties["Inclinaison_deg"].data
        #choix_cam = int(self.properties["Choix_camera"].data)

        switcher = {
        0:cameraUSB,
        1:cameraIntegree,
        2:camera4K,
        3:cameraTwizzy
        }

        #Retourne les bons paramètres intrinseques en fonction de lu choix de la camera
        #self.camMat, self.distortion = switcher.get(choix_cam)()
        self.camMat, self.distortion = cameraTwizzy()
        #self.camMat = np.matrix('1.2333307880480911e+003 0. 6.5608259821013826e+002; 0. 1.2419473055412493e+003 5.5885526096352805e+002 ;0. 0. 1.',dtype=np.float32)
        #self.distortion = np.array((-1.1903351805055828e-001, 1.9497572973143704e-001, 5.5272294360454990e-003, 3.8408944506890742e-003,7.9069991990030047e-003),dtype=np.float32)
        
    def Core(self):
        # Lecture des entrees / Initialisation de la sortie / Allocation memoire tableau
        iDs = self.inputs["tIds"].ioelt.data
        classIDs = self.inputs["tClassIds"].ioelt.data
        iBoxes = self.inputs["tBoxes"].ioelt.data

        # Lecture du timestamp
        ts = self.inputs["tIds"].ioelt.ts

        # Calcul nombre d'objet
        nObjets = self.inputs["tIds"].ioelt.vector_size

        #Creation de l'element de sortie pour les objets
        objectsOut = rtmaps.types.Ioelt()
        objectsOut.data = []
        objectsOut.ts = ts
        
        # Traitement des boites
        Boxes = [iBoxes[i*4:i*4+4] for i,id in enumerate(iDs)]

        # Boucler pour chaque track la distance et créer un RealObjet, ajouter ce realObjext au tableau
        if(nObjets > 0):
            for tID,classID,Box in zip(iDs,classIDs,Boxes):
                
                newObjet = rtmaps.types.RealObject() #On créé un nouvel objet
                newObjet.kind = 0  
                newObjet.id = tID #Attribution ID
                newObjet.color = 0xFCFF00 #Couleur de l'objet

                pixel = (Box[0]+Box[2]/2.0,Box[1]+Box[3]) #On prend le pixel du segment bas au mileu de la boite
                
                coord = pixel2World(pixel,0,self.h_cam,self.incli_cam,self.camMat,self.distortion) #Transformation pixel2World
                
                #Ecriture des coordonées xyz
                newObjet.x = coord[0,0]
                newObjet.y = coord[0,1]
                newObjet.z = 0.9

                #Specifier un type vehicule permet d'avoir accès aux attributs de taille de l'objet
                newObjet.data = rtmaps.types.Vehicle() 
                
                newObjet.data.width = 0.3
                newObjet.data.length = 0.3
                newObjet.data.height = 1.8
                
                #Classification de l'objet dans misc1
                newObjet.misc1 = classID

                #Ajout dans le tableau de sortie
                objectsOut.data.append(newObjet)
            self.outputs["Objects"].write(objectsOut)

        # Si pas d'objet on renvoi un objet qui n'existe pas vraiment, de coordonnées (-100,0). 
        # Car les données du temps precedent sont envoyés à l'infini quand on
        # ecrit rien sur la sortie (bug bloc python). Mettre un bloc filtre zone 
        # permet alors de supprimer cet objet imaginaire.
        else: 

            a = rtmaps.types.RealObject()
            a.x = -100
            a.y = 0
    
            self.outputs["Objects"].write([a],ts)
            
    def Death(self):
        pass
