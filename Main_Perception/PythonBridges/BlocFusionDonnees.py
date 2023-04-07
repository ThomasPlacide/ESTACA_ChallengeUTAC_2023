#Bibliotheques de base

import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent

#Bibliotheques ajoutées
from scipy.optimize import linear_sum_assignment

import numpy as np
from numpy.lib.function_base import append
from numpy.core.fromnumeric import transpose
from numpy.lib.twodim_base import eye
from numpy.linalg import inv

####################
#FONCTIONS AJOUTEES#
####################

#ASSOCIATION

def DIST(coordA, coordB):
    dx = coordA[0] - coordB[0]
    dy = coordA[1] - coordB[1]
    return np.hypot(dx,dy)

def computeMatrixDIST(detectionIds,trackingIds,detectionCoord,trackingCoord):
	"""Calcul de la matrice DIST entre la liste de detection et la liste de tracking"""

	# Initialisation Matrice IOU
	MatrixDIST = np.zeros((len(trackingIds),len(detectionIds)))

	# Boucler pour chaque tracking, pour chaque detection
	for i,coordT, in enumerate(trackingCoord):
		for j,coordD, in enumerate(detectionCoord):
			
            # Calcul DISTANCE 
			MatrixDIST[i,j] = DIST(coordT,coordD)

	return MatrixDIST

def prepareAssociation(MatrixDIST,detectionIds,trackingIds,SEUIL_DIST):
	
	#Utilisation de linear_sum_assignment (Algorithme hongrois)
	B = MatrixDIST<SEUIL_DIST
	C = np.array(MatrixDIST)*np.array(B) + np.array(100*~B)
	row_ind,col_ind = linear_sum_assignment(C) 

	#On categorise chaque trackers / detection en association ou NON
	matched_idx = np.array([row_ind,col_ind])
	matched_idx = np.transpose(matched_idx)

	unmatched_trackers, unmatched_detection = [],[]
	for t,trk in enumerate(trackingIds):
		if(t not in matched_idx[:,0]):
			unmatched_trackers.append(t)
	
	for d,det in enumerate(detectionIds):
		if(d not in matched_idx[:,1]):
			unmatched_detection.append(d)

	matches = []
	
	#On verifie que le seuil de distance est depassé sinon on ne considère pas d'association
	for m in matched_idx:
		if (MatrixDIST[m[0],m[1]]>SEUIL_DIST): # Si la distance est trop élevée
			unmatched_detection.append(m[1])
			unmatched_trackers.append(m[0])
		else:                                  # Sinon on valide l'association
			matches.append(m.reshape(1,2))
	
	if(len(matches) == 0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis = 0)
	
	return matches, np.array(unmatched_detection),np.array(unmatched_trackers)
	
def Association(detectionIds,trackingIds,detectionCoord,trackingCoord,detectionClassIds,SEUIL_DIST):

	#Calcul de la matrice IOU
	MatDIST = computeMatrixDIST(detectionIds,trackingIds,detectionCoord,trackingCoord)

	#Preparation des associations (association des trackers et detections), detection sans asso, et trackers sans asos
	matches,unmatched_detections,unmatched_trackers = prepareAssociation(MatDIST,detectionIds,trackingIds,SEUIL_DIST)

	#Tableaux qui vont stocker les boites et Ids des trackers associés avec les detections
	nCoord = []
	nIds = []
	nClassIds = []

	#Pour chaque association
	for row in matches:
		#On récupère les indices
		indexTracking = row[0] 
		indexDetec = row[1]

		#On ajoute dans les tableaux
		nIds.append(trackingIds[indexTracking])
		nCoord.append(detectionCoord[indexDetec])
		nClassIds.append(detectionClassIds[indexDetec])
	return nIds,nCoord,nClassIds,unmatched_detections,unmatched_trackers

def findSmallestMissingId(arr):
    #Fonction permettant de générer le plus petit entier naturel manquant dans une liste 
    distinct = {*arr}

    index = 0
    while True:
        if index not in distinct:
            return index
        else:
            index +=1

# FILTRE KALMANN

class kalmanFilterClass():

    def __init__(self,classID,trackID,x): #Constructeur du filtre
        
        self._classID = classID #Classe
        self._trackID = trackID #Identifiant unique
        self._age = 0 #Age du track
        self._cptr = 0 #Compteur de non detection

        #Attributs specifiques au filtre
        self._Q = np.matrix(np.ones((4,4)))*0.01
        self._P = np.matrix(np.eye(4))*10
        
        x = np.matrix(np.concatenate((x,[0,0])))

        if np.shape(x)[1] != 1: # On fait en sorte que x soit sous forme de vecteur colonne
            self._x = np.matrix(np.reshape(x,(np.shape(x)[1],1)))
        else:
            self._x = np.matrix(x)	
        
        self._H = np.matrix(np.concatenate((np.matrix(np.eye(2)),np.matrix(np.zeros((2,2)))),axis=1))
        self._R_CAMERA = np.matrix(eye(2))*0.5
        self._R_RADAR = np.matrix(eye(2))*0.4

    def prediction(self,dt):
        #Etape de prediction
        self._F = np.matrix([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self._x = self._F*self._x # x = F*x
       
        self._P = self._F*self._P*transpose(self._F) # P = F*P*F'

    def updateCAMERA(self,z,dt):

        #Prediction + Mise à jour, quand la donnée provient de la camera
        self.prediction(dt)
        z = np.transpose(np.matrix(z))

        y = z - self._H*self._x # y = z - H*x
        
        S = self._H*self._P*transpose(self._H) + self._R_CAMERA # S = H*P*H' + R
        K = self._P*transpose(self._H)*inv(S) # K = P*H'/S
        
        self._x = self._x + K*y # x = x + K*y
        
        self._P = (eye(self._F.shape[0]) - K*self._H)*self._P # P = (I - K*H)*P

    def updateRADAR(self,z,dt):
        #Prediction + Mise à jour, quand la donnée provient du radar
        self.prediction(dt)
        z = np.transpose(np.matrix(z))

        y = z - self._H*self._x # y = z - H*x
        
        S = self._H*self._P*transpose(self._H) + self._R_RADAR # S = H*P*H' + R
        K = self._P*transpose(self._H)*inv(S) # K = P*H'/S
        
        self._x = self._x + K*y # x = x + K*y
        
        self._P = (eye(self._F.shape[0]) - K*self._H)*self._P # P = (I - K*H)*P
    
    def update(self,z,dt,capteur):
        #Test pour appeler la bonne fonction de mise à jour
        if capteur == 0:
            self.updateCAMERA(z,dt)
        if capteur == 1:
            self.updateRADAR(z,dt)

    def getCount(self):
        #Retourne le compteur de non detection
        return self._cptr

    def incrementCount(self,value):
        #Incremente le compteur de non detection
        self._cptr += value

    def getCoord(self):
        #retourne les coordonnées de l'objet
        return np.squeeze(np.array(self._x[0:2]).flatten())

    def getID(self):
        #retourne l'id du filtre
        return self._trackID

    def getClassID(self):
        #retourne la classification de l'objet associé au filtre 
        return self._classID

    def getAllInfos(self):
        #Retourne les coordonnées, l'id, et la classe
        return self.getCoord(),self.getID(),self.getClassID()

    def getAge(self):
        #retourne l'age de l'objet
        return self._age
    
    def growUp(self):
        #augmente l'age de l'objet
        self._age += 1

    def hasNoClass(self):
        #Retourne vrai si l'objet n'est pas classifié
        return self._classID == -1

    def resetCount(self):
        #reset le compteur de non detection
        self._cptr = 0

    def setClassID(self,id):
        #Permet de définir la classe de l'objet associé au filtre
        self._classID = id

#############
#BLOC RTMAPS#
#############

class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor

        #Important pour pouvoir lire une entrée à la fois
        self.force_reading_policy(rtmaps.reading_policy.REACTIVE) 
        
        #Initialisation du tableau qui va contenir les filtres
        self.kalmannFilters = []
        self.t1 = 0

    def Dynamic(self):

        #INPUTS
        self.add_input("ObjetsCamera", rtmaps.types.REAL_OBJECT) #0
        self.add_input("ObjetsRadar", rtmaps.types.REAL_OBJECT) #1

        #OUTPUTS
        self.add_output("Objets1", rtmaps.types.REAL_OBJECT,50) # define output

        #PROPERTIES
        self.add_property("tMaxNoDetec",1.0)
        self.add_property("seuil_dist",1.0)
        self.add_property("seuil_age",1)
   
    def Birth(self):
        print("Python Birth")

        #Reinitialisation du tableau qui contient les filtres
        self.kalmannFilters = []

        #Lecture des valeurs des propriétés du bloc
        self.tMaxNoDetec = self.properties["tMaxNoDetec"].data*1000000
        self.seuil_dist = self.properties["seuil_dist"].data
        self.SEUIL_AGE = self.properties["seuil_age"].data

        #Reinitialisation de la variable temporelle
        self.t1 = 0

    def Core(self):
        #On recupère l'entrée activante du bloc (Camera Ou Radar)
        capteur = self.input_that_answered

        if capteur == 0: #Si la camera est activante
            ioeltIn = self.inputs["ObjetsCamera"].ioelt

        elif capteur == 1: #Si le radar est activant
            ioeltIn = self.inputs["ObjetsRadar"].ioelt
          
        #Informations temporelles
        ts = ioeltIn.ts
        dt = ts - self.t1
        self.t1 = ts

        #Lecture données objets
        ObjectsIn = ioeltIn.data
        nOBJECTS = ioeltIn.vector_size

        #Creation élément d'ecriture sortie
        IoeltOut = rtmaps.types.Ioelt()
        IoeltOut.ts = ts
        IoeltOut.data = []

        if nOBJECTS > 0:

            #Preparation données pour association
            dCoords,dIds,dClassIds = [],[],[]
            for i,Obj in enumerate(ObjectsIn):
                coord = [Obj.x,Obj.y]
                dCoords.append(coord)
                dIds.append(i)
                if capteur == 0:
                    dClassIds.append(Obj.misc1)
                if capteur == 1:
                    dClassIds.append(-1)

            if not self.kalmannFilters: #Si tracking vide ou non initialisé
                for nID,nCOORD,nCLASSID in zip(dIds,dCoords,dClassIds):
                    self.kalmannFilters.append(kalmanFilterClass(nCLASSID,nID,nCOORD))
            else:
                
                tCoords,tIds = [],[]
                for i,filtre in enumerate(self.kalmannFilters):
                    coord = filtre.getCoord()
                    tID = filtre.getID()
                    tCoords.append(coord)
                    tIds.append(tID)
                
                #Association
                matched_Ids,matched_Coords,matched_ClassIds,unmatched_detections,unmatched_trackers, = Association(dIds,tIds,dCoords,tCoords,dClassIds,self.seuil_dist) 
            
                nKalmannFilters = []

                #CAS A - Track existant, detection associée : on met à jour le track avec le filtre, reset compteur non detection, augmentation de l'âgex
                for i,(id,coord) in enumerate(zip(matched_Ids,matched_Coords)):
                    for filtre in self.kalmannFilters: #On parcourt les filtres
                        if filtre.getID() == id: #On trouve le filtre associé
                            filtre.update(coord,dt,capteur) #On met à jour le filtre avec la mesure 
                            filtre.resetCount() #Reset le compteur de non detection
                            filtre.growUp() #Augmentation de l'âge

                            if capteur == 0: #Si la detection vient de la camera on met à jour la classe
                                filtre.setClassID(matched_ClassIds[i])
                            
                            nKalmannFilters.append(filtre) #On ajoute au nouveau tableau de filtres   
                
                #CAS B - Track existant, pas de detection : on augmente le compteur de non detection, maj grâce à l'estimation du filtre (modèle)
                for i in unmatched_trackers:
                    for filtre in self.kalmannFilters: #On parcour les filtres
                        if filtre.getID() == tIds[i]: #On trouve le filtre associé
                            if filtre.getCount() < self.tMaxNoDetec: #Si le compteur de non detection ne depasse pas le seuil

                                filtre.incrementCount(dt) #On augmente le compteur
                                filtre.prediction(dt) #On predit son etat

                                nKalmannFilters.append(filtre) #On ajoute au nouveau tableau de filtres 
                
                nIds = [filtre.getID() for filtre in nKalmannFilters] #On recupère les ids des filtres existants
                
                #CAS C - Nouvelle detection : Creation nouvau filtre avec la detection comme donnée d'initialisation
                for i in unmatched_detections: #Pour chaque detection sans asso
                    tID = findSmallestMissingId(nIds) #On créé un nouvel ID (le plus petit disponible)
                    nCoord = dCoords[i] #On recupère les coordonnées
                    
                    if capteur == 0:
                        cID = ObjectsIn[i].misc1 # Classification retournée par la camera
                    else:
                        cID = -1 # -1 : Aucune classe

                    filtre = kalmanFilterClass(cID,tID,nCoord) #On créé un filtre
                    nKalmannFilters.append(filtre) #On ajoute au filtre
                
                self.kalmannFilters = nKalmannFilters #On stocke les filtres pour le prochain appel

        else: #Si pas d'objets detectés
           
            nKalmannFilters = []

            #On met à jour les etats des filtres existants si ils sont encore valides
            for filtre in self.kalmannFilters: 
              if filtre.getCount() < self.tMaxNoDetec:
                    filtre.incrementCount(dt)
                    filtre.prediction(dt)

                    nKalmannFilters.append(filtre)
        
            self.kalmannFilters = nKalmannFilters

        #ECRITURE DES DONNEES DANS L'ELEMENT DE SORTIE
        objectsOut = []
        for filtre in self.kalmannFilters:
            
            #On ecrit sur la sortie seulement les filtres avec un age minimum
            if filtre.getAge() > self.SEUIL_AGE:   
                coord,tID,classID = filtre.getAllInfos()
                newObjet = rtmaps.types.RealObject()
                newObjet.kind = 0
                newObjet.id = tID
                newObjet.color = 0x00FF40

                # Coordonnées XYZ
                newObjet.x = coord[0]
                newObjet.y = coord[1]
                newObjet.z = 0.5

                # Specifier un type vehicule permet d'avoir accès aux attributs de taille de l'objet
                newObjet.data = rtmaps.types.Vehicle() 
                            
                newObjet.data.width = 1
                newObjet.data.length = 1
                newObjet.data.height = 1
                            
                newObjet.misc1 = classID

                objectsOut.append(newObjet)

        #Si il y a au moins un objet alors on l'ecrit en sortie
        if len(objectsOut) > 0:
            IoeltOut.data = objectsOut
            self.outputs["Objets"].write(IoeltOut)
        #Sinon on renvoit un objet de pose (-100,0), 
        # Car bug en python pour renvoyer une liste de RealObjects vide, 
        # si on ne fais pas ça, le bloc renvoit d'anciennes données.
        # Cela permet donc d'ecraser les données precedentes
        else:
            a = rtmaps.types.RealObject()
            a.x = -100
            a.y = 0
            a.misc1 = -1
            self.outputs["Objets"].write([a],ts)
        
    
    def Death(self):
        pass
