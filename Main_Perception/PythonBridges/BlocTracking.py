#This is a template code. Please save it in a proper .py file.
import rtmaps.types
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent # base class 

from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from numpy.lib.function_base import append
from numpy.core.fromnumeric import transpose
from numpy.lib.twodim_base import eye
from numpy.linalg import inv
import copy as cp
from PIL import Image as im
import copy as cp
from yolov7 import YOLOv7

###################
# FONCTIONS PERSO #
###################

#FILTRE OBJETS
def filtreObjets(Ids,Boxes): #Retourne les infos des boites pietons, chaque ligne correspond à une boite (x,y,w,h)
    BoxesOut =  np.array([Boxes[i*4:i*4+4] for i,id in enumerate(Ids) if id <= 3])
    ClassificationIds = np.array([id for id in Ids if id <= 3])
    return ClassificationIds,BoxesOut
	
#FONCTION TRACKING
def IOU(boxA, boxB):
	"""Calcul IOU pour deux boites"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def yoloBox_to_corner(Boxes):
	"""Transformation (x,y,w,h) en (x1,y1,x2,y2) [Coin haut gauche, width, height -> Coin haut gauche, Coin bas droit]"""
	x,y,w,h = Boxes
	x1 = x 
	y1 = y 
	x2 = x + w
	y2 = y + h
	return [x1,y1,x2,y2]

def computeMatrixIOU(detectionIds,trackingIds,detectionBoxes,trackingBoxes):
	"""Calcul de la matrice IOU entre la liste de detection et la liste de tracking"""

	# Initialisation Matrice IOU
	MatrixIOU = np.zeros((len(trackingIds),len(detectionIds)))

	# Boucler pour chaque tracking, pour chaque detection
	for i,tBoxe, in enumerate(trackingBoxes):

		tBoxe = yoloBox_to_corner(tBoxe) #Preparation dans le bon format pour calcul IOU

		for j,dBoxe, in enumerate(detectionBoxes):
			
			dBoxe = yoloBox_to_corner(dBoxe) #Preparation dans le bon format pour calcul IOU
			
			# Calcul IOU
			MatrixIOU[i,j] = IOU(dBoxe,tBoxe)

		
	return MatrixIOU

def prepareAssociation(MatrixIOU,detectionIds,trackingIds,seuil_iou):
	
	#Utilisation de linear_sum_assignment (Algorithme hongrois)
	row_ind,col_ind = linear_sum_assignment(-MatrixIOU) 

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
	
	#On verifier que le seuil d'IOU est depassé sinon on ne considère pas d'association
	for m in matched_idx:
		if (MatrixIOU[m[0],m[1]]<seuil_iou): 
			unmatched_detection.append(m[1])
			unmatched_trackers.append(m[0])
		else:
			matches.append(m.reshape(1,2))
	
	if(len(matches) == 0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis = 0)
	
	return matches, np.array(unmatched_detection),np.array(unmatched_trackers)
	
def Association(detectionIds,trackingIds,detectionBoxes,trackingBoxes,seuil_iou):

	#Calcul de la matrice IOU
	MatIOU = computeMatrixIOU(detectionIds,trackingIds,detectionBoxes,trackingBoxes) 

	#Preparation des associations (association des trackers et detections), detection sans asso, et trackers sans asos
	matches,unmatched_detections,unmatched_trackers = prepareAssociation(MatIOU,detectionIds,trackingIds,seuil_iou) 
	
	#Tableaux qui vont stocker les boites et Ids des trackers associés avec les detections
	nBoxes = []
	nIds = []

	#Pour chaque association
	for row in matches:
		#On récupère les indices
		indexTracking = row[0] 
		indexDetec = row[1]

		#On ajoute dans les tableaux
		nIds.append(trackingIds[indexTracking])
		nBoxes.append(detectionBoxes[indexDetec])

	return nIds,nBoxes,unmatched_detections,unmatched_trackers

def findSmallestMissingId(arr):
	#Fonction permettant de générer le plus petit entier naturel manquant dans une liste 
    distinct = {*arr}

    index = 0
    while True:
        if index not in distinct:
            return index
        else:
            index +=1

#DRAWIMAGE
def drawImage(Frame,cIds,tIds,Boxes):
	#Fonction qui permet d'ecrire sur une image specifié les tracks detectés
	classes ={
	0:"Pieton",
	1:"Velo",
	2:"Voiture",
	3:"Moto"}

	for cId,tId,Box in zip(cIds,tIds,Boxes):
		
		x1,y1,x2,y2 = yoloBox_to_corner(Box)
		y3 = int(abs(y1 -2))
		text = str(tId) + ", " + classes[cId]
		#Frame = cv2.rectangle(Frame,(x1,y1),(x2,y2),(244,80,60),2)
		#Frame = cv2.putText(Frame,text,(x1,int(y3)),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))

	return Frame

#FILTRE KALMANN
class kalmanFilterBox():

    def __init__(self,classID,trackID,x): #Constructeur du filtre
        self._classID = classID
        self._trackID = trackID
        self._cptr = 0

        self._Q = np.matrix(np.eye(8))*100
        self._P = np.matrix(np.eye(8))*10
        
        x = np.matrix(np.concatenate((x,[0,0,0,0])))

        if np.shape(x)[1] != 1: # On fait en sorte que x soit sous forme de vecteur colonne
            self._x = np.matrix(np.reshape(x,(np.shape(x)[1],1)))
        else:
            self._x = np.matrix(x)	
        
        self._H = np.matrix(np.concatenate((np.matrix(np.eye(4)),np.matrix(np.zeros((4,4)))),axis=1))
        self._R = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,5,0],[0,0,0,5]])

    def prediction(self,dt):
		#Prediction
        self._F = np.matrix([[1,0,0,0,dt,0,0,0],[0,1,0,0,0,dt,0,0],[0,0,1,0,0,0,dt,0],[0,0,0,1,0,0,0,dt],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
        self._x = self._F*self._x # x = F*x
       
        self._P = self._F*self._P*transpose(self._F) # P = F*P*F'

    def update(self,z,dt):
		#Prediction + Mise à jour
        self.prediction(dt)
        z = np.transpose(np.matrix(z))

        y = z - self._H*self._x # y = z - H*x
        
        S = self._H*self._P*transpose(self._H) + self._R # S = H*P*H' + R
        K = self._P*transpose(self._H)*inv(S) # K = P*H'/S
        
        self._x = self._x + K*y # x = x + K*y
        
        self._P = (eye(self._F.shape[0]) - K*self._H)*self._P # P = (I - K*H)*P

    def getCount(self):
		#Retourne le compteur de non detection
        return self._cptr

    def incrementCount(self,value):
		#Augmente le compteur de non detection
        self._cptr += value

    def returnTrack(self):
		#Retourne les informations du tracker (Classe,ID,Boite)
        return self._classID,self._trackID, np.array(self._x[0:4].flatten(),dtype=int).squeeze()
	
    def resetCount(self):
		#Reset le compteur de non detection
        self._cptr = 0

###############
# BLOC RTMAPS #
###############

class rtmaps_python(BaseComponent):
	def __init__(self):
		BaseComponent.__init__(self)# call base class constructor
		self.force_reading_policy(rtmaps.reading_policy.SYNCHRO)
		
		# Listes permettant le tracking des objets (t-1)   
		self.trackingIds = [] 
		self.trackingBoxes = [] 
		# Liste des filtres de kalmann
		self.kalmannFilters = []
		# Liste des id de classification 
		self.classIds = []
		# Reinitialisation du temps, sert à calculer le temps entre deux appels de boucle
		self.t1 = 0

	def Dynamic(self):

		#ENTREES
		self.add_input("Boxes", rtmaps.types.UINTEGER64) 
		self.add_input("Ids", rtmaps.types.UINTEGER64)
		self.add_input("Image",rtmaps.types.IPL_IMAGE)
		self.add_input("Scores",rtmaps.types.ANY)

		#SORTIES
		self.add_output("outImage",rtmaps.types.IPL_IMAGE)
		self.add_output("tIds",rtmaps.types.UINTEGER64,500)
		self.add_output("tBoxes",rtmaps.types.UINTEGER64,2000)
		self.add_output("tClassIds",rtmaps.types.UINTEGER64,500)

		#PROPRIETES
		self.add_property("Temps_max_non_detec_sec",1.5)
		self.add_property("Seuil_IOU",0.4)
		
	def Birth(self):

		#Reset des tableaux
		self.trackingIds = []
		self.trackingBoxes = []
		self.kalmannFilters = []
		self.classIds = []
		self.t1 = 0

		# Temps maximum sans detection (en microsec)
		self.tMaxNoDetec = self.properties["Temps_max_non_detec_sec"].data*1000000
		self.seuilIOU = self.properties["Seuil_IOU"].data
		print("Python Birth")

	def Core(self):

		BoxesIn = self.inputs["Boxes"].ioelt # Lecture des boites de detection de YOLO
		IdsIn = np.array(self.inputs["Ids"].ioelt.data) # Lecture des Ids de detection de YOLO
		nDetec = BoxesIn.vector_size/4 #C alcul du nombre de detections
		ImageOut = cp.copy(self.inputs["Image"].ioelt)
		
		t2 = BoxesIn.ts # Lecture du timestamp
		dt = t2 - self.t1 # Calcul de deltaT (temps entre deux boucle)
		self.t1 = t2 # Ecriture sur t1 pour la boucle prochaine

		if nDetec > 0:

			classIds,Boxes = filtreObjets(IdsIn,BoxesIn.data) # On recupère les boîtes des pietons 
			nObjets = len(Boxes) # Calcul du nombre d'objets filrés detectés
			
			if nObjets > 0:
				Ids =  [i for i in range(0,int(len(Boxes)))] # Attribution d'une id pour chaque detection
				
				if not self.trackingIds: # Si la liste de tracking est vide
					
					print("Initialisation tracking")
					# On initialise le tracking des objets
					self.trackingIds = Ids 
					self.trackingBoxes = Boxes
					self.classIds = classIds

					# Initialisation des filtres de kalmann pour chaque track
					for (cId,tId,Box) in zip(classIds,Ids,Boxes):
						self.kalmannFilters.append(kalmanFilterBox(cId,tId,Box))
				
				else: 
					# Associsation des detections aux tracks par calcul d'IOU + algorithme hongrois
					matched_ids, matched_boxes, unmatched_detections, unmatched_trackers = Association(Ids,self.trackingIds,Boxes,self.trackingBoxes,self.seuilIOU) 
					
					
					nClassIds,nTrackingIds,nTrackingBoxes,newKalmannFilters = [],[],[],[]
					
					#CAS A - Track existant, detection associée : on met à jour le track avec le filtre
					for i,(id,Boxe) in enumerate(zip(matched_ids,matched_boxes)):
						for filtre in self.kalmannFilters:
							if filtre._trackID == id: #On trouve le filtre correspondant
								filtre.update(Boxe,dt) #On met à jour le filtre à l'aide de la detection YOLO
								filtre.resetCount() #On reset le compteur de non detection

								cID,tID,Box = filtre.returnTrack() #On récupère les nouvelle données de ce track

								#On met à jour dans les autres tableaux
								nClassIds.append(cID)
								nTrackingIds.append(tID)
								nTrackingBoxes.append(Box)
								newKalmannFilters.append(filtre)
								
					
					#CAS B - Track existant, pas de detection : on augmente le compteur de non detection, maj grâce à l'estimation du filtre (modèle)
					for i in unmatched_trackers: #Pour chaque tracker sans association
						for filtre in self.kalmannFilters: 
							if filtre._trackID == self.trackingIds[i]: #On cherche son filtre correspondant
								if filtre.getCount() < self.tMaxNoDetec: #Si le compteur n'a pas depassé la valeur max
								
									filtre.incrementCount(dt) #On augmente le compteur
									filtre.prediction(dt) #On fait la prediction

									cID,tID,Box = filtre.returnTrack() #On récupère les nouvelle données de ce track

									#On met à jour dans les autres tableaux
									nClassIds.append(cID)
									nTrackingIds.append(tID)
									nTrackingBoxes.append(Box)
									newKalmannFilters.append(filtre)
					
					#CAS C - Nouvelle detection : Creation nouvau filtre avec la detection comme donnée d'initialisation
					for i in unmatched_detections:
						
						tID = findSmallestMissingId(nTrackingIds) #On attribue un nouvel ID non utilisé
						nBox = Boxes[i] #On stocke la nouvelle boîte
						cID = classIds[i] #On stocke l'ID de classification
						filtre = kalmanFilterBox(cID,tID,nBox) #Creation du nouveau filtre

						#On met à jour dans les autres tableaux
						nClassIds.append(cID)
						nTrackingIds.append(tID)
						nTrackingBoxes.append(Boxes[i])
						newKalmannFilters.append(filtre)
					
					
					#On stocke les tableaux de trackings & filtres pour la boucle suivante
					self.classIds,self.trackingIds,self.trackingBoxes,self.kalmannFilters = nClassIds,nTrackingIds,nTrackingBoxes,newKalmannFilters
					
			else: #Si pas d'objets detectés

				nClassIds,nTrackingIds,nTrackingBoxes,newKalmannFilters = [],[],[],[]
				#On augmente incremente les compteurs des filtres de kalmann
				for filtre in self.kalmannFilters:
					if filtre.getCount() < self.tMaxNoDetec: #Si le compteur n'a pas depassé la valeur max
						
						filtre.incrementCount(dt) #On augmente le compteur
						filtre.prediction(dt) #On fait la prediction
						cID,tID,Box = filtre.returnTrack() #On récupère les nouvelle données de ce track


						#On met à jour dans les autres tableaux
						nClassIds.append(cID)
						nTrackingIds.append(tID)
						nTrackingBoxes.append(Box)
						newKalmannFilters.append(filtre)
					
				self.classIds,self.trackingIds,self.trackingBoxes,self.kalmannFilters = nClassIds,nTrackingIds,nTrackingBoxes,newKalmannFilters	
		else:
			nClassIds,nTrackingIds,nTrackingBoxes,newKalmannFilters = [],[],[],[]
			
			#On augmente les compteurs des filtres de kalmann
			for filtre in self.kalmannFilters:
				if filtre.getCount() < self.tMaxNoDetec: #Si le compteur n'a pas depassé la valeur max
						
					print("PASSAGE PAS DOBJETS")
					filtre.incrementCount(dt) #On augmente le compteur
					filtre.prediction(dt) #On fait la prediction
					cID,tID,Box = filtre.returnTrack() #On récupère les nouvelle données de ce track


					#On met à jour dans les autres tableaux
					nClassIds.append(cID)
					nTrackingIds.append(tID)
					nTrackingBoxes.append(Box)
					newKalmannFilters.append(filtre)
					
			self.classIds,self.trackingIds,self.trackingBoxes,self.kalmannFilters = nClassIds,nTrackingIds,nTrackingBoxes,newKalmannFilters	

		#On affiche sur l'image, les boites, l'ID de track, et la classe de l'objet
		ImageOut.data.image_data = drawImage(ImageOut.data.image_data,self.classIds,self.trackingIds,self.trackingBoxes)
		

		#On ecrit les données sur la sortie
		self.outputs["outImage"].write(ImageOut)
		self.outputs["tIds"].write(np.array(self.trackingIds,dtype=np.uint64),t2) 
		self.outputs["tBoxes"].write(np.array(self.trackingBoxes,dtype=np.uint64).flatten().squeeze(),t2)
		self.outputs["tClassIds"].write(np.array(self.classIds,dtype=np.uint64),t2)

	def Death(self):
		pass
