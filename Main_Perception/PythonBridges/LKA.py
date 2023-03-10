
import LKA_lib as L
import rtmaps.types
import numpy as np
import rtmaps.core as rt 
import rtmaps.reading_policy 
from rtmaps.base_component import BaseComponent 

def Weightener(Image): 
    '''
    Pipeline pour détection et inscription de ligne sur un flux vidéo
    '''
    Gray_Image =[]
    HoughLines = []
    Cropped = []
    WeightedImage = []
    
    Gray_Image = L.grayscale(Image)
    Gray_Image = L.gaussian_blur(Image, 3)

    lowTr = 150
    upTr = 350
    Gray_Image = L.canny(Gray_Image, lowTr, upTr)

    # plt.imshow(Gray_Image, cmap='gray')
    # Gray_Image.shape

    trapeze = np.array( [[(0, 350), (600,350), (900, 540), (0, 540)]] )
    Cropped = L.region_of_interest(Gray_Image, trapeze)
    # plt.imshow(Cropped, cmap='gray')

    HoughLines = L.hough_lines(Cropped, rho=2, theta=np.pi/6, threshold=1, min_line_len=4, max_line_gap=25)
    # plt.imshow(HoughLines)

    WeightedImage = L.weighted_img(HoughLines, Image, α=0.8, β=1., γ=0.)
    
    return WeightedImage

class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) 

    def Dynamic(self):
        #Entree
        self.add_input("Image", rtmaps.types.REAL_OBJECT)

        #Sortie
        self.add_output("ImageTraitee", rtmaps.types.AUTO)
        self.add_output("CoordonneesLignes", rtmaps.types.AUTO) 

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        pass

    # Core() is called every time you have a new input
    def Core(self):
        pass

       
    # Death() will be called once at diagram execution shutdown
    def Death(self):
        pass

