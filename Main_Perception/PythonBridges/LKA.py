
import LKA_lib as L
import numpy as np 

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