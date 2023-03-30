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
                
    
        
        


def process_image(image):
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
                    cv2.line(lines_image, (x_start, y_vert), (x_end, height), (0, 255, 0), 2)
                if 0.5 < a:
                    b = y1 - a * x1
                    x_start = int((y_vert - b) / a)
                    x_end = int((height - b) / a)
                    cv2.line(lines_image, (x_start, y_vert), (x_end, height), (255, 0, 0), 2)
            except:
                pass
    img = cv2.addWeighted(image, 0.8, lines_image, 1, 0)
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


import cv2

# Define the function to process each frame
def process_frame(frame):
    # Do some processing on the frame
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return processed_frame

# Open the video file
cap = cv2.VideoCapture(r"C:/Users/thoai-kevin.huynh/Vision par ordi/TP4/test_videos/test_pieton.mp4")

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Create the output window
cv2.namedWindow('output', cv2.WINDOW_NORMAL)

# Loop through the frames
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Process the frame
    processed_frame = process_image(frame)

    # Display the processed frame
    cv2.imshow('output', processed_frame)

    # Wait for a key press
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()

