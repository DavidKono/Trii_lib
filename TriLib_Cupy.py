import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

#image corresponds to the numpy array being drawn on
#picture is the actual png being pulled from 

#puts selection as xy image
def put_rect(topleft_coordinate, pulled_rect, image):

    mask = pulled_rect[:, :, 3] == 1

    image_region = image[topleft_coordinate[0]:mask.shape[0] + topleft_coordinate[0], topleft_coordinate[1]:mask.shape[1] + topleft_coordinate[1]]
    
    image_region[mask] = pulled_rect[:, :, :3][mask]


#pulls selection of yx picture
def pull_rect(vertices, picture):
    #first need to swap vertices due to y,x image indexing
    vertices = vertices[:, ::-1]

    #sort vertices by y 
    sorted_indices = np.argsort(vertices[:, 1])
    vertices = vertices[sorted_indices] 

    x_min, x_max, y_min, y_max = get_rect(vertices, picture)
    x_range = cp.abs(x_max - x_min + 1)
    y_range = cp.abs(y_max - y_min + 1)

    #returned and also used in pulling
    topleft_coordinate = cp.array([x_min, y_min])

    pulled_rect = cp.zeros((x_range, y_range, 4), dtype=np.uint8)

    #if vertical
    if is_not_vertical(vertices) == False:
        remix_verticals(vertices)

    for i in range(2):
        m1, c1 = calculate_consts(vertices[0 + i], vertices[1 + i])
        m2, c2 = calculate_consts(vertices[0], vertices[2])

        if vertices[0 + i][1] != vertices[1 + i][1]:

            c1, c2, m1, m2 = sort_left_to_right(vertices[1][1], c1, c2, m1, m2)

            pull_loop(c1, c2, m1, m2, vertices[0 + i], vertices[1 + i], pulled_rect, picture, topleft_coordinate)

    return topleft_coordinate, pulled_rect


#polygon pulls 1 scanline from picture
def pull_loop(c1, c2, m1, m2, lower, upper, pulled_rect, picture, topleft_coordinate):

    y_lower = int(lower[1])
    y_upper = int(upper[1]) + 1

    #ensures image is not pulled out of bounds
    if y_lower < 0:
        y_lower = 0
    if y_upper > picture.shape[1]:
        y_upper = picture.shape[1] - 1

    for y in range(y_lower, y_upper):
        x1 = cp.round((y - c1)/m1) 
        x2 = cp.round((y - c2)/m2) 

        #clip 
        clip_and_write(x1, x2, y, pulled_rect, picture, topleft_coordinate)
    return


#clip x's and write to pulled rect
def clip_and_write(x1, x2, y, pulled_rect, picture, topleft_coordinate):
    width = pulled_rect.shape[0] + topleft_coordinate[0]

    if x1 + topleft_coordinate[0] < 0 and x2 + topleft_coordinate[0] >= 0:
        x1 = topleft_coordinate[0]
    elif x1 < 0:
        return
    if x2 > width - 1 and x1 <= width - 1:
        x2 = width - 1
    elif x2 > width - 1:
        return

    x = cp.arange(int(x1), int(x2) + 1)

    pulled_rect[x - topleft_coordinate[0], y - topleft_coordinate[1], :3] = picture[x,y]
    pulled_rect[x - topleft_coordinate[0], y - topleft_coordinate[1], 3] = 1


def get_rect(vertices, picture):

    x_min = cp.amin(vertices[:, 0]).astype(int) 
    if x_min < 0:
        x_min = 0
    y_min = cp.amin(vertices[:, 1]).astype(int)
    if y_min < 0:    
        y_min = 0

    x_max = cp.amax(vertices[:, 0]).astype(int)
    if x_max > picture.shape[0] - 1:
        x_max = picture.shape[0] - 1
    y_max = cp.amax(vertices[:, 1]).astype(int)
    if y_max > picture.shape[1] - 1:  
        y_max = picture.shape[1] - 1

    return x_min, x_max, y_min, y_max



def square_colour_differences(vertices, image1, image2):
    topleft1, poly1 = pull_rect(vertices, image1)
    topleft2, poly2 = pull_rect(vertices, image2)

    mask = poly1[:, :, 3] == 1

    #sum of squares
    sum_square_diff = np.sum(np.sum(np.square(poly1[:, :, :3][mask] - poly2[:, :, :3][mask]), axis=-1))
    return sum_square_diff

def colour_differences(vertices, image1, image2):
    topleft1, poly1 = pull_rect(vertices, image1)
    topleft2, poly2 = pull_rect(vertices, image2)

    mask = poly1[:, :, 3] == 1

    #sum of sums
    sum_square_diff = np.sum(np.sum(np.abs(poly1[:, :, :3][mask] - poly2[:, :, :3][mask]), axis=-1))
    return sum_square_diff



def average_colour(vertices, picture):
    #first need to swap vertices due to y,x image indexing
    vertices = vertices[:, ::-1]

    #sort vertices by y 
    sorted_indices = np.argsort(vertices[:, 1])
    vertices = vertices[sorted_indices] 

    x_min, x_max, y_min, y_max = get_rect(vertices, picture)
    x_range = np.abs(x_max - x_min + 1)
    y_range = np.abs(y_max - y_min + 1)

    

    #returned and also used in pulling
    topleft_coordinate = np.array([x_min, y_min])

    pulled_rect = np.zeros((x_range, y_range, 4), dtype=np.uint8)

    #if vertical
    if is_not_vertical(vertices) == False:
        remix_verticals(vertices)

    total_colour = np.zeros((4))

    for i in range(2):
        m1, c1 = calculate_consts(vertices[0 + i], vertices[1 + i])
        m2, c2 = calculate_consts(vertices[0], vertices[2])

        if vertices[0 + i][1] != vertices[1 + i][1]:

            c1, c2, m1, m2 = sort_left_to_right(vertices[1][1], c1, c2, m1, m2)
            total_colour += pull_colour_loop(c1, c2, m1, m2, vertices[0 + i], vertices[1 + i], pulled_rect, picture, topleft_coordinate)

    avg_colour = total_colour[:3]//total_colour[3]
    return avg_colour

def pull_colour_loop(c1, c2, m1, m2, lower, upper, pulled_rect, picture, topleft_coordinate):

    y_lower = int(lower[1])
    y_upper = int(upper[1]) + 1

    #ensures image is not pulled out of bounds
    if y_lower < 0:
        y_lower = 0
    if y_upper > picture.shape[1]:
        y_upper = picture.shape[1] - 1

    total_colour = np.zeros((4))

    for y in range(y_lower, y_upper):
        x1 = np.round((y - c1)/m1) 
        x2 = np.round((y - c2)/m2) 

        #clip 
        width = pulled_rect.shape[0] + topleft_coordinate[0]
        if x1 < 0 and x2 >= 0:
            x1 = 0
        elif x1 < 0:
            return total_colour
        if x2 > width - 1 and x1 <= width - 1:
            x2 = width - 1
        elif x2 > width - 1:
            return total_colour

        total_colour[3] += x2 - x1

        x = np.arange(int(x1), int(x2) + 1)
        temp = picture[x,y]
        total_colour[:3] += np.sum(temp, axis = 0)
        
    return total_colour



def fill(vertices, image, colour):
    #sort vertices by y 
    sorted_indices = np.argsort(vertices[:, 1])
    vertices = vertices[sorted_indices]

    #corresponds to triangles that have only horizontal and non-vertical lines
    if is_not_vertical(vertices):
        draw(vertices, image, colour)
    else:
        remix_verticals(vertices)
        draw(vertices, image, colour)
    return image

#prevent edge cases by making minor adjustment to vertices
def remix_verticals(vertices):
    if vertices[0][0] == vertices[1][0]:
        vertices[1][0] += 1
    if vertices[1][0] == vertices[2][0]:
        vertices[2][0] += 1
    if vertices[0][0] == vertices[2][0]:
        vertices[2][0] -= 1

def is_not_vertical(vertices):
    if vertices[0][0] != vertices[1][0] and vertices[1][0] != vertices[2][0] and vertices[0][0] != vertices[2][0]:
        return True
    else:
        return False

def draw(vertices, image, colour):
    for i in range(2):
        m1, c1 = calculate_consts(vertices[0 + i], vertices[1 + i])
        m2, c2 = calculate_consts(vertices[0], vertices[2])

        if vertices[0 + i][1] != vertices[1 + i][1]:
            c1, c2, m1, m2 = sort_left_to_right(vertices[1][1], c1, c2, m1, m2)
            draw_loop(c1, c2, m1, m2, vertices[0 + i], vertices[1 + i], image, colour)
            

def calculate_consts(lower, higher):
    m = (higher[1] - lower[1])/(higher[0]-lower[0])
    c = lower[1] - m*lower[0]

    return m, c

#resorts eq constants in order of leftmost, rightmost
def sort_left_to_right(middle_y, c1, c2, m1, m2):

    x1 = int((middle_y - c1)/m1)
    x2 = int((middle_y - c2)/m2)

    if x1 > x2:
        m1, m2, c1, c2 = m2, m1, c2, c1 
    return c1, c2, m1, m2 

def draw_loop(c1, c2, m1, m2, lower, upper, image, colour):
    y_lower = int(lower[1])
    y_upper = int(upper[1]) + 1
    if y_lower < 0:
        y_lower = 0
    if y_upper > image.shape[0]:
        y_upper = image.shape[0] - 1

    for y in range(y_lower, y_upper):
        x1 = np.round((y - c1)/m1)
        x2 = np.round((y - c2)/m2)

        #clip 
        clip_and_draw(x1, x2, y, image.shape[1], image, colour)
    return


#clips x's if needed, or excludes them from being drawn at all if out of bounds
def clip_and_draw(x1, x2, y, width, image, colour):
    if x1 < 0 and x2 >= 0:
        x1 = 0
    elif x1 < 0:
        return
    if x2 > width - 1 and x1 <= width - 1:
        x2 = width - 1
    elif x2 > width - 1:
        return

    x = np.arange(int(x1), int(x2) + 1)
    image[y, x] = colour


def show_image(image):
    plt.imshow(image)
    plt.show()