import TriLib
import pygame
import numpy as np
import time
import cv2
import sys


#load and bgr to rgb into numpy array
picture = cv2.imread("C:/Users/Daithi/Documents/Coding/trilib/tsunami.jpg")
picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
picture_np = np.asarray(picture)

width = picture_np.shape[1]
height = picture_np.shape[0]


initial_vertices = np.array([
        [(0, 0), (100, 0), (0, 100)], #right angle  
        [(0, 0), (100, 0), (50, 86)], #equilateral
        [(0, 0), (100, 0), (50, 50)], #isoceles
        [(0, 0), (50, 0), (25, 100)], #long isoceles

        [(0, 0), (100, 0), (0, 60)], #scalene
        [(0, 0), (100, 0), (0, 30)], #long scalene
        ])


def main():
    vertices, centroids = recenter_triangles(initial_vertices)

    current_image = np.full((height, width, 3), 255, dtype=np.uint16)
    #uncomment to load from last save
    #current_image = np.load('current_image.npy') 

    DISPLAYSURF = pygame.display.set_mode((width, height))

    while True:
       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminate()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    terminate()

        #draw surface from last iteration
        image_surface = pygame.surfarray.make_surface(current_image)
        image_surface = pygame.transform.rotate(image_surface, -90)
        image_surface = pygame.transform.scale(image_surface, (width//2, height//2))
        image_surface = pygame.transform.flip(image_surface, True, False)
        DISPLAYSURF.blit(image_surface, (width // 2 - image_surface.get_width() // 2, height // 2 - image_surface.get_height() // 2))
        pygame.display.update()

        #get poly
        polygons = generate_polygons(vertices, 2,5,3)
        top_polygons = get_top_polygons(polygons, 3, current_image)

        #from N top_polys, evolve, needs variance parameter, number of evolutions, and number selected per evo
        final_evolution = evolution(top_polygons, 0, 1, 1, 5, 2, 3, current_image)
        print(final_evolution)

        #update image
        TriLib.fill(final_evolution, current_image, TriLib.average_colour(final_evolution, picture))

        np.save('current_image.npy', current_image)


#starting with a list of default tri shapes, generate random tris
def generate_polygons(vertices, num_rotations, num_sizes, num_positions):
    num_tris = len(vertices)
    num_types = num_tris * num_rotations * num_sizes * num_positions

    polygons = np.zeros((num_types, 3, 2))
    h = 0
        
    rotated_verts = random_rotate(vertices, num_rotations, 100)
    
    scaled_verts = random_scale(rotated_verts, num_sizes, num_rotations, num_tris)

    polygons = random_pos(scaled_verts, num_positions, num_sizes, num_rotations, num_tris)
    #print("Generating Polygons", h, "/", num_types)
    h += num_positions * num_sizes * num_rotations

    return polygons


def recenter_triangles(vertices):
    #recenters polygons to 0,0
    centroids = np.zeros_like(vertices)
    for i in range(len(vertices)):
        mean_x = np.mean(vertices[i][:, 0])
        mean_y = np.mean(vertices[i][:, 1])
        centroids[i][:, 0] = mean_x
        centroids[i][:, 1] = mean_y
    centered_vertices = vertices - centroids

    return centered_vertices, centroids

def random_rotate(polys, num_rotations, percent):
    variance = percent/100
    theta = np.random.uniform((-np.pi * variance), (np.pi * variance), size = num_rotations)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrices = np.stack((cos_theta, -sin_theta, sin_theta, cos_theta), axis = 1).reshape(num_rotations, 2, 2)

    rotated_arrays = np.zeros((len(polys) * num_rotations, 3, 2))
    for i in range(num_rotations):
        for f in range(len(polys)):
            rotated_arrays[i + f * num_rotations] = np.dot(polys[f], rotation_matrices[i])  

    return rotated_arrays


def random_scale(vertices, num_sizes, num_rotations, num_tris):
    N = num_sizes * num_rotations * num_tris

    scales = np.exp(np.random.uniform(-0.5, 2, size=N))
    scales = scales[:, np.newaxis, np.newaxis]

    scaled = np.repeat(vertices, num_sizes, axis=0)
    scaled = scaled * scales

    return scaled  

#simply scaled by percent variance
def scale(vertices, num_sizes, num_rotations, scale_variance_percentage):
    N = num_sizes * num_rotations 

    scale_variance = scale_variance_percentage/100
    scales = np.random.uniform((1 - scale_variance), (1 + scale_variance), size=N)
    scales = scales[:, np.newaxis, np.newaxis]

    scaled = np.repeat(vertices, num_sizes, axis=0)
    scaled = scaled * scales

    return scaled  



#takes in polys (vertices), and makes adds new centroid pos to each
def random_pos(vertices, num_random_positions, num_sizes, num_rotations, num_tris):
    N = num_random_positions * num_sizes * num_rotations * num_tris

    x = np.random.randint(0, width, size = N)
    y = np.random.randint(0, height, size = N)

    centroids = np.zeros(shape=(N, 3, 2))
    centroids[:, :, 0] = x[:, np.newaxis]
    centroids[:, :, 1] = y[:, np.newaxis]

    repos_vertices = np.repeat(vertices, num_random_positions, axis=0)
    repos_vertices = repos_vertices + centroids

    return repos_vertices.astype(int)


#from list of polygons, returns those with best scores
def get_top_polygons(polygons, num_wanted_polys, current_image):

    #initialises scores and array for top polys
    score = np.zeros(num_wanted_polys)
    top_polygons = np.zeros((num_wanted_polys, 3, 2))

    for h in range(len(polygons)):
        avg_colour = TriLib.average_colour(polygons[h], picture_np)

        #add current polygon to top list if its difference is bigger than biggest diff
        this_score = get_score(current_image, polygons[h], avg_colour)
        if this_score > np.max(score):
            print(this_score)

            index = np.argmin(score) 
            score[index] = this_score
            top_polygons[index] = polygons[h]

            #print("this_score", this_score) 
            
        #print(h)

    return top_polygons


#big difference score indicates a large improvement towards either small pixel accuracy, or wide pixel improvements which targets large polys first, then small polys
def get_score(current_image, polygon, avg_colour):
    #using TriLib.colour_differences(vertices, image1, image2)
    #get difference before adding colour change, and then after, with picture being used in both. then find the improvement between these

    diff_before = TriLib.square_colour_differences(polygon, picture_np, current_image)

    temp_image = np.copy(current_image)
    TriLib.fill(polygon, temp_image, avg_colour)

    diff_after = TriLib.square_colour_differences(polygon, picture_np, temp_image)

    difference_score = (diff_before - diff_after).astype(np.int32)

    return difference_score


def evolution(top_polys, num_random_positions, num_sizes, num_rotations, variance_percent, evo_count, polys_per_evo, current_image):

    for _ in range(evo_count):
        evolved_polys = evolve(top_polys, num_random_positions, num_sizes, num_rotations, variance_percent)

        top_polys = get_top_polygons(evolved_polys, polys_per_evo, current_image)

    final_evolution = get_top_polygons(evolved_polys, 1, current_image)[0]

    return final_evolution.astype(int)



def evolve(top_polys, num_random_positions, num_sizes, num_rotations, variance_percent):
    #recreate polys
    #needs size, rotation, and scale of og
    N = len(top_polys) * (num_rotations + 1) * (num_sizes + 1)

    evolved_polys = np.zeros((N, 3, 2))
    #keeps original 
    evolved_polys[:len(top_polys)] = len(top_polys)

    centered_polys, centroids = recenter_triangles(top_polys)
    #also return centroid for pos


    #rotate
    #add additional adjusted polys to og top_polys, each for percentage of rotation values, N times
    evolved_polys[len(top_polys):len(top_polys) + len(top_polys)*num_rotations] = random_rotate(centered_polys, num_rotations, variance_percent)
    
    #scale
    #keeps all of last rotated polys, create additonal scaled ones
    evolved_polys[len(top_polys) + len(top_polys)*num_rotations:N] = scale(evolved_polys[:len(top_polys) + len(top_polys)*num_rotations], num_sizes, num_rotations, variance_percent)

    #temp, add centroids back
    for i in range((num_rotations + 1) * (num_sizes + 1)):
        for f in range(len(top_polys)):
            evolved_polys[i * 3 + f] += centroids[f] 
        
    #pos
    #to centroid, add percent of max width and height, keeping originals as well
    return evolved_polys



def terminate():
    pygame.quit() 
    sys.exit()

main()
pygame.quit() 