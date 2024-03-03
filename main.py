import TriLib
import numpy as np
import time
import cv2



#todo bugfixes
#check that diff score goes down after adding average
#make sure pull rect doesnt go outside bounds, especially for average, sometimes returns black, red, cyan, magenta etc


def get_score(current_image, polygon, avg_colour):
    #using TriLib.colour_differences(vertices, image1, image2)
    #get difference before adding colour change, and then after, with picture being used in both. then find the improvement between these

    diff_before = TriLib.colour_differences(polygon, picture_np, current_image)

    temp_image = np.copy(current_image)
    TriLib.fill(polygon, temp_image, avg_colour)

    diff_after = TriLib.colour_differences(polygon, picture_np, temp_image)

    difference_score = (diff_before - diff_after)#.astype(np.int32)

    return difference_score




#load and bgr to rgb into numpy array
picture = cv2.imread("C:/Users/Daithi/Documents/Coding/trilib/tsunami.jpg")
picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
picture_np = np.asarray(picture)

image = np.full((picture_np.shape[0], picture_np.shape[1], 3), 0, dtype=np.uint16)

colour = np.array([100,100,100])

start_time = time.time()

vertices = np.array([(picture_np.shape[1] - 1, picture_np.shape[0] - 1), (picture_np.shape[1] - 1, 300), (400, picture_np.shape[0] - 1)])   


#first score with random colour
print(get_score(image, vertices, colour))
avg_colour = TriLib.average_colour(vertices, picture) 
#then with average
print(get_score(image, vertices, avg_colour))
#then fill poly on image to see how this affects
image = TriLib.fill(vertices, image, avg_colour)
print(get_score(image, vertices, avg_colour))

print(avg_colour)

""" #colour diff with random colour
print("before", TriLib.colour_differences(vertices, picture_np, image))

image = TriLib.fill(vertices, image, avg_colour)

#colour diff with avg colour
print("after", TriLib.colour_differences(vertices, picture_np, image))  """

topleft_coordinate, pulled_rect = TriLib.pull_rect(vertices, picture_np)
TriLib.put_rect(topleft_coordinate, pulled_rect, image)

image = TriLib.fill(vertices, image, avg_colour)


""" vertices = np.array([(100,434), (775,0), (775, 434)])   
image = TriLib.fill(vertices, image, colour)
topleft_coordinate, pulled_rect = TriLib.pull_rect(vertices, picture_np)
TriLib.put_rect(topleft_coordinate, pulled_rect, image)  """


end_time = time.time()
epoch_time = int((end_time - start_time) * 1000)
print("Time taken this epoch = ", epoch_time, " ms")
print("") 

TriLib.show_image(image)




