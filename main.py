from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, inf


####  TODO  ####
# combine points
# non connected islands
# line weights in are_points_connected are looser as the line is getting longer



####  ALL THE WEIGHTS ####
line_correctness = 0.5 #how much of a line must lay on actual pixels
line_range = 3 #how many pixels off the line can an actual pixel be for us to count it
points_for_inclomplete = 0.8 #how much points would we give to a line for a pixel that is not on the line but is near it
delta_weight = 3 #line accuracy (more - less lines)
delta_dealbreaker = 8 #delta that makes me not use this point
min_pixels_in_line = 5 # randomass weight, leave it as it is, ur prob better off not changing it

def make_positive(n):
    if n < 0:
        return -n
    return n

def distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def are_points_connected(p1, p2, matrix, weight=line_correctness, range_weight=line_range, points_for_inclomplete_weight=points_for_inclomplete):
    print(p1, p2)
    if p2[0] > p1[0]:
        tempo = (p1[0], p1[1])
        p1 = (p2[0], p2[1])
        p2 = (tempo[0], tempo[1])
    print(p1, p2)
    
    k = (p1[1]-p2[1]) / (p1[0]-p2[0])
    if (p1[0]-p2[0]) == 0 or k == 0:
        k2 = inf
    else:
        k2 = -(1/k) #k of the perpendicular function
    
    total_pixels = 0
    total_true_pixels = 0
    for x in range(0, p1[0]-p2[0]):
        print(x + p2[0], round(k*x + p2[1]), matrix[round(k*x + p2[1])][x + p2[0]])
        total_pixels += 1
        if matrix[round(k*x + p2[1])][x + p2[0]]:
            total_true_pixels += 1
        else:
            if k2 == inf:
                for y2 in range(-range_weight, range_weight+1):
                    if round(k*x + p2[1]) + y2 < len(matrix) and x + p2[0] < len(matrix[0]) and round(k*x + p2[1]) + y2 >= 0 and x + p2[0] < len(matrix[0]) >= 0:
                        if matrix[round(k*x + p2[1]) + y2][x + p2[0]] == 1:
                            total_true_pixels += points_for_inclomplete_weight
                            break
            else:    
                for x2 in range(-range_weight, range_weight+1):
                    if round(k*x + p2[1] + k2*x2) < len(matrix) and x + x2 + p2[0] < len(matrix[0]) and round(k*x + p2[1] + k2*x2) >= 0 and x + x2 + p2[0] >= 0:
                        print(x, x2, k, k2, p2[1], p2[0], len(matrix), len(matrix[0]))
                        if matrix[round(k*x + p2[1] + k2*x2)][x + x2 + p2[0]] == 1:
                            total_true_pixels += points_for_inclomplete_weight
                            break
    if total_pixels == 0:
        return True
    if total_true_pixels / total_pixels >= weight:
        return True
    return False
        

image_path = "images/img10.png"  
original_image = Image.open(image_path)

grayscale_image = original_image.convert('L')

grayscale_array = np.array(grayscale_image)

binary_matrix = (grayscale_array < 200).astype(int)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
####

indices = np.argwhere(binary_matrix)
ones = np.argwhere(binary_matrix == 1)

sum_x = 0
n_x = 0

sum_y = 0
n_y = 0

x_min = ones[0]
x_max = ones[0]

total_x_diff_squares = 0
total_diff_deltas = 0

start = (ones[0][1], ones[0][0])

sets = []
visited = {}
bordering_pixels = []
in_border = {}
row, col = ones[0][0], ones[0][1]
last_row, last_col = 0, 0
pixels_in_line = 0
endpoint_points = []
points_extremes = []
while True:
    pixels_in_line += 1
    visited[(row, col)] = True
    #print(col, row)
    if col != 0:
        if not (row, col-1) in visited and binary_matrix[row][col-1] == 1:
            bordering_pixels.append((row, col-1))
            visited[(row, col-1)] = True
    if col != len(binary_matrix[0])-1:
        if not (row, col+1) in visited and binary_matrix[row][col+1] == 1:
            bordering_pixels.append((row, col+1))
            visited[(row, col+1)] = True
    if row != 0:
        if not (row-1, col) in visited and binary_matrix[row-1][col] == 1:
            bordering_pixels.append((row-1, col))
            visited[(row-1, col)] = True
    if row != len(binary_matrix)-1:
        if not (row+1, col) in visited and binary_matrix[row+1][col] == 1:
            bordering_pixels.append((row+1,col))
            visited[(row+1,col)] = True
            
    
    
    
    sum_x += col
    n_x += 1
    sum_y += row
    n_y += 1
    
    average_x = sum_x/n_x
    average_y = sum_y/n_y
    
    total_x_diff_squares += (col - average_x)**2
    total_diff_deltas += (col - average_x) * (row - average_y)
    
    m = total_diff_deltas / total_x_diff_squares
    if total_x_diff_squares == 0:
        m = total_diff_deltas
    c = -m * average_x + average_y
    
    y_delta = (m*col + c) - row # the difference between expected and real y value
    #print(f"({col}, {row}), [{average_x}, {average_y}], {y_delta}")
    
    if make_positive(y_delta) > delta_weight:
        print("Making a new line")
        if make_positive(y_delta) > delta_dealbreaker:
            tempo = (row, col)
            row = last_row
            col = last_col
        
        if (last_row != start[1] or last_col != start[0]) and pixels_in_line > min_pixels_in_line:
            sets.append([m, c, x_min, x_max, start, (col, row)])
            endpoint_points.append(start)
            endpoint_points.append((col, row))
            points_extremes.append((x_min[0], x_min[1]))
            points_extremes.append((x_max[0], x_max[1]))
            
        if make_positive(y_delta) > delta_dealbreaker:
            row, col = tempo[0], tempo[1]
        start = (col, row)
        x_min = (col, row)
        x_max = (col, row)
        pixels_in_line = 0
        sum_x = 0
        n_x = 0
        sum_y = 0
        n_y = 0
        total_x_diff_squares = 0
        total_diff_deltas = 0
        
        
    if col > x_max[0]:
        x_max = (col, row)
    elif col < x_min[0]:
        x_min = (col, row)
        
    min_delta = 9999
    id = 0
    last_row = row
    last_col = col
    for pixel in range(len(bordering_pixels)):
        #print(make_positive((m*bordering_pixels[pixel][1] + c) - bordering_pixels[pixel][0]))
        if make_positive((m*bordering_pixels[pixel][1] + c) - bordering_pixels[pixel][0]) < min_delta:
            if distance((bordering_pixels[pixel][1], bordering_pixels[pixel][0]), (last_col, last_row)) > 8:
                continue
            
            col = bordering_pixels[pixel][1]
            row = bordering_pixels[pixel][0]
            id = pixel
    if len(bordering_pixels) == 0:
        break
    bordering_pixels.pop(id)
    
        
if make_positive(y_delta) > delta_dealbreaker:
    row = last_row
    col = last_col
print(pixels_in_line, 11)
if pixels_in_line > min_pixels_in_line:  
    print('making a line')
    sets.append([m, c, x_min, x_max, start, (col, row)])
actual_lines = []
for set in sets:
    actual_lines.append((set[2], set[3]))
    
#point logic
points = []
points_seen = {}
        
for point in points_extremes:
    if not point in points_seen:
        points_seen[point] = True
        points.append(point)

#line logic

for i in range(len(actual_lines)-1, -1, -1):
    if not are_points_connected(actual_lines[i][0], actual_lines[i][1], binary_matrix):
        actual_lines.pop(i)

for line in actual_lines:
    ax3.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'r-', linewidth=1, marker='o')
    pass

'''
for point in endpoint_points:
    ax3.plot(point[0], point[1], 'ro',)
    
for point in points_extremes:
    ax3.plot(point[0], point[1], 'go',)
'''
for point in points:
    ax3.plot(point[0], point[1], 'go',)
    


    
    




######
ax1.imshow(original_image)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(grayscale_image, cmap='gray')
ax2.set_title('Grayscale Image')
ax2.axis('off')

ax3.imshow(binary_matrix, cmap='binary')
ax3.set_title('Binary Matrix (Threshold: 100)')
ax3.axis('off')

plt.tight_layout()
plt.show()
