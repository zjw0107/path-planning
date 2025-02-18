# -*- coding: utf-8 -*-
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from osgeo import gdal
import time
import heapq
import random
import copy
import matplotlib.colors as mcolors
import itertools

raster_path = "F:/Doctor/data/areaDEM300.tif"
raster_path1 = "F:/Doctor/data/EsriareaZB300.tif"
raster_path2 = "F:/Doctor/data/soil300.tif"

dataset = gdal.Open(raster_path)

rows = dataset.RasterYSize
cols = dataset.RasterXSize
v = [[0] * cols for _ in range(rows)]
for i in range(rows):
    for j in range(cols):
        v[i][j] = dataset.ReadAsArray(j, i, 1, 1)[0, 0]

dataset1 = gdal.Open(raster_path1)
rows1 = dataset1.RasterYSize
cols1 = dataset1.RasterXSize
s = [[0] * cols for _ in range(rows)]
for i in range(rows1):
    for j in range(cols1):

        s[i][j] = dataset1.ReadAsArray(j, i, 1, 1)[0, 0]

dataset2 = gdal.Open(raster_path2)
rows2 = dataset2.RasterYSize
cols2 = dataset2.RasterXSize
l= [[0] * cols2 for _ in range(rows2)]
for i in range(rows2):
    for j in range(cols2):
        l[i][j] = dataset2.ReadAsArray(j, i, 1, 1)[0, 0]


def read_raster(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise Exception("Error opening the raster file")
    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray()
    return raster_array
dataset = None

def get_slope(row1, col1, row2, col2):
    height_diff = v[row2][col2] - v[row1][col1]
    distance = ((row2 - row1) ** 2 + (col2 - col1) ** 2) ** 0.5
    distance1 = distance*30
    slope = height_diff / distance1
    return slope


def move(row, col, action):
 
    movements = [(-1, -1), (-1,0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    next_row = row + movements[action][0]
    next_col = col + movements[action][1]
    if next_row < 0 or next_row >= rows:
        next_row = row
        next_col = col
    if next_col < 0 or next_col >= cols:
        next_row = row
        next_col = col
    next_row = max(0, min(next_row, rows - 1))
    next_col = max(0, min(next_col, cols - 1))
    return next_row, next_col

def cost_slope(current_pos,neighbor):
    row,col=current_pos[0],current_pos[1]
    next_row, next_col = neighbor[0],neighbor[1]
    slope = get_slope(row, col,next_row, next_col)
    if slope == 0:
        reward = 1
    elif -0.176 < slope <= 0:
        reward = 2
    elif -0.3639 < slope <= -0.176:
        reward = 4
    elif -0.5773 < slope <= -0.3639:
        reward = 6
    elif 0 < slope <= 0.0875:
        reward = 2
    elif 0.0875 < slope <= 0.176:
        reward = 3
    elif 0.176 < slope <= 0.2679:
        reward = 4
    elif 0.2679 < slope <= 0.3639:
        reward = 5
    elif 0.3639 < slope <= 0.4663:
        reward = 6
    else:
        reward = 100
    return reward
def cost_goal(current_pos, neighbor,goal):
    row, col = current_pos[0], current_pos[1]
    next_row, next_col = neighbor[0], neighbor[1]
    positive_distances = []
    distances = []
    goal_point1 = goal
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= row + i < rows and 0 <= col + j < cols:
                if s[row + i][col + j] == 1 or s[row + i][col + j] == 2:
                    distance = 100000
                else:
                    distance = ((row + i - goal_point1[0]) ** 2 + (col + j - goal_point1[1]) ** 2) ** 0.5
                positive_distances.append((distance, row + i, col + j))
    sorted_negative_slopes = sorted(positive_distances, key=lambda x: x[0])
    distances.extend(sorted_negative_slopes)
    found_match = False
    for i, (distance, r, c) in enumerate(distances):
        if r == next_row and c == next_col:
            found_match = True
            break
    if found_match:
        distance_rank = distances.index((distance, next_row, next_col))

        if distance_rank == 0:
            reward = 0

        elif distance_rank == 1:
            reward = 1

        elif distance_rank == 2:
            reward = 2

        elif distance_rank == 3:
            reward = 3

        elif distance_rank == 4:
            reward = 4

        elif distance_rank == 5:
            reward = 5

        elif distance_rank == 6:
            reward = 6

        elif distance_rank == 7:
            reward = 7
        else:
            reward = 100 
    else:
        reward = 100
    return  reward

def cost_surface(neighbor):
    row = neighbor[0]
    col = neighbor[1]
    if s[row][col] == 1:
        reward = 100
    elif s[row][col] == 2:
        reward = 100
    elif s[row][col] == 5:
        reward = 10
    elif s[row][col] == 7:
        reward = 10
    elif s[row][col] == 8:
        reward = 0
    elif s[row][col] == 11:
        reward = 0
    else:
        reward = 0
    return reward

def distance(current_start, waypoint):
    distance = ((current_start[0] - waypoint[0]) ** 2 + (current_start[1] - waypoint[1]) ** 2) ** 0.5 * 30
    return distance

def dijkstra(grid_size, start, goal, s):
   
    rows, cols = grid_size

    directions = [(-1, -1), (-1,0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
   
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
  
    distance = {start: (0, None)}
    while priority_queue:
 
        dist, current = heapq.heappop(priority_queue)

        if current == goal:
          
            path = [goal]
            while current != start:
               
                current = distance[current][1]
                path.append(current)
            return dist, path[::-1]
        for dx, dy in directions:
   
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
       
                cost_increment = cost_surface(neighbor)
                cost_slope1 = cost_slope(current,neighbor)
                cost_goal1 = cost_goal(current,neighbor,goal)
                new_dist = dist + cost_increment + cost_slope1 + cost_goal1
                if neighbor not in distance or new_dist < distance[neighbor][0]:
                
                    distance[neighbor] = (new_dist, current)
              
                    heapq.heappush(priority_queue, (new_dist, neighbor))
                 
    return float('inf'), []  

start_time = time.time()
file_path = "F:/Doctor/data/areaDEM300.tif"
raster_array = read_raster(file_path)

path = []
points = [(20,30), (50, 70), (90, 100),(140,130),(200,260)]
waypoints_1 = [(20,30), (50, 70), (90, 100),(140,130),(200,260)]
import math


start_point = points[0]  
end_point = points[-1] 
remaining_points = points[1:-1]

sorted_points = [start_point]  
current_point = start_point 
path = []

while remaining_points:
    
    closest_point = min(remaining_points,key=lambda p: math.sqrt((p[0] - current_point[0]) ** 2 + (p[1] - current_point[1]) ** 2),)
    
    sorted_points.append(closest_point)
   
    current_point = closest_point
   
    remaining_points.remove(closest_point)

sorted_points.append(end_point)
start_time = time.time()
start_time1 = time.time()
for i in range(len(sorted_points)-1):
    cost1,subpath = dijkstra(raster_array.shape,sorted_points[i], sorted_points[i+1], raster_array)
    cost1 += cost1
    path.extend(subpath)
end_time1 = time.time()
total_runtime1 = end_time1 - start_time1
print(f"O TOTAL TIME：{total_runtime1} 秒")
print(f"O TOTAL COST：{cost1}")

path2 = [point for point in path if point not in waypoints_1]
for point in path2:
    row, col = point
    s[row][col] = 1

path1 = [] 
cost2 = 0
start_time2 = time.time()
for i in range(len(sorted_points)-1):
    cost2,subpath1 = dijkstra(raster_array.shape,sorted_points[i], sorted_points[i+1], raster_array)
    cost2 += cost2
    path1.extend(subpath1)
end_time2 = time.time()
total_runtime2 = end_time2 - start_time2
print(f"S TOTAL TIME：{total_runtime2} ")
print(f"S TOTAL COST：{cost2}")
end_time = time.time()
total_runtime = end_time - start_time
print(f"TOTAL TIME：{total_runtime} ")

def show_path_2d1(start, end):
    raster_path = "F:/Doctor/data/areaDEM300.tif"
    raster_array = read_raster(raster_path)
    plt.figure()
    plt.subplot()
    plt.imshow(raster_array, cmap='viridis', origin='upper')
    plt.colorbar()
    for wp in waypoints_1:
        plt.scatter(wp[1], wp[0], color='black', marker='o', s=60)  
    for i in range(len(path) - 1):
        current_height = raster_array[path[i][0], path[i][1]]
        next_height = raster_array[path[i + 1][0], path[i + 1][1]]

        if current_height < next_height:
            color = 'red'  
        elif current_height > next_height:
            color = 'yellow'  
        else:
            color = 'blue'  

        plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], color=color, linewidth=2)
    for i in range(len(path1) - 1):

        current_height = raster_array[path1[i][0], path1[i][1]]
        next_height = raster_array[path1[i + 1][0], path1[i + 1][1]]
 
        if current_height < next_height:
            color = 'red'  
        elif current_height > next_height:
            color = 'yellow'  
        else:
            color = 'blue'  
     
        plt.plot([path1[i][1], path1[i + 1][1]], [path1[i][0],path1[i + 1][0]], color=color, linewidth=1)
    plt.scatter(start[1], start[0], color='red', marker='s', s=100)  
    plt.scatter(end[1], end[0], color='blue', marker='o', s=100)  
    plt.axis('equal')  

    raster_path1 = "F:/Doctor/data/EsriareaZB300.tif"
    raster_array1 = read_raster(raster_path1)
   
    cmap = mcolors.ListedColormap(['black','blue','aqua' ,'green', 'pink','gray','azure'])
    bounds = [0,1,2, 5, 7,8,11,255]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure()
    
    plt.subplot()
    plt.imshow(raster_array1, cmap=cmap, norm=norm, origin='upper')
    for wp in waypoints_1:
        plt.scatter(wp[1], wp[0], color='black', marker='o', s=60) 
    for i in range(len(path) - 1):

        current_height = raster_array[path[i][0], path[i][1]]
        next_height = raster_array[path[i + 1][0], path[i + 1][1]]
    
        if current_height < next_height:
            color = 'red'  
        elif current_height > next_height:
            color = 'yellow' 
        else:
            color = 'blue'  
     
        plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], color=color, linewidth=2)
    for i in range(len(path1) - 1):
     
        current_height = raster_array[path1[i][0], path1[i][1]]
        next_height = raster_array[path1[i + 1][0], path1[i + 1][1]]
   
        if current_height < next_height:
            color = 'red' 
        elif current_height > next_height:
            color = 'yellow' 
        else:
            color = 'blue'
     
        plt.plot([path1[i][1], path1[i + 1][1]], [path1[i][0],path1[i + 1][0]], color=color, linewidth=1)
    plt.imshow(raster_array1, cmap=cmap, norm=norm, origin='upper')
    plt.scatter(start[1], start[0], color='red', marker='s', s=100)  
    plt.scatter(end[1], end[0], color='blue', marker='o', s=100)  
    plt.axis('equal')
    plt.show()
show_path_2d1(start_point, end_point)

