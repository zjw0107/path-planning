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

#获取每个栅格的高度值
dataset = gdal.Open(raster_path)# 打开栅格影像
# 获取栅格影像的行数和列数
rows = dataset.RasterYSize
cols = dataset.RasterXSize
v = [[0] * cols for _ in range(rows)]#用来创建一个二维列表（2D list）并初始化所有元素为0
# 逐行逐列读取栅格影像的像素值
for i in range(rows):
    for j in range(cols):
        # 读取当前像元值
        v[i][j] = dataset.ReadAsArray(j, i, 1, 1)[0, 0]#获取每个栅格的高程

#获取每个栅格的地类类型
dataset1 = gdal.Open(raster_path1)
# 获取栅格影像的行数和列数
rows1 = dataset1.RasterYSize
cols1 = dataset1.RasterXSize
s = [[0] * cols for _ in range(rows)]#用来创建一个二维列表（2D list）并初始化所有元素为0
# 逐行逐列读取栅格影像的像素值
for i in range(rows1):
    for j in range(cols1):
        # 读取当前像元值
        s[i][j] = dataset1.ReadAsArray(j, i, 1, 1)[0, 0]#获取每个栅格的地类

#获取每个栅格的土壤类型
dataset2 = gdal.Open(raster_path2)
# 获取栅格影像的行数和列数
rows2 = dataset2.RasterYSize
cols2 = dataset2.RasterXSize
l= [[0] * cols2 for _ in range(rows2)]#用来创建一个二维列表（2D list）并初始化所有元素为0
# 逐行逐列读取栅格影像的像素值
for i in range(rows2):
    for j in range(cols2):
        # 读取当前像元值
        l[i][j] = dataset2.ReadAsArray(j, i, 1, 1)[0, 0]#获取每个栅格的地类


def read_raster(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise Exception("Error opening the raster file")
    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray()
    #这行代码调用ReadAsArray方法，该方法将栅格波段中的像素值读取为NumPy数组。
    # 这个数组raster_array包含了图像中每个像素的值，形状与影像的大小相匹配。
    return raster_array# 逐行逐列读取像元值
dataset = None# 关闭栅格影像数据集

#计算坡度角
def get_slope(row1, col1, row2, col2):
    # 计算两个栅格的高度差
    height_diff = v[row2][col2] - v[row1][col1]
    # 计算两个栅格中心点的距离
    distance = ((row2 - row1) ** 2 + (col2 - col1) ** 2) ** 0.5
    distance1 = distance*30
    # 计算坡度值
    slope = height_diff / distance1
    return slope

#设置无人车移动方向
def move(row, col, action):
    # 定义八个方向的移动
    movements = [(-1, -1), (-1,0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # 根据动作确定下一步的移动方向
    next_row = row + movements[action][0]
    next_col = col + movements[action][1]
    if next_row < 0 or next_row >= rows:
        next_row = row
        next_col = col
    if next_col < 0 or next_col >= cols:
        next_row = row
        next_col = col
    # 确保下一步不超出边界
    next_row = max(0, min(next_row, rows - 1))
    next_col = max(0, min(next_col, cols - 1))
    return next_row, next_col

#坡度通行代价计算
def cost_slope(current_pos,neighbor):
    row,col=current_pos[0],current_pos[1]
    next_row, next_col = neighbor[0],neighbor[1]
    slope = get_slope(row, col,next_row, next_col)
    # 根据坡度排序和坡度值设置奖励值
    if slope == 0:
        reward = 1
    # -10-0
    elif -0.176 < slope <= 0:
        reward = 2
    # -20--10
    elif -0.3639 < slope <= -0.176:
        reward = 4
    #-20--30
    elif -0.5773 < slope <= -0.3639:
        reward = 6
    #0-5
    elif 0 < slope <= 0.0875:
        reward = 2
    #5-10
    elif 0.0875 < slope <= 0.176:
        reward = 3
    # 10-15
    elif 0.176 < slope <= 0.2679:
        reward = 4
    # 15-20
    elif 0.2679 < slope <= 0.3639:
        reward = 5
    # 20-25
    elif 0.3639 < slope <= 0.4663:
        reward = 6
    else:
        reward = 100
        #第六
    return reward
#目标通行代价计算
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
            # 对距离排序按照
    sorted_negative_slopes = sorted(positive_distances, key=lambda x: x[0])
    distances.extend(sorted_negative_slopes)
    # 找到next_row和next_col对应栅格的索引位置
    found_match = False
    for i, (distance, r, c) in enumerate(distances):
        if r == next_row and c == next_col:
            found_match = True
            break
    # 如果找到匹配的相邻栅格，根据排序位置设置奖励值
    if found_match:
        distance_rank = distances.index((distance, next_row, next_col))
        # 根据距离排序和距离值设置奖励值
        #第一
        if distance_rank == 0:
            reward = 0
        # 第二
        elif distance_rank == 1:
            reward = 1
        # 第三
        elif distance_rank == 2:
            reward = 2
        # 第四
        elif distance_rank == 3:
            reward = 3
        # 第五
        elif distance_rank == 4:
            reward = 4
        # 第六
        elif distance_rank == 5:
            reward = 5
        # 第七
        elif distance_rank == 6:
            reward = 6
        # 第8
        elif distance_rank == 7:
            reward = 7
        else:
            reward = 100 # 如果不符合上述条件，则奖励值为0
    else:
        reward = 100 # 如果未找到匹配的相邻栅格，奖励值设为-100
    return  reward
#地表通行代价计算
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
#气象影响
def distance(current_start, waypoint):
    distance = ((current_start[0] - waypoint[0]) ** 2 + (current_start[1] - waypoint[1]) ** 2) ** 0.5 * 30
    return distance
#使用dijkstra算法寻找路径
def dijkstra(grid_size, start, goal, s):
    # Dijkstra 算法函数，用于寻找栅格地图中从起点到目标点的最短路径
    rows, cols = grid_size
    # 获取栅格地图的行数和列数
    directions = [(-1, -1), (-1,0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    #priority_queue` 已经可以作为一个优先队列使用，并且可以使用 `heapq.heappop(priority_queue)` 高效地弹出具有最小距离的节点,[(1, (27, 19))]
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
    # 初始化一个字典distance，字典用于记录从起始节点到其他节点的最短距离信息以及对应的父节点,{(28, 19): (0, None), (27, 19): (1, (28, 19))}
    distance = {start: (0, None)}
    while priority_queue:
        # 当优先队列不为空时，执行以下循环
        dist, current = heapq.heappop(priority_queue)
        # 从优先队列中弹出距离最小的节点
        if current == goal:
            # 如果当前节点是目标节点，返回最短路径
            path = [goal]
            while current != start:
                # 从目标节点回溯到起点，构建最短路径
                current = distance[current][1]
                path.append(current)
            return dist, path[::-1]
        for dx, dy in directions:
            # 遍历当前节点的相邻节点
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # 如果邻居在图内
                cost_increment = cost_surface(neighbor)
                cost_slope1 = cost_slope(current,neighbor)
                cost_goal1 = cost_goal(current,neighbor,goal)
                new_dist = dist + cost_increment + cost_slope1 + cost_goal1
                if neighbor not in distance or new_dist < distance[neighbor][0]:
                # neighbor not in distance: 检查当前邻居节点是否已经在 distance 字典中记录。如果邻居节点不在字典中，表示它是第一次被探索，
                # 因此需要更新它的距离信息和父节点。neighbor not in distance:
                #检查从当前节点经过邻居节点到达起始节点的新路径长度 new_dist 是否比已知的最短路径长度如果新的路径更短，则更新距离信息和父节点
                    distance[neighbor] = (new_dist, current)
                #从起始节点到达邻居节点的新的最短距离以及该路径上邻居节点的父节点
                    heapq.heappush(priority_queue, (new_dist, neighbor))
                  #是将一个新的节点（邻居节点）及其与起始节点的距离推入优先队列（最小堆）的操作
    # 如果遍历完所有节点仍然没有找到路径，则返回空列表
    return float('inf'), []   # 返回无穷大代表未找到路径

start_time = time.time()
file_path = "F:/Doctor/data/areaDEM300.tif"
raster_array = read_raster(file_path)
"""
#三个路径点
path= []
points = [(20,150),(120, 80),(200,30)]
waypoints_1 = [(20,150),(120, 80),(200,30)]

#四个路径点
path= []
points = [(120,150),(120, 30), (210, 20),(270,30)]
waypoints_1= [(120,150),(120, 30), (210, 20),(270,30)]
"""
#五个路径点
path = []
points = [(20,30), (50, 70), (90, 100),(140,130),(200,260)]
waypoints_1 = [(20,30), (50, 70), (90, 100),(140,130),(200,260)]
import math

# 设置起点和终点
start_point = points[0]  # 起点
end_point = points[-1]  # 终点 [(120,150), (120, 30), (210, 20),(280,40)]
# 移除起点和终点
remaining_points = points[1:-1]
# 初始化路径

sorted_points = [start_point]  # 最终排序的点列表
current_point = start_point  # 从起点开始
path = []
# 按距离找到最近的点
while remaining_points:
    # 找到距离当前点最近的点
    closest_point = min(remaining_points,key=lambda p: math.sqrt((p[0] - current_point[0]) ** 2 + (p[1] - current_point[1]) ** 2),)
    # 将最近的点加入排序数组
    sorted_points.append(closest_point)
    # 更新当前点为最近的点
    current_point = closest_point
    # 从剩余点列表中移除已找到的点
    remaining_points.remove(closest_point)
# 将终点加入排序数组
sorted_points.append(end_point)
start_time = time.time()
start_time1 = time.time()
for i in range(len(sorted_points)-1):
    cost1,subpath = dijkstra(raster_array.shape,sorted_points[i], sorted_points[i+1], raster_array)
    cost1 += cost1
    path.extend(subpath)
end_time1 = time.time()
total_runtime1 = end_time1 - start_time1
print(f"最优路径总运行时间：{total_runtime1} 秒")
print(f"最优路线总代价：{cost1}")
#次优路径计算
path2 = [point for point in path if point not in waypoints_1]
for point in path2:
    row, col = point
    s[row][col] = 1
# 重新规划路径
path1 = []  # 重置路径
cost2 = 0
# 遍历每个要点
start_time2 = time.time()
for i in range(len(sorted_points)-1):
    cost2,subpath1 = dijkstra(raster_array.shape,sorted_points[i], sorted_points[i+1], raster_array)
    cost2 += cost2
    path1.extend(subpath1)
end_time2 = time.time()
total_runtime2 = end_time2 - start_time2
print(f"次优路径总运行时间：{total_runtime2} 秒")
print(f"次优路线总代价：{cost2}")
# 计算总运行时间
end_time = time.time()
total_runtime = end_time - start_time
print(f"总运行时间：{total_runtime} 秒")

#DEM图和路径，植被和路径，含有最优和次优路线
def show_path_2d1(start, end):
    raster_path = "F:/Doctor/data/areaDEM300.tif"
    raster_array = read_raster(raster_path)
    plt.figure()
    # 显示栅格影像
    plt.subplot()
    plt.imshow(raster_array, cmap='viridis', origin='upper')
    plt.colorbar()
    for wp in waypoints_1:
        plt.scatter(wp[1], wp[0], color='black', marker='o', s=60)  # 绘制waypoints，颜色设置为红色，标记为"x"
    for i in range(len(path) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path[i][0], path[i][1]]
        next_height = raster_array[path[i + 1][0], path[i + 1][1]]
        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'blue'  # 黄色代表下坡
        # 绘制路径段，并设置颜色
        plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], color=color, linewidth=2)
    for i in range(len(path1) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path1[i][0], path1[i][1]]
        next_height = raster_array[path1[i + 1][0], path1[i + 1][1]]
        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'blue'  # 紫色代表平坦
        # 绘制路径段，并设置颜色
        plt.plot([path1[i][1], path1[i + 1][1]], [path1[i][0],path1[i + 1][0]], color=color, linewidth=1)
    plt.scatter(start[1], start[0], color='red', marker='s', s=100)  # 绘制起点坐标，颜色设置为红色
    plt.scatter(end[1], end[0], color='blue', marker='o', s=100)  # 绘制终点坐标，颜色设置为蓝色
    plt.axis('equal')  # 设置坐标轴比例为相等，保证显示的图形不会被拉伸
    plt.show()

    raster_path1 = "F:/Doctor/data/EsriareaZB300.tif"
    raster_array1 = read_raster(raster_path1)
    # 定义颜色映射，将值为5的像素映射为绿色，值为7的像素映射为灰色
    cmap = mcolors.ListedColormap(['black','blue','aqua' ,'green', 'pink','gray','azure'])
    bounds = [0,1,2, 5, 7,8,11,255]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure()
    # 显示栅格影像
    plt.subplot()
    plt.imshow(raster_array1, cmap=cmap, norm=norm, origin='upper')
    for wp in waypoints_1:
        plt.scatter(wp[1], wp[0], color='black', marker='o', s=60)   # 绘制waypoints，颜色设置为红色，标记为"x"
    for i in range(len(path) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path[i][0], path[i][1]]
        next_height = raster_array[path[i + 1][0], path[i + 1][1]]
        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'blue'  # 紫色代表平坦
        # 绘制路径段，并设置颜色
        plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], color=color, linewidth=2)
    for i in range(len(path1) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path1[i][0], path1[i][1]]
        next_height = raster_array[path1[i + 1][0], path1[i + 1][1]]
        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'blue'  # 紫色代表平坦
        # 绘制路径段，并设置颜色
        plt.plot([path1[i][1], path1[i + 1][1]], [path1[i][0],path1[i + 1][0]], color=color, linewidth=1)
    plt.imshow(raster_array1, cmap=cmap, norm=norm, origin='upper')
    plt.scatter(start[1], start[0], color='red', marker='s', s=100)  # 绘制起点坐标，颜色设置为红色
    plt.scatter(end[1], end[0], color='blue', marker='o', s=100)  # 绘制终点坐标，颜色设置为蓝色
    plt.axis('equal')
    plt.show()
show_path_2d1(start_point, end_point)

#带高度的3D含有最优路径和次优路径，输出最优路径的指标
def show_path_3d(start, end):
    import math
    # 读取栅格影像数据
    raster_path = "F:/Doctor/data/areaDEM300.tif"
    raster_array = read_raster(raster_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制DEM图
    X = np.arange(0, 300)
    Y = np.arange(0, 300)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, raster_array, cmap='viridis', alpha=0.7)
    total_slope = 0
    slopes = []  # 保存所有的坡度值
    turn_count = 0
    total_distance = 0
    uphill_distance = 0
    downhill_distance = 0
    hill_distance = 0
    uphill_slope = 0
    downhill_slope = 0
    for i in range(len(path)-1):  # 遍历path中所有点（除了最后一个点）
        if len(path)>1:
            # 计算每两个点之间的水平距离和垂直距离
            horizontal_distance = math.sqrt((path[i+1][1] - path[i][1]) ** 2 + (path[i+1][0] - path[i][0]) ** 2) * 30
            vertical_difference = abs(v[path[i+1][0]][path[i+1][1]] - v[path[i][0]][path[i][1]])
            distance = (horizontal_distance ** 2 + vertical_difference ** 2) ** 0.5
            total_distance += distance
            if horizontal_distance == 0:
                slope_angle = 0  # 设置为90度或其他默认值
            else:
                slope_angle = math.degrees(math.atan(vertical_difference / horizontal_distance))
            slope_angle = round(slope_angle, 2)
            slopes.append(slope_angle)
            if slope_angle < 0:
                slope_angle = abs(slope_angle)  # 将负斜率转换为绝对值
            total_slope += slope_angle
             # 将坡度值添加到列表中
            height_difference = v[path[i+1][0]][path[i+1][1]] - v[path[i][0]][path[i][1]]
            if height_difference > 0:
                uphill_distance += distance
                uphill_slope += slope_angle
            elif height_difference < 0:
                downhill_distance += distance
                downhill_slope += slope_angle
            elif height_difference == 0:
                hill_distance += distance
        # 检查是否有拐弯
    for i in range(1, len(path) - 1):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        x3, y3 = path[i + 1]
        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if cross_product != 0:
            turn_count += 1
    if len(path) > 1:
        average_slope = total_slope / (len(path) - 1)
        average_slope = round(average_slope, 2)
    else:
        average_slope = 0
    max_slope = max(slopes)  # 找到所有坡度值中的最大值
    min_slope = min(slopes)
    total_slope = sum(slopes)  # 计算所有坡度值的总和
    # 绘制路径

    for i in range(len(path) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path[i][0], path[i][1]]
        next_height = raster_array[path[i + 1][0], path[i + 1][1]]

        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'black'  # 黄色代表下坡
        # 绘制路径段，并设置颜色
        ax.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], [current_height, next_height], color=color,
            linewidth=2)
    for i in range(len(path1) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path1[i][0], path1[i][1]]
        next_height = raster_array[path1[i + 1][0], path1[i + 1][1]]
        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'black'  # 黄色代表下坡
        # 绘制路径段，并设置颜色
        ax.plot([path1[i][1], path1[i + 1][1]], [path1[i][0], path1[i + 1][0]], [current_height, next_height], color=color,
            linewidth=2)
    # 绘制起点和终点
    ax.scatter(start[1], start[0], raster_array[start[0], start[1]] + 1, color='red', marker='o', label='起点')
    ax.scatter(end[1], end[0], raster_array[end[0], end[1]] + 1, color='blue', marker='s', label='终点')
    # 调整坐标轴范围和标签
    ax.set_xlim(0, 301)
    ax.set_ylim(0, 301)
    ax.set_zlim(np.min(raster_array), np.max(raster_array) + 1)
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    ax.set_zlabel('高度')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    print("第一条路径长度:", total_distance)
    print("第一条路径上坡总距离:", uphill_distance)
    print("第一条路径下坡总距离:", downhill_distance)
    print("第一条路径平坦路长离:", hill_distance)
    print("第一条路径总坡度:", total_slope)
    print("第一条路径上坡总坡度:", uphill_slope)
    print("第一条路径下坡总坡度:", downhill_slope)
    print("第一条路径平均坡度:", average_slope)
    print("第一条路径最大坡度:", max_slope)
    print("第一条路径最小坡度:", min_slope)
    print("第一条路径拐弯个数:", turn_count)
    print("第一条路径所有路径点：", len(path))
    ax.view_init(elev=50, azim=70)
    plt.show()
show_path_3d(start_point, end_point)
#3D次优路径，只含有次优路径，输出次优路径的信息
def show_path_3d1(start, end):
    import math
    # 读取栅格影像数据
    raster_path = "F:/Doctor/data/areaDEM300.tif"
    raster_array = read_raster(raster_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制DEM图
    X = np.arange(0, 300)
    Y = np.arange(0, 300)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, raster_array, cmap='viridis', alpha=0.7)
    total_slope = 0
    slopes = []  # 保存所有的坡度值
    turn_count = 0
    total_distance = 0
    uphill_distance = 0
    downhill_distance = 0
    hill_distance = 0
    uphill_slope = 0
    downhill_slope = 0
    for i in range(len(path1)-1):  # 遍历path中所有点（除了最后一个点）
        if len(path1)>1:
            # 计算每两个点之间的水平距离和垂直距离
            horizontal_distance = math.sqrt((path1[i+1][1] - path1[i][1]) ** 2 + (path1[i+1][0] - path1[i][0]) ** 2) * 30
            vertical_difference = abs(v[path1[i+1][0]][path1[i+1][1]] - v[path1[i][0]][path1[i][1]])
            distance = (horizontal_distance ** 2 + vertical_difference ** 2) ** 0.5
            total_distance += distance
            if horizontal_distance == 0:
                slope_angle = 0  # 设置为90度或其他默认值
            else:
                slope_angle = math.degrees(math.atan(vertical_difference / horizontal_distance))
            slope_angle = round(slope_angle, 2)
            slopes.append(slope_angle)  # 将坡度值添加到列表中
            if slope_angle < 0:
                slope_angle = abs(slope_angle)  # 将负斜率转换为绝对值
            total_slope += slope_angle
            height_difference = v[path1[i+1][0]][path1[i+1][1]] - v[path1[i][0]][path1[i][1]]
            if height_difference > 0:
                uphill_distance += distance
                uphill_slope += slope_angle
            elif height_difference < 0:
                downhill_distance += distance
                downhill_slope += slope_angle
            elif height_difference == 0:
                hill_distance += distance
        # 检查是否有拐弯
    for i in range(1, len(path1) - 1):
        x1, y1 = path1[i - 1]
        x2, y2 = path1[i]
        x3, y3 = path1[i + 1]
        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if cross_product != 0:
            turn_count += 1
    if len(path1) > 1:
        average_slope = total_slope / (len(path1) - 1)
        average_slope = round(average_slope, 2)
    else:
        average_slope = 0
    max_slope = max(slopes)  # 找到所有坡度值中的最大值
    min_slope = min(slopes)
    total_slope = sum(slopes)  # 计算所有坡度值的总和
    # 绘制路径
    for i in range(len(path1) - 1):
        # 获取当前点和下一点的高程
        current_height = raster_array[path1[i][0], path1[i][1]]
        next_height = raster_array[path1[i + 1][0], path1[i + 1][1]]
        # 判断高程变化，设置路径颜色
        if current_height < next_height:
            color = 'red'  # 绿色代表上坡
        elif current_height > next_height:
            color = 'yellow'  # 黄色代表下坡
        else:
            color = 'black'  # 黄色代表下坡
        # 绘制路径段，并设置颜色
        ax.plot([path1[i][1], path1[i + 1][1]], [path1[i][0], path1[i + 1][0]], [current_height, next_height], color=color,
            linewidth=2)
    # 绘制起点和终点
    ax.scatter(start[1], start[0], raster_array[start[0], start[1]] + 1, color='red', marker='o', label='起点')
    ax.scatter(end[1], end[0], raster_array[end[0], end[1]] + 1, color='blue', marker='s', label='终点')
    # 调整坐标轴范围和标签
    ax.set_xlim(0, 301)
    ax.set_ylim(0, 301)
    ax.set_zlim(np.min(raster_array), np.max(raster_array) + 1)
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    ax.set_zlabel('高度')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    print("第二条路径长度:", total_distance)
    print("第二条路径上坡总距离:", uphill_distance)
    print("第二条路径下坡总距离:", downhill_distance)
    print("第二条路径平坦路长离:", hill_distance)
    print("第二条路径总坡度:", total_slope)
    print("第二条路径上坡总坡度:", uphill_slope)
    print("第二条路径下坡总坡度:", downhill_slope)
    print("第二条路径平均坡度:", average_slope)
    print("第二条路径最大坡度:", max_slope)
    print("第二条路径最大坡度:", min_slope)
    print("第二条路径拐弯个数:", turn_count)
    print("第二条路径所有路径点：", len(path))
    ax.view_init(elev=50, azim=70)
    plt.show()
show_path_3d1(start_point, end_point)