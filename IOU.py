#求交并比iou
import os.path
from collections import deque
import cv2
import numpy as np
import time as tm
from shapely.geometry import Polygon, box
def get_IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    return S_cross / (S1 + S2 - S_cross)

def calculate_intersection_area(rectangle, polygon_points):
    """
    计算矩形框和多边形的重合面积，再除以多边形面积。矩形框的坐标格式是[x1,y1,x2,y2]，其中左上角x1,y1,右下角x2,y2。
    多边形的坐标是[{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, {'x': x3, 'y': y3},...]
    创建矩形对象
    """
    x1, y1, x2, y2 = rectangle
    rect = box(x1, y1, x2, y2)

    # 创建多边形对象
    polygon = Polygon([(point['x'], point['y']) for point in polygon_points])

    # 计算交集
    intersection = rect.intersection(polygon)

    # 计算交集面积
    intersection_area = intersection.area

    # 计算多边形面积
    polygon_area = polygon.area

    # 返回交集面积与多边形面积的比值
    if polygon_area == 0:
        return 0  # 避免除以零的情况
    
    return intersection_area / polygon_area

# 导入必要的模块
from IOU import get_IOU, calculate_intersection_area

# 示例数据
rec1 = (10, 10, 50, 50)  # 矩形框1
rec2 = (30, 30, 70, 70)  # 矩形框2

rectangle = (10, 10, 50, 50)  # 矩形框
polygon_points = [
    {'x': 20, 'y': 20},
    {'x': 40, 'y': 20},
    {'x': 40, 'y': 40},
    {'x': 20, 'y': 40}
]  # 多边形的顶点

# 调用 get_IOU 函数
iou = get_IOU(rec1, rec2)
print(f"IOU of rec1 and rec2: {iou}")

# 调用 calculate_intersection_area 函数
intersection_ratio = calculate_intersection_area(rectangle, polygon_points)
print(f"Intersection area ratio of rectangle and polygon: {intersection_ratio}")