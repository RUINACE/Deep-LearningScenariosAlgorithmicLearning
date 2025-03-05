import cv2
import numpy as np

def detect_motion_frame_diff(video_path, threshold=30):
    """
    使用帧差法检测视频中的运动。
    
    参数:
    video_path (str): 视频文件的路径
    threshold (int): 像素差异的阈值，默认为30
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
    
    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 将当前帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算帧差
        frame_diff = cv2.absdiff(prev_gray, gray)
        
        # 应用阈值处理
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 对阈值图像进行形态学操作以去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 计算阈值图像中的白色像素数量
        motion_pixels = cv2.countNonZero(thresh)
        
        # 判断是否有运动发生
        if motion_pixels > 500:  # 这个阈值可以根据实际情况调整
            print("Movement detected!")
        else:
            print("No movement.")
        
        # 显示结果
        cv2.imshow('Frame', frame)
        cv2.imshow('Frame Difference', frame_diff)
        cv2.imshow('Threshold', thresh)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        # 更新前一帧
        prev_gray = gray
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
video_path = r"E:\Drilling_Phase2\itempoint_logic\推送yujing-1s数据\xiepo\right\30171SL-1111-2222-斜坡吊重物时，人从跑道上通过-20250111061812.mp4"
detect_motion_frame_diff(video_path)