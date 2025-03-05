#高斯混合模型GMM
import cv2

def detect_motion(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 初始化高斯混合模型背景减除器
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 应用高斯混合模型进行背景减除
        fgmask = fgbg.apply(frame)
        
        # 对前景掩码进行形态学操作以去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # 计算前景掩码中的白色像素数量
        motion_pixels = cv2.countNonZero(fgmask)
        
        # 判断是否有运动发生
        if motion_pixels > 500:  # 这个阈值可以根据实际情况调整
            print("Movement detected!")
        else:
            print("No movement.")
        
        # 显示结果
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgmask)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
video_path = r"E:\Drilling_Phase2\itempoint_logic\推送yujing-1s数据\xiepo\right\30171SL-1111-2222-斜坡吊重物时，人从跑道上通过-20250111061812.mp4"
detect_motion(video_path)