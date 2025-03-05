import cv2
import numpy as np
import os

# 主输出目录
main_output_dir = r'E:\Drilling_Phase2\itempoint_logic\50427\projectPoint\itemPoint\teleportation\positivesample\guangliufa_test'
os.makedirs(main_output_dir, exist_ok=True)

# 视频输出目录
video_output_dir = os.path.join(main_output_dir, 'videos')
os.makedirs(video_output_dir, exist_ok=True)

# 图像输出目录
image_output_dir = os.path.join(main_output_dir, 'images')
os.makedirs(image_output_dir, exist_ok=True)

# 读取视频
video_path = r"E:\Drilling_Phase2\itempoint_logic\推送yujing-1s数据\xiepo\right\30171SL-1111-2222-斜坡吊重物时，人从跑道上通过-20250111061812.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

region_size = 16

# 读取第一帧并初始化 gray_0
ret, frame_0 = cap.read()
if not ret:
    print("无法读取第一帧")
    exit()
gray_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

# 打开文件以写入最长箭头的长度和颜色以及矩形坐标
info_file_path = os.path.join(main_output_dir, 'longest_arrows_info.txt')
rectangle_file_path = os.path.join(main_output_dir, 'rectangles_info.txt')
arrow_points_file_path = os.path.join(main_output_dir, 'arrow_points_info.txt')

with open(info_file_path, 'w') as info_file, open(rectangle_file_path, 'w') as rectangle_file, open(arrow_points_file_path, 'w') as arrow_points_file:
    frame_count = 0
    while True:
        ret, frame_1 = cap.read()
        if not ret:
            break
        
        gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(gray_0, gray_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
        # 可视化光流
        h, w = flow.shape[:2]
        img = frame_1.copy()  # 复制原始帧以便绘制光流
        
        longest_arrow_length = 0
        longest_arrow_color = None
        longest_arrow_angle = None
        longest_arrow_positions = []
        all_arrow_positions = []  # 存储所有箭头的位置和长度
        
        # 绘制光流区域
        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                # 提取区域内的光流
                region_flow = flow[y:y+region_size, x:x+region_size]
                
                # 计算区域内的平均光流
                avg_fx = np.mean(region_flow[..., 0])
                avg_fy = np.mean(region_flow[..., 1])
                
                # 计算平均光流的方向
                angle = np.arctan2(avg_fy, avg_fx) * 180 / np.pi  # 计算角度
                
                # 根据角度选择颜色
                hue = int(180 - angle / 2) % 180  # 将角度映射到 0-180 度
                saturation = 255
                value = 255  # 固定亮度
                
                bgr_color = cv2.cvtColor(np.array([[[hue, saturation, value]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
                bgr_color = tuple(bgr_color.tolist())
                
                # 绘制箭头
                center_x = x + region_size // 2
                center_y = y + region_size // 2
                end_x = int(center_x + avg_fx * 4)  # 放大箭头长度以便观察
                end_y = int(center_y + avg_fy * 4)
                arrow_length = np.sqrt((end_x - center_x)**2 + (end_y - center_y)**2)
                cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), bgr_color, 1)
                
                # 记录所有箭头的位置和长度
                all_arrow_positions.append(((center_x, center_y), (end_x, end_y), arrow_length))
                
                # 记录最长箭头
                if arrow_length > longest_arrow_length:
                    longest_arrow_length = arrow_length
                    longest_arrow_color = bgr_color
                    longest_arrow_angle = angle
                    longest_arrow_positions = [((center_x, center_y), (end_x, end_y))]
                elif arrow_length == longest_arrow_length:
                    longest_arrow_positions.append(((center_x, center_y), (end_x, end_y)))
        
        # 写入最长箭头的信息
        if longest_arrow_length > 0:
            info_file.write(f"Frame {frame_count}: Longest Arrow Length: {longest_arrow_length}, Angle: {longest_arrow_angle:.2f}°, Color: {longest_arrow_color}\n")
            
            if longest_arrow_length > 40:
                # 保存最长箭头的起点终点坐标
                for start, end in longest_arrow_positions:
                    arrow_points_file.write(f"Frame {frame_count}: Start Point: {start}, End Point: {end}\n")
                
                # 寻找附近的箭头
                nearby_arrows = []
                for start, end, length in all_arrow_positions:
                    if abs(length - longest_arrow_length) < 10:  # 长度相近的阈值
                        nearby_arrows.append((start, end, length))
                
                # 构建矩形
                if len(nearby_arrows) >= 4:
                    # 这里假设我们找到了四个合适的箭头来构建矩形
                    # 实际应用中可能需要更复杂的逻辑来确保形成矩形
                    rect_points = []
                    for start, end, _ in nearby_arrows:
                        rect_points.extend([start, end])
                    
                    # 使用最小外接矩形算法来拟合这些点
                    rect = cv2.minAreaRect(np.array(rect_points))
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # 保存矩形坐标信息
                    rectangle_file.write(f"Frame {frame_count}: Rectangle Points: {box.tolist()}\n")
                    
                    # 在图像上标记矩形
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        
        # 显示叠加后的图像
        # cv2.imshow('Optical Flow with Video Regions', img)

        # 写入输出视频
        # out.write(img)

        # 保存当前帧图像
        image_output_path = os.path.join(image_output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(image_output_path, img)
    
        # 更新前一帧
        gray_0 = gray_1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

cap.release()
# out.release()
cv2.destroyAllWindows()