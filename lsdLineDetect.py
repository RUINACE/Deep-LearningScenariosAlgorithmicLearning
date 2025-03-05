# lsd 直线检测
import cv2
import numpy as np
import collections

def GetLSDLines_louti(f, path, img, index_frame, loutiname, loutiNumber):
    """
    获取LSD直线检测结果，返回最适合的斜率
    """
    image = img.copy()
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 膨胀
    image = cv2.dilate(image, kernel_2)
    # 腐蚀
    image = cv2.erode(image, kernel_1)
    # 去噪
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化核函数
    image = cv2.filter2D(image, -1, kernel=kernel)
    # 将彩色图片灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 去噪处理
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # 边缘增强
    gray = cv2.Laplacian(denoised, cv2.CV_8U, ksize=3)
    edges = cv2.Canny(gray, 50, 150)
    # LSD直线检测
    # 创建一个LSD对象
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    output = np.copy(image)

    # 定义一个字典来存储斜率及其出现的次数
    slopes = collections.defaultdict(int)
    # 存储每条线段的信息
    lines_with_slope = []
    positive_slope_count = 0
    negative_slope_count = 0
    # 计算每条直线的斜率，并统计每个斜率出现的次数
    if lines is None:
        f.write("找不到直线\n")       
        return 0, 1
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)
        # 更新斜率计数器
        if slope > 0:
            positive_slope_count += 1
        elif slope < 0:
            negative_slope_count += 1
        slopes[slope] += 1
        lines_with_slope.append((slope, (x1, y1, x2, y2)))
        # cv2.line(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    height ,width= output.shape[:2]
    if positive_slope_count > negative_slope_count:
        stair = 1  # 下降，都是正数
        duijiaoxianSlope = (height / width)
        if duijiaoxianSlope < 0:
            duijiaoxianSlope = -duijiaoxianSlope
            f.write("对角线Slope为负，不符合实际：{}\n".format(duijiaoxianSlope))
        lengths = [(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), (x1, y1, x2, y2), slope) 
                   for slope, (x1, y1, x2, y2) in lines_with_slope 
                   if slope > 0]  # 只取下降的线段
    else:
        stair = 0  # 上升，都是负数
        duijiaoxianSlope = -(height / width)
        lengths = [(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), (x1, y1, x2, y2), slope) 
                   for slope, (x1, y1, x2, y2) in lines_with_slope 
                   if slope < 0]  # 只取上升的线段
        
    differences = [(abs(line[2] - duijiaoxianSlope), line[0], line[1], line[2]) for line in lengths]
    # f.write("duijiaoxianSlope：{}\n".format(duijiaoxianSlope))
    # differences.sort(key=lambda x: (x[0], -x[1]))  # 按对角线差值排序，差值小的排前面，差值相同，长度长的排前面
    differences.sort(key=lambda x: (-x[1], x[0]))  # 按对角线差值排序，长度长的排前面,相同差值小的排前面

    if differences:
        for i in range(len(differences)):
           if differences[i][0] < 0.2:
               break
        difference, _, (x1, y1, x2, y2), slope = differences[i]
        if difference > 1:
            f.write("差值大于1，使用对角线斜率\n")
            slope = duijiaoxianSlope
        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.5
        # font_color = (0, 0, 255)
        # thickness = 1
        # label = str(duijiaoxianSlope)  # 获取标签
        # text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        # text_x = x2 + 10  # 在矩形框右侧10像素处
        # text_y = y1 + text_size[1] // 2  # 文本垂直居中
        # cv2.putText(output, label, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imencode(".jpg", output)[1].tofile(str(path.joinpath(f'{index_frame}lsd_line{loutiname}{loutiNumber}.jpg')))
        # f.write("斜率：{}\n".format(slope))
        return slope
    return 0



def GetLSDLines(img):
    """
    获取LSD直线检测结果，返回所有检测到的直线
    """
    image = img.copy()
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 膨胀
    image = cv2.dilate(image, kernel_2)
    # 腐蚀
    image = cv2.erode(image, kernel_1)
    # 去噪
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化核函数
    image = cv2.filter2D(image, -1, kernel=kernel)
    # 将彩色图片灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 去噪处理
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # 边缘增强
    gray = cv2.Laplacian(denoised, cv2.CV_8U, ksize=3)
    edges = cv2.Canny(gray, 50, 150)
    # LSD直线检测
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(edges)
    
    if lines is None:
        return []
    
    # 提取直线信息
    lines = lines.reshape(-1, 4)
    return lines

# 示例调用
if __name__ == "__main__":
    import cv2

    # 读取视频文件
    video_path = r"E:\Drilling_Phase2\itempoint_logic\推送yujing-1s数据\xiepo\right\30171SL-1111-2222-斜坡吊重物时，人从跑道上通过-20250111061812.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
        else:
            # 调用 GetLSDLines 函数
            lines = GetLSDLines(frame)
            print("Detected lines:", lines)

            # 显示检测到的直线
            output = frame.copy()
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imshow('Detected Lines', output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cap.release()