#模型推理所用的包 
import numpy as np
import cv2
import os
from hobot_dnn import pyeasy_dnn as dnn
from bputools.format_convert import imequalresize, bgr2nv12_opencv
import lib.pyyolotools as yolotools

#ROS2所用的包
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import math  # 用于角度和弧度转换

#用于main_testFPS.py
import threading
import queue

#main.py

def model_inference(frame, model_path, classes_name_path):
    """
        param: 
            frame: 图像数据 (由 img_get() 函数通过 OpenCV 获取)
            model_path: 模型路径
            classes_name_path: 类别标签路径

        returns:
            class_ids: 标签类别
            confidences: 置信度
            boxes: 检测目标的边界框 (x, y, w, h)
            class_list: 类别名称
            colors: RGB 颜色
    """
    def get_hw(pro):
        if pro.layout == "NCHW":
            return pro.shape[2], pro.shape[3]
        else:
            return pro.shape[1], pro.shape[2]

    def format_yolov5(frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    # 设置参数
    thre_confidence = 0.4
    thre_score = 0.25
    thre_nms = 0.45
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    # 加载模型
    models = dnn.load(model_path)
    model_h, model_w = get_hw(models[0].inputs[0].properties)
    print(f"模型输入尺寸: {model_h}x{model_w}")

    # 加载类别名
    with open(classes_name_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    # 图像预处理
    inputImage = format_yolov5(frame)
    img = imequalresize(inputImage, (model_w, model_h))
    nv12 = bgr2nv12_opencv(img)

    # 模型推理
    t1 = cv2.getTickCount()
    outputs = models[0].forward(nv12)
    t2 = cv2.getTickCount()
    outputs = outputs[0].buffer
    print(f'推理时间: {(t2 - t1) * 1000 / cv2.getTickFrequency()} ms')

    # 后处理
    image_width, image_height, _ = inputImage.shape
    fx, fy = image_width / model_w, image_height / model_h
    try:
        class_ids, confidences, boxes = yolotools.pypostprocess_yolov5(
            outputs[0][:, :, 0], fx, fy, thre_confidence, thre_score, thre_nms
        )
    except Exception as e:
        print(f"后处理出现异常: {str(e)}")
        class_ids, confidences, boxes = None, None, None

    return class_ids, confidences, boxes, class_list, colors


def img_convert(image_path):
    """
        param:
            image_path: 待检测图像路径

        returns:
            frame: 图像数据
    """
    # 加载图像
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法加载图像文件: {image_path}")
        return None
    return frame


def img_show(class_ids, confidences, boxes, class_list, colors, frame):
    # 绘制检测框
    if class_ids is not None and confidences is not None and boxes is not None:
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, f"{class_list[classid]} {confidence:.2f}", 
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
    # 显示结果
    cv2.imshow("Detection Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def persecond_img_catch(output_folder, flag, cap):
    """
    实时从摄像头拍摄照片并存储到指定文件夹，每次调用时自动拍摄一张图片，编号从1开始。
    
    param:
        output_folder: 存储照片的文件夹路径。
        flag: 输入值, False为没拍照; True为拍照(暂时留空)
        
    return:
        latest_photo_path: 最新拍摄照片的完整路径（字符串），
        如果未拍摄任何照片则返回文件夹里编号最新的图片的路径。
    """

    # 检查存储文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取当前已有文件的最大编号
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    if existing_files:
        max_index = max([int(os.path.splitext(f)[0]) for f in existing_files if f.split('.')[0].isdigit()])
    else:
        max_index = 0

    # 打开摄像头
    cap = cv2.VideoCapture(8)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None

    print("每次调用时拍摄一张照片，按 'q' 键退出程序。")
    latest_photo_path = None

    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        cap.release()
        cv2.destroyAllWindows()
        return None

    # 显示实时画面
    # cv2.imshow("Press 'q' to quit", frame)

    # 判断是否拍照
    if flag == True:
        max_index += 1
        latest_photo_path = os.path.join(output_folder, f"{max_index}.jpg")
        cv2.imwrite(latest_photo_path, frame)
        print(f"照片已保存: {latest_photo_path}")

    # 释放摄像头资源并关闭窗口
    cap.release() #交给main.py中的main函数
    # cv2.destroyAllWindows()

    # 如果拍摄了照片，则返回最新拍摄的照片路径；否则返回编号最新的图片路径
    if latest_photo_path:
        return latest_photo_path
    else:
        # 返回文件夹中编号最新的图片路径
        if existing_files:
            latest_existing_photo = os.path.join(output_folder, f"{max_index}.jpg")
            return latest_existing_photo
        else:
            return None


def block_in_safety(class_ids, boxes, class_list, frame):
    """
    过滤掉 'blue_safe', 'yellow_safe', 'black_safe' 标签，
    并且移除位于 safety_blue 匂域内的 'blue', 'yellow', 'black' 类物块。
    
    param:
        class_ids: 检测到的类别 ID 列表
        boxes: 检测到的框列表，每个框格式为 (x, y, w, h)
        class_list: 类别名称列表
        frame: 当前图像（用于处理和显示）

    return:
        new_class_ids: 剔除安全区物块后的类别 ID 列表
        new_boxes: 剔除安全区物块后的边界框列表
        new_class_list: 更新后的类别名称列表
        frame_with_mask: 剔除区域后的图像
    """
    # 参数检查，确保传入的参数非空
    if (class_ids is None) or (boxes is None) or (class_list is None) or (frame is None or frame.size == 0):
        print("输入参数为空，无法处理！")
        return [], [], [], frame  # 返回空列表和原始图像

    safety_blue_box = None  # 用于存储安全区的坐标 (x, y, w, h)

    # 遍历 class_ids 来找到 safety_blue 区域
    for class_id, box in zip(class_ids, boxes):
        if class_list[class_id] == 'safety_blue':
            safety_blue_box = box
            break

    if safety_blue_box is None:
        print("没有找到 safety_blue 区域！")
        return class_ids, boxes, class_list, frame

    # 获取安全区的坐标 (x, y, w, h)
    x, y, w, h = safety_blue_box

    # 遍历物块类别并根据条件删除
    new_class_ids = []
    new_boxes = []
    new_class_list = []  # 新的类别列表
    for class_id, box in zip(class_ids, boxes):
        class_name = class_list[class_id]
        cx, cy, _, _ = box  # 获取物块的中心坐标
        
        # 如果是 'blue_safe', 'yellow_safe', 'black_safe' 标签则跳过
        if class_name in ['blue_safe', 'yellow_safe', 'black_safe']:
            continue
        
        # 判断物块是否在 safety_blue 区域内
        if class_name in ['blue', 'yellow', 'black']:
            if x <= cx <= x + w and y <= cy <= y + h:
                continue  # 剔除在安全区内的物块

        # 如果不在安全区内或不是需要剔除的物块，保留
        new_class_ids.append(class_id)
        new_boxes.append(box)
        new_class_list.append(class_name)  # 添加类别名到新的列表

    # 不再生成掩膜，直接返回更新后的值
    return new_class_ids, new_boxes, new_class_list, frame



def max_area(class_ids, boxes, frame):
    """
    根据 class_ids 和 boxes 计算每种颜色中最大面积的框及其质心坐标，
    并在图像上标记这些框。

    param:
        class_ids: 检测到的类别列表或 NumPy 数组
        boxes: 检测到的边界框列表或 NumPy 数组 [(x, y, w, h)]
        frame: 原始图像，用于绘制边界框

    return:
        blue_max: 蓝色框的最大面积
        blue_centroid: 蓝色框最大面积对应物体的质心坐标 (x, y)
        yellow_max: 黄色框的最大面积
        yellow_centroid: 黄色框最大面积对应物体的质心坐标 (x, y)
        black_max: 黑色框的最大面积
        black_centroid: 黑色框最大面积对应物体的质心坐标 (x, y)
        frame: 绘制了最大面积物体边界框的图像
    """
    # 定义类别与颜色的映射
    color_mapping = {0: "blue", 2: "yellow", 4: "black"}  # 修改为对应正确的类别编号
    color_bgr = {"blue": (255, 0, 0), "yellow": (0, 255, 255), "black": (0, 0, 0)}  # BGR颜色值保持不变

    # 初始化每种颜色的最大面积和质心
    max_areas = {"blue": 0, "yellow": 0, "black": 0}
    centroids = {"blue": None, "yellow": None, "black": None}
    max_boxes = {"blue": None, "yellow": None, "black": None}

    # 检查输入是否为空或类型不匹配
    if not isinstance(boxes, (list, np.ndarray)) or not isinstance(class_ids, (list, np.ndarray)):
        return max_areas["blue"], centroids["blue"], max_areas["yellow"], centroids["yellow"], max_areas["black"], centroids["black"], frame

    # 如果 boxes 或 class_ids 为空，直接返回默认值
    if len(boxes) == 0 or len(class_ids) == 0:
        return max_areas["blue"], centroids["blue"], max_areas["yellow"], centroids["yellow"], max_areas["black"], centroids["black"], frame

    # 遍历检测结果，计算每个框的面积
    for i, box in enumerate(boxes):
        x, y, w, h = box
        area = w * h

        # 获取当前框的类别和对应颜色
        class_id = class_ids[i]
        color = color_mapping.get(class_id, None)

        # 更新对应颜色的最大面积、质心和框
        if color and area > max_areas[color]:
            max_areas[color] = area
            centroids[color] = (x + w // 2, y + h // 2)
            max_boxes[color] = box

    # 提取不同颜色的最大面积和质心
    blue_max, blue_centroid = max_areas["blue"], centroids["blue"]
    yellow_max, yellow_centroid = max_areas["yellow"], centroids["yellow"]
    black_max, black_centroid = max_areas["black"], centroids["black"]

    return blue_max, blue_centroid, yellow_max, yellow_centroid, black_max, black_centroid, frame




def target_choose(blue_max, blue_centroid, yellow_max, yellow_centroid, black_max, black_centroid, frame):
    """
    根据不同颜色的最大面积，选择最大的那个，并返回最大值及其对应的质心坐标。

    param:
        blue_max: 蓝色框的最大面积
        blue_centroid: 蓝色框最大面积对应物体的质心坐标 (x, y)
        yellow_max: 黄色框的最大面积
        yellow_centroid: 黄色框最大面积对应物体的质心坐标 (x, y)
        black_max: 黑色框的最大面积
        black_centroid: 黑色框最大面积对应物体的质心坐标 (x, y)
        frame: 原始图像，用于绘制边界框

    return:
        max_area: 最大面积的值
        max_centroid: 最大面积对应物体的质心坐标 (x, y)
        frame: 绘制了最大面积物体边界框的图像
        class_ids: 对应的物体类别列表
        boxes: 对应物体的边界框列表
    """

    # 创建一个包含所有颜色的最大面积及对应质心的列表
    areas_and_centroids = [
        (blue_max, blue_centroid, 0),  # 蓝色对应类别 0
        (yellow_max, yellow_centroid, 1),  # 黄色对应类别 1
        (black_max, black_centroid, 2)  # 黑色对应类别 2
    ]
    
    # 按照最大面积从大到小排序
    areas_and_centroids.sort(key=lambda x: x[0], reverse=True)

    # 获取最大面积及其对应的质心和类别
    max_area, max_centroid, class_id = areas_and_centroids[0]

    # 如果最大面积对应的质心为 None，则说明没有识别到物体
    if max_centroid is None:
        return max_area, None, frame, [], [], 0

    # 根据最大面积的目标绘制边界框和质心
    if max_centroid is not None:
        x, y = max_centroid
        color_bgr = (0, 255, 0)  # 绿色，用于绘制框和质心
        
    # 假设的边界框宽高
    w = h = int(math.sqrt(max_area))  # 通过面积计算宽度和高度

    # 返回最大面积、质心、图像以及 class_ids 和 boxes
    class_ids = [class_id]  # 仅返回选择的目标类别
    boxes = [[x - w // 2, y - h // 2, w, h]]  # 仅返回选择的边界框

    return max_area, max_centroid, frame, class_ids, boxes, w


def if_in_range(frame, blue_centroid, yellow_centroid, black_centroid):
    """
    判断蓝色、黄色、黑色质心坐标是否在指定范围内，并实时显示视频流。
    
    param:
        frame: 实时视频帧。
        blue_centroid: 蓝色框最大面积的质心坐标 (x, y) 或 None。
        yellow_centroid: 黄色框最大面积的质心坐标 (x, y) 或 None。
        black_centroid: 黑色框最大面积的质心坐标 (x, y) 或 None。

    return:
        blue_in_range: 蓝色质心是否在指定范围内 (True/False)。
        yellow_in_range: 黄色质心是否在指定范围内 (True/False)。
        black_in_range: 黑色质心是否在指定范围内 (True/False)。
    """

    # 指定范围
    x_min, x_max = 145, 430
    y_min, y_max = 340, 480

    # 初始化结果
    blue_in_range = False
    yellow_in_range = False
    black_in_range = False

    # 定义颜色映射
    color_bgr = {"blue": (255, 0, 0), "yellow": (0, 255, 255), "black": (0, 0, 0)}

    # 判断蓝色质心是否在范围内
    if blue_centroid:
        bx, by = blue_centroid
        blue_in_range = x_min <= bx <= x_max and y_min <= by <= y_max
        color = (0, 255, 0) if blue_in_range else (0, 0, 255)
        cv2.putText(frame, f"Blue: {blue_in_range}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 判断黄色质心是否在范围内
    if yellow_centroid:
        yx, yy = yellow_centroid
        yellow_in_range = x_min <= yx <= x_max and y_min <= yy <= y_max
        color = (0, 255, 0) if yellow_in_range else (0, 0, 255)
        cv2.putText(frame, f"Yellow: {yellow_in_range}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 判断黑色质心是否在范围内
    if black_centroid:
        kx, ky = black_centroid
        black_in_range = x_min <= kx <= x_max and y_min <= ky <= y_max
        color = (0, 255, 0) if black_in_range else (0, 0, 255)
        cv2.putText(frame, f"Black: {black_in_range}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 返回不同颜色是否在范围内
    return blue_in_range, yellow_in_range, black_in_range


def run_if_in_range(video_source, model_path, classes_name_path):
    """
    #( 调试范围函数 )
    调用 block_pick 并实时打印蓝色、红色、黑色质心是否在范围内的结果。
    
    参数:
        video_source: 视频流源（例如摄像头索引或视频文件路径）。
        model_path: 模型文件路径。
        classes_name_path: 类别名称文件路径。
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 键退出程序。\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频流")
            break

        # 调用推理和后处理（使用假设函数 model_inference 和 max_area）
        class_ids, confidences, boxes, class_list, colors = model_inference(frame, model_path, classes_name_path)
        blue_max, blue_centroid, red_max, red_centroid, black_max, black_centroid, frame = max_area(
            class_ids, boxes, frame
        )

        # 调用 block_pick 并获取结果
        blue_in_range, red_in_range, black_in_range = if_in_range(frame, blue_centroid, red_centroid, black_centroid)

        # 实时打印结果
        print(f"Blue: {blue_in_range}, Red: {red_in_range}, Black: {black_in_range}")

        # 按键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("退出程序")
            break

    cap.release()
    cv2.destroyAllWindows()


def centroid_align(class_ids=None, boxes=None, image=None, max_centroid=None, screen_width=640, fov=120, publisher_=None):
    """
    根据静态图片中的质心位置调整机器人方向，使其对准屏幕中心线。

    param:
        screen_width: 图像水平分辨率，默认 640 像素
        fov: 摄像头水平视场角（角度），默认 120°
        class_ids: 检测到的类别 ID 列表
        boxes: 检测到的框列表，每个框格式为 (x, y, w, h)
        image: 当前的静态图像（由外部代码加载）
        max_centroid: 当前选择目标的质心坐标 (x, y)
        publisher_: 传入的发布者，用于发送速度指令

    return:
        is_align: 如果调整方向成功，返回 True，否则返回 False
    """
    is_align = False

    # 检查是否提供了图像
    if image is None:
        print("没有提供图像，无法进行调整。")
        return is_align

    # 计算每像素角度分辨率（以弧度为单位）
    angle_per_pixel = (fov / screen_width) * (math.pi / 180)  # 每像素弧度
    print(f"每像素弧度分辨率: {angle_per_pixel:.6f} 弧度")

    # 计算屏幕中心线的位置
    center_x = screen_width // 2
    angular_speed = 0.9  # 固定角速度 (rad/s)

    # 用于标志旋转任务是否完成
    rotation_done = False

    # 如果 max_centroid 存在且旋转任务尚未完成
    if max_centroid and not rotation_done:
        cx, _ = max_centroid

        # 计算目标偏移角度（以弧度为单位）
        target_angle = (cx - center_x) * angle_per_pixel

        # 判断质心在中心线的哪一侧，并设定旋转方向
        if target_angle > 0:
            direction = "右侧"
            angular_sign = -1  # 右旋为负
        else:
            direction = "左侧"
            angular_sign = 1  # 左旋为正

        # 打印方向信息
        print(f"质心在中心线的{direction}，目标角度: {target_angle:.6f} rad")

        # 累计旋转角度逻辑
        total_rotated_angle = 0.0
        start_time = time.time()

        while abs(target_angle) > abs(total_rotated_angle):
            current_time = time.time()
            elapsed_time = current_time - start_time

            # 计算当前旋转角度
            delta_angle = angular_speed * elapsed_time
            total_rotated_angle += delta_angle * angular_sign

            # 更新 Twist 消息
            twist = Twist()
            twist.angular.z = angular_speed * angular_sign
            publisher_.publish(twist)

            # 打印当前累计旋转角度
            print(f"累计旋转角度: {total_rotated_angle:.6f} rad, 目标角度: {target_angle:.6f} rad")

            # 更新开始时间
            start_time = current_time

        # 旋转完成后停止机器人
        twist = Twist()
        twist.angular.z = 0.0
        publisher_.publish(twist)
        print("旋转任务完成，停止机器人")

        # 设置完成标志
        rotation_done = True

    # 任务完成后退出
    if rotation_done:
        print("已完成所有任务，准备退出节点")
        is_align = True

    return is_align


def go_to_block(judge_to_block, publisher_):
    """
    控制小车根据 judge_to_block 的值决定是否继续前进。

    :param judge_to_block: bool, 若为 True，则小车停止；若为 False，则小车前进。(由if_in_range函数判断)
    :param publisher_: rclpy 的 publisher，负责发布速度指令(形参保证速度指令仅发布一次)
    """
    twist = Twist()

    if not judge_to_block:
        twist.linear.x = 0.25  # 设置线速度为 0.15 m/s 向前移动
        publisher_.publish(twist)  # 发送指令
        print("小车开始前进，速度为 0.15 m/s")
        return 0
    else:
        twist.linear.x = 0.0  # 停止小车
        publisher_.publish(twist)  # 发送指令
        print("小车停止运动。")
        return 1


def get_safety_blue(class_ids, boxes, class_list):
    """
    提取与 'safety_blue' 标签对应的框的质心。

    param:
        class_ids: 类别列表（由model_inference函数返回）
        boxes: 边界框列表（由model_inference函数返回）
        class_list: 类别名称列表（由model_inference函数返回）

    return:
        safety_blue_centroid: safety_blue 标签对应的框的质心坐标 (x, y)
    """
    safety_blue_centroid = None  # 初始化返回的质心坐标为 None

    # 检查class_ids和boxes是否为空（使用 .size 判断）
    if class_ids is None or boxes is None or class_ids.size == 0 or boxes.size == 0:
        print("没有检测到有效的目标数据，无法提取质心。")
        return None

    # 遍历所有检测到的类别
    for class_id, box in zip(class_ids, boxes):
        class_name = class_list[class_id]
        
        # 检查是否是 'safety_blue' 类别
        if class_name == "safety_blue":
            x, y, w, h = box
            # 计算质心坐标
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            safety_blue_centroid = (centroid_x, centroid_y)
            break  # 找到第一个 safety_blue 后退出循环

    return safety_blue_centroid


def turn_angle(publisher_, angle):
    """
    控制小车旋转指定的角度，角度范围 -360 到 360 度。
    
    param:
        publisher_: ROS2 发布者，用于发送速度指令。
        angle: 旋转角度，范围在 -360 到 360 度之间。
               正值表示顺时针旋转，负值表示逆时针旋转。
    """
    if not (-360 <= angle <= 360):
        print("错误：角度范围应在 -360 到 360 之间！")
        return

    # 目标旋转角度（转换为弧度）
    target_angle = angle * (3.141592653589793 / 180)  # 转换为弧度
    
    # 固定角速度 (rad/s)
    angular_speed = 0.9  # 调整角速度以适应你的机器人
    
    # 旋转角度累计
    total_rotated_angle = 0.0
    
    # 记录起始时间
    start_time = time.time()
    
    # 判断旋转方向
    angular_sign = 1 if angle < 0 else -1  # 负角度逆时针，正角度顺时针
    
    while abs(total_rotated_angle) < abs(target_angle):
        # 计算经过的时间
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # 计算当前旋转的角度
        delta_angle = angular_speed * elapsed_time
        total_rotated_angle += delta_angle * angular_sign
        
        # 发布旋转指令
        twist = Twist()
        twist.angular.z = angular_speed * angular_sign
        publisher_.publish(twist)
        
        # 打印当前累计旋转角度
        print(f"累计旋转角度: {total_rotated_angle:.6f} rad, 目标角度: {target_angle:.6f} rad")
        
        # 更新起始时间
        start_time = current_time
        
    # 旋转完成后停止机器人
    twist = Twist()
    twist.angular.z = 0.0
    publisher_.publish(twist)
    
    print(f"旋转 {angle}° 完成，停止机器人。")


def move_distance(publisher_, distance, speed=0.2):
    """
    控制小车移动指定的距离，单位为米。

    param:
        publisher_: ROS2 发布者，用于发送速度指令。
        distance: 目标行驶距离，单位为米。
        speed: 小车的行驶速度，单位为米每秒，默认值为 0.2 米/秒。
    """
    if distance == 0:
        print("错误：目标距离必须大于 0 米！")
        return
    
    # 计算所需时间（秒）
    time_to_move = abs(distance) / abs(speed)  # 目标时间 = 距离 / 速度（取绝对值）
    
    # 记录起始时间
    start_time = time.time()
    
    # 设置小车前进或倒退
    twist = Twist()
    twist.linear.x = speed if distance > 0 else -speed  # 根据距离的符号决定前进或倒退
    publisher_.publish(twist)
    
    direction = "前进" if distance > 0 else "倒退"
    print(f"开始{direction}，目标距离: {abs(distance)} 米，预计时间: {time_to_move:.2f} 秒。")
    
    # 累计已行驶距离
    total_travelled_distance = 0.0
    
    while total_travelled_distance < abs(distance):
        # 计算经过的时间
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # 计算已行驶的距离
        total_travelled_distance = abs(speed * elapsed_time)  # 使用绝对值处理倒退情况
        
        # 打印已行驶的距离
        print(f"已行驶距离: {total_travelled_distance:.2f} 米, 目标距离: {abs(distance)} 米")
    
    # 停止小车
    twist.linear.x = 0.0  # 停止前进
    publisher_.publish(twist)
    
    print(f"已完成行驶 {abs(distance)} 米，停止机器人。")


def self_spin_block(cap, output_folder, model_path, classes_name_path, car_control_publisher):
    """
    小车不停地原地自转，每秒拍摄一张照片并进行推理，如果检测到目标则停止。

    param:
        cap: 摄像头的cv2.VideoCapture对象。
        output_folder: 存储照片的文件夹路径。
        model_path: 训练模型的路径。
        classes_name_path: 类别名称文件的路径。
        car_control_publisher: ROS2发布器，用于控制小车。

    return:
        True: 如果检测到目标，停止并返回True。
        False: 如果未检测到目标，继续拍照和推理。
    """
    # 设置自转角速度
    angular_velocity = 0.95  # 你可以根据实际需求调整角速度的值
    
    # 创建Twist消息并设置角速度
    twist = Twist()
    twist.angular.z = angular_velocity
    
    # 发布角速度控制指令，让小车开始自转
    car_control_publisher.publish(twist)
    
    flag = False
    while not flag:
        # 每次拍摄一张照片
        latest_photo_path = persecond_img_catch(output_folder, True, cap)
        image = img_convert(latest_photo_path)

        if True:
            # 进行模型推理
            class_ids, confidences, boxes, class_list, colors = model_inference(image, model_path, classes_name_path)
            
            # 判断是否检测到目标
            if class_ids is not None and confidences is not None:
                # 检查是否有有效的检测结果
                if len(class_ids) > 0 and len(confidences) > 0:
                    for idx, confidence in zip(class_ids, confidences):  # 使用zip将class_ids和confidences配对
                        if confidence > 0.4:  # 根据设定的置信度阈值判断
                            # 获取对应的类别名称
                            class_name = class_list[idx]
                            print(f"检测到目标: {class_name}, 置信度: {confidence}")

                            # 判断目标是否为蓝色、黄色或黑色
                            if class_name.lower() in ['blue', 'yellow', 'black']:
                                print("检测到目标为蓝色、黄色或黑色，停止拍照。")
                                flag = True    
            print("未检测到目标，继续拍照。")
        
        # 等待1秒钟再拍摄下一张照片
        time.sleep(1)

    # 停止小车自转，设置angular.z为0
    twist.angular.z = 0.0
    car_control_publisher.publish(twist)

    return flag


def self_spin_safety(cap, output_folder, model_path, classes_name_path, car_control_publisher):
    """
    小车不停地原地自转，每秒拍摄一张照片并进行推理，如果检测到目标则停止。

    param:
        cap: 摄像头的cv2.VideoCapture对象。
        output_folder: 存储照片的文件夹路径。
        model_path: 训练模型的路径。
        classes_name_path: 类别名称文件的路径。
        car_control_publisher: ROS2发布器，用于控制小车。

    return:
        True: 如果检测到目标，停止并返回True。
        False: 如果未检测到目标，继续拍照和推理。
    """
    # 设置自转角速度
    angular_velocity = 0.95  # 你可以根据实际需求调整角速度的值
    
    # 创建Twist消息并设置角速度
    twist = Twist()
    twist.angular.z = angular_velocity
    
    # 发布角速度控制指令，让小车开始自转
    car_control_publisher.publish(twist)
    
    flag = False
    while not flag:
        # 每次拍摄一张照片
        latest_photo_path = persecond_img_catch(output_folder, True, cap)
        image = img_convert(latest_photo_path)

        if True:
            # 进行模型推理
            class_ids, confidences, boxes, class_list, colors = model_inference(image, model_path, classes_name_path)
            
            # 判断是否检测到目标
            if class_ids is not None and confidences is not None:
                # 检查是否有有效的检测结果
                if len(class_ids) > 0 and len(confidences) > 0:
                    for idx, confidence in zip(class_ids, confidences):  # 使用zip将class_ids和confidences配对
                        if confidence > 0.4:  # 根据设定的置信度阈值判断
                            # 获取对应的类别名称
                            class_name = class_list[idx]
                            print(f"检测到目标: {class_name}, 置信度: {confidence}")
                            # 判断目标是否为蓝色、黄色或黑色
                            if class_name.lower() == "safety_blue":
                                print("检测到目标为蓝色、黄色或黑色，停止拍照。")
                                flag = True    
            print("未检测到目标，继续拍照。")
        
        # 等待1秒钟再拍摄下一张照片
        time.sleep(1)

    # 停止小车自转，设置angular.z为0
    twist.angular.z = 0.0
    car_control_publisher.publish(twist)

    return flag


def Quit_status(): #手柄退出按键
    with open("/userdata/create_files/gamepad/Q_status.txt", 'r') as f:
        if f.read() == "True":
            return True
        else:
            return False

if __name__ == "__main__":
    image_path = "/userdata/create_files/catch_img/captured_images/image_182.jpg"  # 图像路径
    model_path = "yolov5s_672x672_nv12_colorrec1.bin"  # 模型路径
    classes_name_path = "colorrec.names"  # 类别文件路径
    
    # # 加载图像
    # frame = img_convert(image_path)
    # if frame is None:
    #     exit(1)
    
    # # 模型推理
    # class_ids, confidences, boxes, class_list, colors = model_inference(frame, model_path, classes_name_path)

    # # 显示图片
    # img_show(class_ids, confidences, boxes, class_list, colors, frame)

    
