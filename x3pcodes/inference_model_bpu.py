import numpy as np
import cv2
import os
from hobot_dnn import pyeasy_dnn as dnn
from bputools.format_convert import imequalresize, bgr2nv12_opencv
import sys
import lib.pyyolotools as yolotools


def inference_model_bpu_camera(camera_id, model_path, classes_name_path):
    """
    用于摄像头推理并实时显示结果
    :param camera_id: 摄像头ID，通常是0、1等
    :param model_path: 模型路径
    :param classes_name_path: 数据集标注文件路径
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

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        sys.exit()

    # 加载类别名
    with open(classes_name_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

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

        # 绘制检测框
        if class_ids is not None and confidences is not None and boxes is not None:
            for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                color = colors[int(classid) % len(colors)]
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, f"{class_list[classid]} {confidence:.2f}", 
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # 显示推理结果
        cv2.imshow("Detection Result", frame)

        # 按键控制退出（例如按 'q' 键退出）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 示例调用
camera_id = 8  # 替换为实际的摄像头ID
model_path = "/userdata/create_files/x3pcodes/yolov5s_672x672_nv12_colorrec_blue.bin"  
classes_name_path = "/userdata/create_files/x3pcodes/colorrec_blue.names" 

inference_model_bpu_camera(camera_id, model_path, classes_name_path)
