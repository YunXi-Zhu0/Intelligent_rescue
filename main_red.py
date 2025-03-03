import sys
import os
import threading
import time
import rclpy
import subprocess
from rclpy.node import Node
from geometry_msgs.msg import Twist
from x3pcodes.balltracking_red import *  # 包含 model_inference, max_area, target_choose, if_in_range, go_to_block_test2, etc.

sys.path.append('/userdata/create_files/x3pcodes')
from program_init.init_self import * #按钮启动程序
from Sustaining_inference_video import * #Socket网络流实时推理显示


# 实时推理(仅用于socket网络流显示, 不作为代码中视频流推理部分)
def inference_thread(camera_id, model_path, classes_name_path, cap):
    print("启动实时推理线程...")
    inference_model_bpu_camera(camera_id, model_path, classes_name_path, cap)

# 自主程序
def main_thread():
    print("启动自主运球程序线程...")
    main(model_path, classes_name_path, cap)


def main(model_path, classes_name_path, cap):
    # 初始化 ROS2 只执行一次，避免重复初始化
    rclpy.init()
    node = Node('centroid_align_and_go_to_block')
    publisher_ = node.create_publisher(Twist, '/cmd_vel', 10)

    # 程序自启动(True为按键按下, False为中断)
    isRun = program_init()
    # isRun = True

    # 开局斜向45°躲避
    turn_angle(publisher_, -90)
    move_distance(publisher_, 0.3, speed=0.2) 
    turn_angle(publisher_, 90)

    while isRun:
        judge_to_block = False  # 初始化为 False，表示小车可以移动, 走向物块
        judge_to_safety = False  # 初始化为 False，表示小车可以移动, 走向安全区

        # 设置舵机初始状态, 默认抬起
        pwm_command_str = str(2)  # 将命令值转换为字符串，并传递给 C++ 程序
        result = subprocess.run(['/userdata/create_files/step_motor/cycle50', pwm_command_str], capture_output=True,
                                text=True)  # 调用 C++ 程序，并传递命令值作为参数
        print(result.stdout)  # 打印 C++ 程序的输出

        # 走向物块
        while not judge_to_block:
            # 每秒拍摄一张图片，返回最新图片路径
            image_path = persecond_img_catch("/userdata/create_files/catch_img/img_save", True, cap)
            if not image_path:
                print("未拍摄到图片，退出程序。")
                break

            # 加载图像
            image = img_convert(image_path)
            if image is None:
                print("无法加载图像，退出程序。")
                break

            # 模型推理，获得目标检测结果
            class_ids, confidences, boxes, class_list, colors = model_inference(image, model_path, classes_name_path)

            #剔除安全区内部的物体
            class_ids, boxes, class_list, image = block_in_safety(class_ids, boxes, class_list, image)

            # 计算各颜色目标的最大面积和质心
            red_max, red_centroid, yellow_max, yellow_centroid, black_max, black_centroid, image = max_area(class_ids, boxes, image)

            print(black_max)
            print(black_centroid)

            # 选择最大面积目标，返回目标质心、最新图像、目标类别、目标边界框以及目标当前宽度
            _, max_centroid, image, class_ids, boxes, current_box_width = target_choose(red_max, red_centroid, yellow_max, yellow_centroid, black_max, black_centroid, image)

            # 判断目标是否进入目标区域
            red_in_range, yellow_in_range, black_in_range = if_in_range(image, red_centroid, yellow_centroid, black_centroid)
            judge_to_block = red_in_range or yellow_in_range or black_in_range

            # 调用 centroid_align 和 go_to_block_test2
            if max_centroid is not None:
                if centroid_align(class_ids, boxes, image, max_centroid, screen_width=640, fov=120, publisher_=publisher_):
                    go_to_block(False, publisher_)
            else:
                print("没有识别到物体，退出程序。")
                # turn_angle(publisher_, 100)
                flag = self_spin_block(cap, "/userdata/create_files/catch_img/img_save", model_path, classes_name_path, publisher_)
                if flag:
                    continue

        # **确保小车最终停止**
        isClosed_block = None  # 命令值设定(1表示转动到180°, 2表示转动到90°)
        if max_centroid is not None:
            isClosed_block = go_to_block(True, publisher_)  # 让小车停止

        # # 关闭 ROS2
        # node.destroy_node()  # 现在只有在 main 函数结束时销毁节点
        # rclpy.shutdown()

        print("目标已进入区域，准备框住物体...")

        # 设置舵机，框住目标物体
        pwm_command_str = str(isClosed_block)  # 将命令值转换为字符串，并传递给 C++ 程序
        result = subprocess.run(['/userdata/create_files/step_motor/cycle50', pwm_command_str], capture_output=True,
                                text=True)  # 调用 C++ 程序，并传递命令值作为参数
        print(result.stdout)  # 打印 C++ 程序的输出

        # start_time = time.time() #拾取物块的时间戳

        # 走向安全区
        start_time = None  # 用于记录找到安全区时的时间戳
        while not judge_to_safety:
            # 每秒拍摄一张图片，返回最新图片路径
            image_path = persecond_img_catch("/userdata/create_files/catch_img/img_save", True, cap)
            if not image_path:
                print("未拍摄到图片，退出程序。")
                break

            # 加载图像
            image = img_convert(image_path)
            if image is None:
                print("无法加载图像，退出程序。")
                break

            # 模型推理，获得目标检测结果
            class_ids, confidences, boxes, class_list, colors = model_inference(image, model_path, classes_name_path)

            # 获取蓝方安全区质心
            safety_red_centroid = get_safety_red(class_ids, boxes, class_list)

            # 调用 centroid_align 和 go_to_block
            if safety_red_centroid is not None:
                if centroid_align(class_ids, boxes, image, safety_red_centroid, screen_width=640, fov=120,
                                  publisher_=publisher_):
                    go_to_block(False, publisher_)
                    if start_time is None:  # 只有第一次找到安全区时，才记录时间
                        start_time = time.time()
            else:
                print("没有识别到安全区，退出程序。")
                # turn_angle(publisher_, 100)
                flag = self_spin_safety(cap, "/userdata/create_files/catch_img/img_save", model_path, classes_name_path, publisher_)
                if flag:
                    continue

            # 检查是否已经运行了 8 秒
            if start_time is not None:  # 只有在找到了安全区后才会计时
                elapsed_time = time.time() - start_time
                if elapsed_time > 8:  # 超过 8 秒后让小车停止
                    if safety_red_centroid is not None:
                        isClosed_block = go_to_block(True, publisher_)  # 让小车停止
                        print("小车已停止。")
                        break  # 退出循环，结束程序

        print("已到达安全区，准备释放物体...")
        # 设置舵机，释放目标物体
        pwm_command_str = str(2)  # 将命令值转换为字符串，并传递给 C++ 程序
        result = subprocess.run(['/userdata/create_files/step_motor/cycle50', pwm_command_str], capture_output=True,
                                text=True)  # 调用 C++ 程序，并传递命令值作为参数
        print(result.stdout)  # 打印 C++ 程序的输出

        # 退出安全区, 寻找下一个物体
        move_distance(publisher_, 0.2, -0.25)
        # turn_angle(publisher_, -300)
        self_spin_block(cap, "/userdata/create_files/catch_img/img_save", model_path, classes_name_path, publisher_)



if __name__ == "__main__":
    # 模型和类别文件路径
    model_path = "/userdata/create_files/x3pcodes/yolov5s_672x672_nv12_colorrec_red.bin"
    classes_name_path = "/userdata/create_files/x3pcodes/colorrec_red.names"

    # 摄像头打开
    # camera_id = 8
    # cap = cv2.VideoCapture(camera_id)
    cap = None
    
    main(model_path, classes_name_path, cap)
    
    #使用多线程
    # thread1 = threading.Thread(target=inference_thread, args=(camera_id, model_path, classes_name_path, cap))
    # thread2 = threading.Thread(target=main, args=(model_path, classes_name_path, cap))
    # thread1.start()
    # thread2.start()
    # thread1.join()
    # thread2.join()

    #全部程序完成, 关闭摄像头
    # cap.release()