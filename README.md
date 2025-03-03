# Intelligent_rescue
A repository used to finish the project about Intelligent rescue competition in 2025


主要文件(比赛时自主程序+遥控运动):
1.x3pcodes: balltracking_blue.py编写被调函数; inference_model_bpu_test.py用于测试yolo是否部署成功

2.step_motor: 舵机控制, 头框抬起和下落

3.program_init: 按键一键启动

4.gamepad: 使用ros2控制电机(线速度和角速度), 配合steam手柄按键映射, 可以做到遥控小车

5.catch_img: img_save文件夹用于balltracking_blue.py中persecond_img_catch函数拍摄的照片存放


次要文件(1和2为部分环境, 也可到官网进行下载)
1.toPad_control: python的venv环境, 用于gamepad环境配置

2.WiringPi: 用于舵机控制的库, step_motor中文件编译所需

3.Sustaining_inference_video.py调用服务端socket网络流(Socket.py文件), 显示实时推理结果; gradio_page.py用于本地端接收回传信息

