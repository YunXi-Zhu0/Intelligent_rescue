import sys
import os
import rclpy
import subprocess
import time  # 导入时间模块
from rclpy.node import Node
from geometry_msgs.msg import Twist
from x3pcodes.balltracking import *       # 包含 model_inference, max_area, target_choose, if_in_range, go_to_block_test2, etc.
from start_line45.start_line45 import *   # 其他依赖模块
sys.path.append('/userdata/create_files/x3pcodes')

if __name__ == "__main__":
    pwm_command_str = str(1) # 将命令值转换为字符串，并传递给 C++ 程序
    result = subprocess.run(['/userdata/create_files/step_motor/cycle50', pwm_command_str], capture_output=True, text=True) # 调用 C++ 程序，并传递命令值作为参数
    print(result.stdout) # 打印 C++ 程序的输出