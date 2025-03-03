import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard
import subprocess

#手柄按键映射
def main(args=None):
    rclpy.init(args=args)
    node = Node('padControl')
    publisher_ = node.create_publisher(Twist, '/cmd_vel', 10)
    twist = Twist()

    with open("/userdata/create_files/gamepad/Q_status.txt",'w') as f:
        f.write("False") #状态检测

    def on_press(key):
        nonlocal twist
        try:
            if key.char == 'w':
                twist.linear.x = 0.2  # 增大线速度
            elif key.char == 's':
                twist.linear.x = -0.2
            elif key.char == 'a':
                twist.angular.z = 1.0  # 增大角速度
            elif key.char == 'd':
                twist.angular.z = -1.0
            elif key.char == 'n':
                twist.linear.x = 0.3
            elif key.char == 'j':
                pwm_command_str = str(2)  # 将命令值转换为字符串，并传递给 C++ 程序
                result = subprocess.run(['/userdata/create_files/step_motor/cycle30', pwm_command_str], capture_output=True,text=True)  # 调用 C++ 程序，并传递命令值作为参数
                print(result.stdout)  # 打印 C++ 程序的输出
            elif key.char == 'k':
                pwm_command_str = str(1)  # 将命令值转换为字符串，并传递给 C++ 程序
                result = subprocess.run(['/userdata/create_files/step_motor/cycle30', pwm_command_str], capture_output=True,text=True)  # 调用 C++ 程序，并传递命令值作为参数
                print(result.stdout)  # 打印 C++ 程序的输出
            elif key.char == 'q':
                print("Q ok")
                with open("/userdata/create_files/gamepad/Q_status.txt",'w') as f:
                    f.write("True") #状态更新
        except AttributeError:
            pass

    def on_release(key):
        nonlocal twist
        try:
            if key.char in ('w', 's'):
                twist.linear.x = 0.0
            elif key.char in ('a', 'd'):
                twist.angular.z = 0.0
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    timer = node.create_timer(0.1, lambda: timer_callback(publisher_, twist, node))
    rclpy.spin(node)

    listener.stop()
    node.destroy_node()
    rclpy.shutdown()

def timer_callback(publisher_, twist, node):
    publisher_.publish(twist)
    # node.get_logger().info(f"Linear: {twist.linear.x}, Angular: {twist.angular.z}")

if __name__ == '__main__':
    main()
