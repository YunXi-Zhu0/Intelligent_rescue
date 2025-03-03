#!/usr/bin/env python3

import Hobot.GPIO as GPIO
import time

# 定义全局变量flag
flag = False

#传回布尔值, True为按键按下, False为中断; 用于低电平开关按钮一键启动
def program_init():
    global flag  # 使用全局变量

    # 设置GPIO模式为BOARD模式
    GPIO.setmode(GPIO.BOARD)

    # 设置按钮的GPIO引脚号
    button_pin = 24  # 你可以根据实际连接的引脚修改这个数字

    # 设置按钮引脚为输入，并启用内部上拉电阻（初始为高电平）
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.RISING)  # 上拉电阻

    # 定义按钮按下的回调函数
    def button_pressed_callback(channel):
        global flag  # 通过global声明使用全局变量
        flag = True  # 按钮按下时将flag设置为True

    # 设置事件检测，按下按钮时触发回调函数
    GPIO.add_event_detect(button_pin, GPIO.FALLING, callback=button_pressed_callback, bouncetime=300)

    try:
        # 保持程序运行，等待按钮事件触发
        print("请按下按钮，一键启动程序！")
        while True:
            time.sleep(0.1)
            if flag:  # 如果按钮按下，flag为True
                return True

    except KeyboardInterrupt:
        print("程序被中断")

    finally:
        # 清理GPIO设置
        GPIO.cleanup()

if __name__ == "__main__":
    if program_init():
        print("按钮按下，程序已启动！")
    else:
        print("程序未启动！")
