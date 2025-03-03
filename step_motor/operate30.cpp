#include <wiringPi.h>
#include <softPwm.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <time.h>

#define pwm_pin 16  // 假设使用GPIO 16引脚

// 设置终端为无需回车模式
void set_terminal_mode(int enable) {
    static struct termios oldt, newt;
    if (enable) {
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);  // 禁用规范模式和回显
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    } else {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // 恢复原始模式
    }
}

// 获取键盘输入（无需回车）
char getch() {
    char c;
    if (read(STDIN_FILENO, &c, 1) < 0) {
        perror("read()");
    }
    return c;
}

unsigned long getMillis() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int main(int argc, char **argv)
{
    // 初始化wiringPi库
    wiringPiSetup();
    softPwmCreate(pwm_pin, 0, 100);  // 创建软件PWM对象，设置占空比范围为0-100

    int current_pwm_value = 18;  // 假设当前占空比为11.5
    int pwm_change = 7;            // 30度对应的占空比变化

    char last_key = '\0';  // 记录上一次按下的键
    int first_key_pressed = 0;  // 标志是否已经按过第一次键

    unsigned long last_time = 0;  // 记录上次更新时间
    unsigned long delay_time = 2000;  // 设置延迟时间（毫秒）

    // 设置终端为无需回车模式
    set_terminal_mode(1);

    printf("舵机控制程序已启动。\n");
    printf("按下 'c' 键：逆时针旋转30度\n");
    printf("按下 'e' 键：顺时针旋转30度\n");
    printf("按下 'q' 键：退出程序\n");
    printf("注意：第一次按键必须是 'c'。\n");

    while (1) {
        char key = getch();  // 获取键盘输入（无需回车）

        if (key == 'c') {
            if (!first_key_pressed) {
                // 第一次按键必须是 'c'
                first_key_pressed = 1;
                printf("第一次按键已记录为 'c'。\n");
            }
            if (last_key != 'c') {
                // 逆时针转30度
                current_pwm_value -= pwm_change;  // 减少占空比
                softPwmWrite(pwm_pin, current_pwm_value);
                printf("逆时针旋转30度完成，当前占空比：%.1f\n", (float)current_pwm_value);
                last_key = 'c';  // 记录上一次按下的键
                last_time = getMillis();  // 记录当前时间
            } else {
                printf("提示：上一次已经按过 'c'，本次无效。\n");
            }
        } else if (key == 'e') {
            if (!first_key_pressed) {
                // 第一次按键不能是 'e'
                printf("错误：第一次按键必须是 'c'。\n");
            } else if (last_key != 'e') {
                // 顺时针转30度
                current_pwm_value += pwm_change;  // 增加占空比
                softPwmWrite(pwm_pin, current_pwm_value);
                printf("顺时针旋转30度完成，当前占空比：%.1f\n", (float)current_pwm_value);
                last_key = 'e';  // 记录上一次按下的键
            } else {
                printf("提示：上一次已经按过 'e'，本次无效。\n");
            }
        } else if (key == 'q') {
            // 退出程序
            if (last_key == 'c') {
                // 如果最后一次按下的是 'c'，则先顺时针旋转30度再退出
                current_pwm_value += pwm_change;  // 增加占空比
                softPwmWrite(pwm_pin, current_pwm_value);
                printf("顺时针旋转30度完成，当前占空比：%.1f\n", (float)current_pwm_value);
                delay(3000);  // 等待3秒，确保舵机完成旋转
            }
            printf("程序退出。\n");
            break;
        } else {
            // 无效输入
            printf("无效按键，请按 'c'、'e' 或 'q'。\n");
        }

        // 检查是否到达延迟时间
        if (getMillis() - last_time >= delay_time) {
            // 如果到达延迟时间，继续执行
            last_time = getMillis();  // 重置时间
        }
    }

    // 恢复终端模式
    set_terminal_mode(0);

    return 0;
}
