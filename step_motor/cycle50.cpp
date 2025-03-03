#include <wiringPi.h>
#include <softPwm.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define pwm_pin 16  // 假设使用GPIO 16引脚

// 获取当前时间的毫秒数
unsigned long getmillis() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("错误：缺少命令行参数。\n");
        return 1;
    }

    // 获取命令行参数并将其转换为整数
    int command = atoi(argv[1]);

    wiringPiSetup();                 // 初始化wiringPi库
    softPwmCreate(pwm_pin, 0, 100);  // 创建软件PWM对象，设置占空比范围为0-100

    int pwm_value = 11;  // 默认占空比为11（对应90°）

    // 占空比11抬起, 7落下, 30°
    if (command == 1) 
    {
        pwm_value = 11;  // 设置占空比为13（对应180°）
        printf("占空比为13, 转动到0°\n");
    } 
    else if (command == 2)
    {
        pwm_value = 18;  // 设置占空比为18（对应30°）
        printf("占空比为18, 转动到30°\n");
    }
    else
    {
        printf("占空比值设置错误\n");
        return 1;
    }

    // 设置占空比
    softPwmWrite(pwm_pin, pwm_value);

    // 获取当前时间
    unsigned long start_time = getmillis();
    unsigned long delay_time = 2500;  // 等待 2500 毫秒

    // 模拟阻塞过程
    while (getmillis() - start_time < delay_time) {
        // 程序不会在此停止，继续循环等待直到到达指定的延迟时间
    }

    printf("程序已完成，旋转结束。\n");

    return 0;
}