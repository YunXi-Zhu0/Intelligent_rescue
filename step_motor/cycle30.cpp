#include <wiringPi.h>
#include <softPwm.h>
#include <stdio.h>
#include <stdlib.h>

#define pwm_pin 16  // 假设使用GPIO 16引脚

int main(int argc, char **argv)
{
    // 获取命令行参数并将其转换为整数
    int command = atoi(argv[1]);
    // printf("%d", command);

    wiringPiSetup();                 // 初始化wiringPi库
    softPwmCreate(pwm_pin, 0, 100);  // 创建软件PWM对象，设置占空比范围为0-100

    int pwm_value = 11;  // 默认占空比为15（对应90°）

    //占空比11抬起, 7落下, 30°
    if (command == 1) 
    {
        pwm_value = 11;  // 设置占空比为25（对应180°）
        printf("占空比为13, 转动到0°\n");
    } 
    else if (command == 2)
    {
        pwm_value = 18;  // 设置占空比为15（对应90°）
        printf("占空比为18, 转动到30°\n");
    }
    else
    {
        printf("占空比值设置错误\n");
        return 1;
    }

    // 设置占空比
    softPwmWrite(pwm_pin, pwm_value);
    
    delay(1000);  // 等待 1 秒钟，确保电机有足够的时间完成旋转

    return 0;
}





