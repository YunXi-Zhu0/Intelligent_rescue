# 旭日X3派WiringPi

## 安装WiringPi

安装git工具，执行以下命令：

```bash
sudo apt update
sudo apt install git-core
```

通过git在线获取WiringPi的源代码，执行以下命令：

```bash
git clone https://gitee.com/study-dp/WiringPi.git
```

进入WiringPi目录安装WiringPi。执行以下命令：

```bash
cd WiringPi
./build
```

build.sh 脚本会自动完成WiringPi库的编译与安装。

## 查看命令帮助信息

命令： `gpio -h`

```
root@ubuntu:~# gpio 
gpio: At your service!
  Type: gpio -h for full details and
        gpio readall for a quick printout of your connector details

root@ubuntu:~# gpio -h 
gpio: Usage: gpio -v
       gpio -h
       gpio [-g|-1] ...
       gpio [-d] ...
       [-x extension:params] [[ -x ...]] ...
       gpio [-p] <read/write/wb> ...
       gpio <mode/read/write/aread/awritewb/pwm/pwmTone> ...
       gpio <toggle/blink> <pin>
       gpio readall
       gpio unexportall/exports
       gpio export/edge/unexport ...
       gpio wfi <pin> <mode>
       gpio drive <group> <value>
       gpio pwmf <pin> <frequency> 
       gpio pwmd <pin> <duty_cycle> 
       gpio i2cd/i2cdetect
       gpio rbx/rbd
       gpio wb <value>
       gpio gbr <channel>
       gpio gbw <channel> <value>
```

**主要命令说明：**

readall：读取所有管脚的信息，显示管脚名称和`Physical`,`BCM`、`xPi`三种模式的管脚编号

-g： 命令中的pin参数默认使用 `BCM`序号，如果设置本参数，则使用旭日X3的gpio内部编号（ `xPi`序号），各模式的编号详情请通过 `gpio readall`获取

-l：  通过 `/sys/class/gpio`目录下的文件进行管脚操作，否则都是直接通过操作旭日X3的寄存器操作

-p：使用硬件 `Physical` 编号作为pin的编码模式

mode： 设置管脚的模式，支持以下模式的设置

​		`in`,`input`： 设置管脚为gpio输入

​		`out`,`output`：设置管脚为gpio输出

​		`pwm`,`pwmTone`： 设置管脚为pwm模式

​		`up`： 设置管脚上拉

​		`down`：设置管脚下拉

​		`tri`,`off`： 管脚关闭上下拉

read:  读取管脚的输入电平

write:  设置管脚的输出电平

toggle：使管脚的输出电平反转

blink： 使管脚进入到闪烁状态

unexportall：通过 `/sys/class/gpio`目录下的文件释放所有管脚

exports：显示通过 `/sys/class/gpio`申请的管脚列表

export/edge/unexport：通过通过 `/sys/class/gpio`对单个管脚进行申请和释放，设置触发模式

wfi：管脚设置为输入模式，并且设置中断响应模式

drive： 设置管脚的电流驱动强度

pwmf： 设置pwm的时钟周期，输入值为频率，例如48000表示48KHz, 取值范围最小48000

pwmd：设置pwm的高电平占空比，取值范围1-100

i2cd/i2cdetect： 显示I2C总线上的设备信息

rbx/rbd： 读取0-7这一组管脚的电平值

## 查看管脚信息

命令： `gpio readall`

```
root@ubuntu:~# gpio readall
 +-----+-----+-----------+---X3 PI--+-----------+-----+-----+
 | BCM | xPi |    Name   | Physical |   Name    | xPi | BCM |
 +-----+-----+-----------+----++----+-----------+-----+-----+
 |     |     |      3.3v |  1 || 2  | 5v        |     |     |
 |   2 |   9 |     SDA.0 |  3 || 4  | 5v        |     |     |
 |   3 |   8 |     SCL.0 |  5 || 6  | 0v        |     |     |
 |   4 | 101 | I2S0_MCLK |  7 || 8  | TxD.3     | 111 | 14  |
 |     |     |        0v |  9 || 10 | RxD.3     | 112 | 15  |
 |  17 |   6 |   GPIO. 6 | 11 || 12 | I2S0_BCLK | 102 | 18  |
 |  27 |   5 |   GPIO. 5 | 13 || 14 | 0v        |     |     |
 |  22 |  30 |  GPIO. 30 | 15 || 16 | GPIO. 27  | 27  | 23  |
 |     |     |      3.3v | 17 || 18 | GPIO. 7   | 7   | 24  |
 |  10 |  12 | SPI2_MOSI | 19 || 20 | 0v        |     |     |
 |   9 |  13 | SPI2_MISO | 21 || 22 | GPIO. 29  | 29  | 25  |
 |  11 |  14 | SPI2_SCLK | 23 || 24 | SPI2_CSN  | 15  | 8   |
 |     |     |        0v | 25 || 26 | GPIO. 28  | 28  | 7   |
 |   0 | 106 | I2S1_BCLK | 27 || 28 | I2S1_LRCK | 107 | 1   |
 |   5 | 119 |  GPIO.119 | 29 || 30 | 0v        |     |     |
 |   6 | 118 |  GPIO.118 | 31 || 32 | PWM4      | 25  | 12  |
 |  13 |   4 |      PWM0 | 33 || 34 | 0v        |     |     |
 |  19 | 103 | I2S0_LRCK | 35 || 36 | GPIO. 3   | 3   | 16  |
 |  26 | 105 |  GPIO.105 | 37 || 38 | I2S0_SDIO | 104 | 20  |
 |     |     |        0v | 39 || 40 | I2S1_SDIO | 108 | 21  |
 +-----+-----+-----------+----++----+-----------+-----+-----+
 | BCM | xPi |    Name   | Physical |   Name    | xPi | BCM |
 +-----+-----+-----------+---X3 PI--+-----------+-----+-----+
```

**字段说明：**

Physical：硬件管脚序号

Name：管脚名称

xPi：旭日X3芯片内部管脚

BCM: 通用编码

管脚信息可以查看 [40PIN管脚定义](#40pin_define) 。

## 操作管脚输入输出

读取所有管脚当前的电平状态：

```
sudo gpio allreadall
```

设置为输入模式：

```
sudo gpio mode 26 in
```

读取管脚电平值：

```
sudo gpio read 26
```

设置为输出模式：

```
sudo gpio mode 26 out
```

设置管脚输出高电平：

```
sudo gpio write 26 on
```

或者

```
sudo gpio write 26 up
```

设置管脚输出低电平：

```
sudo gpio write 26 off
```

或者

```
sudo gpio write 26 down
```

响应管脚的上升沿中断：

```
sudo gpio wfi 26 rising
```

响应管脚的下降沿中断：

```
sudo gpio wfi 26 falling
```

同时响应管脚的上升、下降沿中断：

```
sudo gpio wfi 26 both
```

## 硬件PWM功能使用

旭日X3派只有两个硬件PWM管脚在 40pin上，分别是 BCM 编码的 12 和 13 号（硬件编码的 32 和 33管脚，对应旭日X3芯片的 25 和 4号管脚）。

硬件PWM的时钟源为192MHz, 分频系数为 1-4000， 所以输出时钟周期为： 48KHz - 192MHz

设置管脚为pwm模式，设置成功后，管脚默认输出48KHz，占空比为50%的脉冲信号：

```
sudo gpio mode 12 pwm
```

或者

```
sudo gpio mode 12 pwmTone
```

设置pwm的时钟周期，取值范围48000 ~ 192000000（建议不要超过 12MHz）:

```
sudo gpio pwmf 12 48000
```

设置pwm的占空比，按高电平的百分比设置，取值范围 0 ~ 100：

```
sudo gpio pwmd 12 50
```

## 软件PWM功能使用

旭日X3派40PIN的所有功能管脚都可以通过CPU控制输出高、低电平来模拟PWM。

以下是使用软件PWM的代码逻辑：

设置时钟周期为50Hz，占空比时间分别为循环设置为 2ms、5ms、9ms。

```C
#include <stdio.h>
#include <wiringPi.h>
#include <softPwm.h>

/* Macro config define */
#define PWM_1     25U

/* Parameter define */
static int high_edge_time = 0;   // uint :0.1ms
static int cycle_time     = 200; // uint :0.1ms

int main (void)
{
    /* Initial gpio with chip pin ,if use mapping pin -> wiringPiSetup*/
    wiringPiSetupGpio () ;

    // pinMode (PWM_1, SOFT_PWM_OUTPUT) ;
    printf("cycle_time:%d\n", cycle_time);
    pinMode(PWM_1, OUTPUT);
    softPwmCreate(PWM_1, high_edge_time, cycle_time);

    /* Main loop function */
    for (;;)
    {
        /* Control pwm */
        softPwmWrite(PWM_1, 20); // duty = 20/cycle_time
        delay (5000);
        softPwmWrite(PWM_1, 50);
        delay (5000);
        softPwmWrite(PWM_1, 90);
        delay (5000);
    }
    return 0 ;
}
```

