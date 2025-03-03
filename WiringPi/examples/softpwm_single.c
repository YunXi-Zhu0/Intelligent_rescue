/*-----------------------------------------------------------------------------
 Function   : Used for testing RaspberryPi ZeroW's gpio pin -> Relay 
 FileName   : TestRelay.c
 Author     : Tomato
 Time       : 2019.2.4
 Version    : 0.1
 Notes      : None
-----------------------------------------------------------------------------*/
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