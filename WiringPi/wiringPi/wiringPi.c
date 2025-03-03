/*
 * wiringPi:
 *	Arduino look-a-like Wiring library for the Raspberry Pi
 *	Copyright (c) 2012-2017 Gordon Henderson
 *	Additional code for pwmSetClock by Chris Hall <chris@kchall.plus.com>
 *
 *	Thanks to code samples from Gert Jan van Loo and the
 *	BCM2835 ARM Peripherals manual, however it's missing
 *	the clock section /grr/mutter/
 ***********************************************************************
 * This file is part of wiringPi:
 *    https://github.com/WiringPi/WiringPi
 *
 *    wiringPi is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as
 *    published by the Free Software Foundation, either version 3 of the
 *    License, or (at your option) any later version.
 *
 *    wiringPi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with wiringPi.
 *    If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************
 */

// Revisions:
//	19 Jul 2012:
//		Moved to the LGPL
//		Added an abstraction layer to the main routines to save a tiny
//		bit of run-time and make the clode a little cleaner (if a little
//		larger)
//		Added waitForInterrupt code
//		Added piHiPri code
//
//	 9 Jul 2012:
//		Added in support to use the /sys/class/gpio interface.
//	 2 Jul 2012:
//		Fixed a few more bugs to do with range-checking when in GPIO mode.
//	11 Jun 2012:
//		Fixed some typos.
//		Added c++ support for the .h file
//		Added a new function to allow for using my "pin" numbers, or native
//			GPIO pin numbers.
//		Removed my busy-loop delay and replaced it with a call to delayMicroseconds
//
//	02 May 2012:
//		Added in the 2 UART pins
//		Change maxPins to numPins to more accurately reflect purpose


#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <poll.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <asm/ioctl.h>

#include "softPwm.h"
#include "softTone.h"

#include "wiringPi.h"
#include "../version.h"

// Environment Variables

#define	ENV_DEBUG	"WIRINGPI_DEBUG"
#define	ENV_CODES	"WIRINGPI_CODES"
#define	ENV_GPIOMEM	"WIRINGPI_GPIOMEM"


// Extend wiringPi with other pin-based devices and keep track of
//	them in this structure

struct wiringPiNodeStruct *wiringPiNodes = NULL ;

// BCM Magic

#define	BCM_PASSWORD		0x5A000000


// The BCM2835 has 54 GPIO pins.
//	BCM2835 data sheet, Page 90 onwards.
//	There are 6 control registers, each control the functions of a block
//	of 10 pins.
//	Each control register has 10 sets of 3 bits per GPIO pin - the ALT values
//
//	000 = GPIO Pin X is an input
//	001 = GPIO Pin X is an output
//	100 = GPIO Pin X takes alternate function 0
//	101 = GPIO Pin X takes alternate function 1
//	110 = GPIO Pin X takes alternate function 2
//	111 = GPIO Pin X takes alternate function 3
//	011 = GPIO Pin X takes alternate function 4
//	010 = GPIO Pin X takes alternate function 5
//
// So the 3 bits for port X are:
//	X / 10 + ((X % 10) * 3)

// Port function select bits

#define	FSEL_INPT		0b000
#define	FSEL_OUTP		0b001
#define	FSEL_ALT0		0b100
#define	FSEL_ALT1		0b101
#define	FSEL_ALT2		0b110
#define	FSEL_ALT3		0b111
#define	FSEL_ALT4		0b011
#define	FSEL_ALT5		0b010

// Access from ARM Running Linux
//	Taken from Gert/Doms code. Some of this is not in the manual
//	that I can find )-:
//
// Updates in September 2015 - all now static variables (and apologies for the caps)
//	due to the Pi v2, v3, etc. and the new /dev/gpiomem interface

static volatile unsigned int GPIO_PADS ;
static volatile unsigned int GPIO_CLOCK_BASE ;
static volatile unsigned int GPIO_BASE ;
static volatile unsigned int GPIO_PIN_MUX_BASE ;
static volatile unsigned int GPIO_TIMER ;
static volatile unsigned int GPIO_PWM ;

#define	PAGE_SIZE		(4*1024)
#define	BLOCK_SIZE		(4*1024)

static unsigned int usingGpioMem    = FALSE ;
static          int wiringPiSetuped = FALSE ;

// PWM
//	Word offsets into the PWM control region

#define	PWM_CONTROL 0
#define	PWM_STATUS  1
#define	PWM0_RANGE  4
#define	PWM0_DATA   5
#define	PWM1_RANGE  8
#define	PWM1_DATA   9

//	Clock regsiter offsets

#define	PWMCLK_CNTL	40
#define	PWMCLK_DIV	41

#define	PWM0_MS_MODE    0x0080  // Run in MS mode
#define	PWM0_USEFIFO    0x0020  // Data from FIFO
#define	PWM0_REVPOLAR   0x0010  // Reverse polarity
#define	PWM0_OFFSTATE   0x0008  // Ouput Off state
#define	PWM0_REPEATFF   0x0004  // Repeat last value if FIFO empty
#define	PWM0_SERIAL     0x0002  // Run in serial mode
#define	PWM0_ENABLE     0x0001  // Channel Enable

#define	PWM1_MS_MODE    0x8000  // Run in MS mode
#define	PWM1_USEFIFO    0x2000  // Data from FIFO
#define	PWM1_REVPOLAR   0x1000  // Reverse polarity
#define	PWM1_OFFSTATE   0x0800  // Ouput Off state
#define	PWM1_REPEATFF   0x0400  // Repeat last value if FIFO empty
#define	PWM1_SERIAL     0x0200  // Run in serial mode
#define	PWM1_ENABLE     0x0100  // Channel Enable

// Timer
//	Word offsets

#define	TIMER_LOAD	(0x400 >> 2)
#define	TIMER_VALUE	(0x404 >> 2)
#define	TIMER_CONTROL	(0x408 >> 2)
#define	TIMER_IRQ_CLR	(0x40C >> 2)
#define	TIMER_IRQ_RAW	(0x410 >> 2)
#define	TIMER_IRQ_MASK	(0x414 >> 2)
#define	TIMER_RELOAD	(0x418 >> 2)
#define	TIMER_PRE_DIV	(0x41C >> 2)
#define	TIMER_COUNTER	(0x420 >> 2)

// Locals to hold pointers to the hardware

static volatile unsigned int *gpio ;
static volatile unsigned int *pwm ;
static volatile unsigned int *clk ;
static volatile unsigned int *pads ;
static volatile unsigned int *timer ;
static volatile unsigned int *timerIrqRaw ;

// Export variables for the hardware pointers

volatile unsigned int *_wiringPiGpio ;
volatile unsigned int *_wiringPiGpioPinMux ;
volatile unsigned int *_wiringPiPwm ;
volatile unsigned int *_wiringPiClk ;
volatile unsigned int *_wiringPiPads ;
volatile unsigned int *_wiringPiTimer ;
volatile unsigned int *_wiringPiTimerIrqRaw ;


// Data for use with the boardId functions.
//	The order of entries here to correspond with the PI_MODEL_X
//	and PI_VERSION_X defines in wiringPi.h
//	Only intended for the gpio command - use at your own risk!

// piGpioBase:
//	The base address of the GPIO memory mapped hardware IO

#define	GPIO_PERI_BASE_OLD  0x20000000
#define	GPIO_PERI_BASE_2835 0x3F000000
#define	GPIO_PERI_BASE_2711 0xFE000000

static volatile unsigned int piGpioBase = 0 ;
static volatile unsigned int piGpioPinMuxBase = 0 ;

const char *piModelNames [26] =
{
  "Model A",	//  0
  "Model B",	//  1
  "Model A+",	//  2
  "Model B+",	//  3
  "Pi 2",	//  4
  "Alpha",	//  5
  "CM",		//  6
  "Unknown07",	// 07
  "Pi 3",	// 08
  "Pi Zero",	// 09
  "CM3",	// 10
  "Unknown11",	// 11
  "Pi Zero-W",	// 12
  "Pi 3B+",	// 13
  "Pi 3A+",	// 14
  "Unknown15",	// 15
  "CM3+",	// 16
  "Pi 4B",	// 17
  "Pi Zero2-W",	// 18
  "Pi 400",	// 19
  "CM4",	// 20
  "X3 SDB", // 21
  "RDK X3", // 22
  "RDK X3 V1.2", // 23
  "RDK X3 V2", // 24
  "RDK X3 Module", // 25
} ;

const char *piRevisionNames [16] =
{
  "00",
  "01",
  "02",
  "03",
  "04",
  "05",
  "06",
  "07",
  "08",
  "09",
  "10",
  "11",
  "12",
  "13",
  "14",
  "15",
} ;

const char *piMakerNames [16] =
{
  "Sony",	//	 0
  "Egoman",	//	 1
  "Embest",	//	 2
  "Horizon",	//	 3
  "Embest",	//	 4
  "Unknown05",	//	 5
  "Unknown06",	//	 6
  "Unknown07",	//	 7
  "Unknown08",	//	 8
  "Unknown09",	//	 9
  "Unknown10",	//	10
  "Unknown11",	//	11
  "Unknown12",	//	12
  "Unknown13",	//	13
  "Unknown14",	//	14
  "Unknown15",	//	15
} ;

const int piMemorySize [8] =
{
   256,		//	 0
   512,		//	 1
  1024,		//	 2
  2048,		//	 3
  4096,		//	 4
  8192,		//	 5
     0,		//	 6
     0,		//	 7
} ;

// Time for easy calculations

static uint64_t epochMilli, epochMicro ;

// Misc

static int wiringPiMode = WPI_MODE_UNINITIALISED ;
static volatile int    pinPass = -1 ;
static pthread_mutex_t pinMutex ;

// Debugging & Return codes

int wiringPiDebug       = FALSE ;
int wiringPiReturnCodes = FALSE ;

// Use /dev/gpiomem ?

int wiringPiTryGpioMem  = FALSE ;

// sysFds:
//	Map a file descriptor from the /sys/class/gpio/gpioX/value

static int sysFds [64] =
{
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
} ;

// ISR Data

static void (*isrFunctions [64])(void) ;


// Doing it the Arduino way with lookup tables...
//	Yes, it's probably more innefficient than all the bit-twidling, but it
//	does tend to make it all a bit clearer. At least to me!

// pinToGpio:
//	Take a Wiring pin (0 through X) and re-map it to the BCM_GPIO pin
//	Cope for 3 different board revisions here.

static int *pinToGpio ;

// Revision 1, 1.1:

static int pinToGpioR1 [64] =
{
  17, 18, 21, 22, 23, 24, 25, 4,	// From the Original Wiki - GPIO 0 through 7:	wpi  0 -  7
   0,  1,				// I2C  - SDA1, SCL1				wpi  8 -  9
   8,  7,				// SPI  - CE1, CE0				wpi 10 - 11
  10,  9, 11, 				// SPI  - MOSI, MISO, SCLK			wpi 12 - 14
  14, 15,				// UART - Tx, Rx				wpi 15 - 16

// Padding:

      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 31
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 47
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 63
} ;

// Revision 2:

static int pinToGpioR2 [64] =
{
  17, 18, 27, 22, 23, 24, 25, 4,	// From the Original Wiki - GPIO 0 through 7:	wpi  0 -  7
   2,  3,				// I2C  - SDA0, SCL0				wpi  8 -  9
   8,  7,				// SPI  - CE1, CE0				wpi 10 - 11
  10,  9, 11, 				// SPI  - MOSI, MISO, SCLK			wpi 12 - 14
  14, 15,				// UART - Tx, Rx				wpi 15 - 16
  28, 29, 30, 31,			// Rev 2: New GPIOs 8 though 11			wpi 17 - 20
   5,  6, 13, 19, 26,			// B+						wpi 21, 22, 23, 24, 25
  12, 16, 20, 21,			// B+						wpi 26, 27, 28, 29
   0,  1,				// B+						wpi 30, 31

// Padding:

  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 47
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 63
} ;


// RDK X3
static int pinToGpioRdkX3 [64] =
{
  106, 107, // 0, 1
  9, 8,
  101, 119,
  118, 28,
   15, 13,
   12, 14, // 10, 11
  25, 4,
  111, 112,
  3, 6,
  102, 103,
  104, 108,  // 20, 21
  30, 27,
  7, 29,
  105,
  5,

// Padding
  -1, -1, -1, -1,

// Padding:

  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 47
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 63
} ;

// RDK X3 V2
static int pinToGpioRdkX3V2 [64] =
{
  15, 14, // 0, 1
  9, 8,
  101, 119,
  118, 28,
   5, 7,
   6, 3, // 10, 11
  25, 4,
  111, 112,
  20, 12,
  102, 103,
  108, 104,  // 20, 21
  30, 27,
  22, 29,
  117,  13,

// Padding
  -1, -1, -1, -1,

// Padding:

  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 47
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 63
} ;


// physToGpio:
//	Take a physical pin (1 through 26) and re-map it to the BCM_GPIO pin
//	Cope for 2 different board revisions here.
//	Also add in the P5 connector, so the P5 pins are 3,4,5,6, so 53,54,55,56

static int *physToGpio ;

static int physToGpioR1 [64] =
{
  -1,		// 0
  -1, -1,	// 1, 2
   0, -1,
   1, -1,
   4, 14,
  -1, 15,
  17, 18,
  21, -1,
  22, 23,
  -1, 24,
  10, -1,
   9, 25,
  11,  8,
  -1,  7,	// 25, 26

                                              -1, -1, -1, -1, -1,	// ... 31
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 47
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,	// ... 63
} ;

static int physToGpioR2 [64] =
{
  -1,		// 0
  -1, -1,	// 1, 2
   2, -1,
   3, -1,
   4, 14,
  -1, 15,
  17, 18,
  27, -1,
  22, 23,
  -1, 24,
  10, -1,
   9, 25,
  11,  8,
  -1,  7,	// 25, 26

// B+

   0,  1,
   5, -1,
   6, 12,
  13, -1,
  19, 16,
  26, 20,
  -1, 21,

// the P5 connector on the Rev 2 boards:

  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  28, 29,
  30, 31,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
} ;

static int physToGpioRdkX3 [64] =
{
  -1,		// 0
  -1, -1,	// 1, 2
   9, -1,
   8, -1,
   101, 111,
  -1, 112,
  6, 102,
  5, -1,
  30, 27,
  -1, 7,
  12, -1,
  13, 29,
  14, 15,
  -1, 28,	// 25, 26

// B+

   106,  107,
   119, -1,
   118, 25,
  4, -1,
  103, 3,
  105, 104,
  -1, 108,

// the P5 connector on the Rev 2 boards:

  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
} ;

static int physToGpioRdkX3V2 [64] =
{
  -1,		// 0
  -1, -1,	// 1, 2
   9, -1,
   8, -1,
   101, 111,
  -1, 112,
  12, 102,
  13, -1,
  30, 27,
  -1, 22,
  6, -1,
  7, 29,
  3, 5,
  -1, 28,	// 25, 26

   15,  14,
   119, -1,
   118, 25,
  4, -1,
  103, 20,
  117, 108,
  -1, 104,

// the P5 connector on the Rev 2 boards:

  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
} ;

static int physToPin [64] =
{
  -1,		// 0
  -1, -1,	// 1, 2
   2, -1,
   3, -1,
   4, 14,
  -1, 15,
  17, 18,
  27, -1,
  22, 23,
  -1, 24,
  10, -1,
   9, 25,
  11,  8,
  -1,  7,	// 25, 26

// B+

   0,  1,
   5, -1,
   6, 12,
  13, -1,
  19, 16,
  26, 20,
  -1, 21,

// the P5 connector on the Rev 2 boards:

  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  28, 29,
  30, 31,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
} ;

// gpioToGPFSEL:
//	Map a BCM_GPIO pin to it's Function Selection
//	control port. (GPFSEL 0-5)
//	Groups of 10 - 3 bits per Function - 30 bits per port

static uint8_t gpioToGPFSEL [] =
{
  0,0,0,0,0,0,0,0,0,0,
  1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,
  3,3,3,3,3,3,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,
  5,5,5,5,5,5,5,5,5,5,
} ;


// gpioToShift
//	Define the shift up for the 3 bits per pin in each GPFSEL port

static uint8_t gpioToShift [] =
{
  0,3,6,9,12,15,18,21,24,27,
  0,3,6,9,12,15,18,21,24,27,
  0,3,6,9,12,15,18,21,24,27,
  0,3,6,9,12,15,18,21,24,27,
  0,3,6,9,12,15,18,21,24,27,
  0,3,6,9,12,15,18,21,24,27,
} ;


// gpioToGPSET:
//	(Word) offset to the GPIO Set registers for each GPIO pin

static uint8_t gpioToGPSET [] =
{
   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
} ;

// gpioToGPCLR:
//	(Word) offset to the GPIO Clear registers for each GPIO pin

static uint8_t gpioToGPCLR [] =
{
  10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
  11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
} ;


// gpioToGPLEV:
//	(Word) offset to the GPIO Input level registers for each GPIO pin

static uint8_t gpioToGPLEV [] =
{
  13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
  14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
} ;


#ifdef notYetReady
// gpioToEDS
//	(Word) offset to the Event Detect Status

static uint8_t gpioToEDS [] =
{
  16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
  17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,
} ;

// gpioToREN
//	(Word) offset to the Rising edge ENable register

static uint8_t gpioToREN [] =
{
  19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,
  20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
} ;

// gpioToFEN
//	(Word) offset to the Falling edgde ENable register

static uint8_t gpioToFEN [] =
{
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
} ;
#endif


// GPPUD:
//	GPIO Pin pull up/down register

#define	GPPUD	37

/* 2711 has a different mechanism for pin pull-up/down/enable  */
#define GPPUPPDN0                57        /* Pin pull-up/down for pins 15:0  */
#define GPPUPPDN1                58        /* Pin pull-up/down for pins 31:16 */
#define GPPUPPDN2                59        /* Pin pull-up/down for pins 47:32 */
#define GPPUPPDN3                60        /* Pin pull-up/down for pins 57:48 */

static volatile unsigned int piGpioPupOffset = 0 ;

// gpioToPUDCLK
//	(Word) offset to the Pull Up Down Clock regsiter

static uint8_t gpioToPUDCLK [] =
{
  38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,
  39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,
} ;


// gpioToPwmALT
//	the ALT value to put a GPIO pin into PWM mode

static uint8_t gpioToPwmALT [] =
{
          0,         0,         0,         0,         0,         0,         0,         0,	//  0 ->  7
          0,         0,         0,         0, FSEL_ALT0, FSEL_ALT0,         0,         0, 	//  8 -> 15
          0,         0, FSEL_ALT5, FSEL_ALT5,         0,         0,         0,         0, 	// 16 -> 23
          0,         0,         0,         0,         0,         0,         0,         0,	// 24 -> 31
          0,         0,         0,         0,         0,         0,         0,         0,	// 32 -> 39
  FSEL_ALT0, FSEL_ALT0,         0,         0,         0, FSEL_ALT0,         0,         0,	// 40 -> 47
          0,         0,         0,         0,         0,         0,         0,         0,	// 48 -> 55
          0,         0,         0,         0,         0,         0,         0,         0,	// 56 -> 63
} ;

struct xj3PwmInfoStruct {
  int gpio_num;
  int pin_mux_val;
  int pwm_chip_id;
  int pwm_id;
};

static struct xj3PwmInfoStruct xj3PwmInfos [] =
{
  {4, 2, 0, 0},
  {25, 1, 3, 1},
  {-1, 0, 0, 0}
};

static int xj3GetPwmInfo(int gpio_num, struct xj3PwmInfoStruct *pwminfo)
{
  int i = 0;
  while (xj3PwmInfos[i].gpio_num != -1)
  {
    if (xj3PwmInfos[i].gpio_num == gpio_num)
    {
      *pwminfo = xj3PwmInfos[i];
      return 0;
    }
    i++;
  }
  return -1;
}

// static uint8_t gpioToX3PwmFunc [] =
// {
//           0,         0,         0,         0,         2,         0,         0,         0,	//  0 ->  7
//           0,         0,         0,         0,         0,         0,         0,         0, 	//  8 -> 15
//           0,         0,         0,         0,         0,         0,         0,         0, 	// 16 -> 23
//           0,         1,         0,         0,         0,         0,         0,         0,	// 24 -> 31
//           0,         0,         0,         0,         0,         0,         0,         0,	// 32 -> 39
//           0,         0,         0,         0,         0,         0,         0,         0,	// 40 -> 47
//           0,         0,         0,         0,         0,         0,         0,         0,	// 48 -> 55
//           0,         0,         0,         0,         0,         0,         0,         0,	// 56 -> 63
// } ;

// gpioToPwmPort
//	The port value to put a GPIO pin into PWM mode

static uint8_t gpioToPwmPort [] =
{
          0,         0,         0,         0,         0,         0,         0,         0,	//  0 ->  7
          0,         0,         0,         0, PWM0_DATA, PWM1_DATA,         0,         0, 	//  8 -> 15
          0,         0, PWM0_DATA, PWM1_DATA,         0,         0,         0,         0, 	// 16 -> 23
          0,         0,         0,         0,         0,         0,         0,         0,	// 24 -> 31
          0,         0,         0,         0,         0,         0,         0,         0,	// 32 -> 39
  PWM0_DATA, PWM1_DATA,         0,         0,         0, PWM1_DATA,         0,         0,	// 40 -> 47
          0,         0,         0,         0,         0,         0,         0,         0,	// 48 -> 55
          0,         0,         0,         0,         0,         0,         0,         0,	// 56 -> 63

} ;

// gpioToGpClkALT:
//	ALT value to put a GPIO pin into GP Clock mode.
//	On the Pi we can really only use BCM_GPIO_4 and BCM_GPIO_21
//	for clocks 0 and 1 respectively, however I'll include the full
//	list for completeness - maybe one day...

#define	GPIO_CLOCK_SOURCE	1

// gpioToGpClkALT0:

static uint8_t gpioToGpClkALT0 [] =
{
          0,         0,         0,         0, FSEL_ALT0, FSEL_ALT0, FSEL_ALT0,         0,	//  0 ->  7
          0,         0,         0,         0,         0,         0,         0,         0, 	//  8 -> 15
          0,         0,         0,         0, FSEL_ALT5, FSEL_ALT5,         0,         0, 	// 16 -> 23
          0,         0,         0,         0,         0,         0,         0,         0,	// 24 -> 31
  FSEL_ALT0,         0, FSEL_ALT0,         0,         0,         0,         0,         0,	// 32 -> 39
          0,         0, FSEL_ALT0, FSEL_ALT0, FSEL_ALT0,         0,         0,         0,	// 40 -> 47
          0,         0,         0,         0,         0,         0,         0,         0,	// 48 -> 55
          0,         0,         0,         0,         0,         0,         0,         0,	// 56 -> 63
} ;

// gpioToClk:
//	(word) Offsets to the clock Control and Divisor register

static uint8_t gpioToClkCon [] =
{
         -1,        -1,        -1,        -1,        28,        30,        32,        -1,	//  0 ->  7
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, 	//  8 -> 15
         -1,        -1,        -1,        -1,        28,        30,        -1,        -1, 	// 16 -> 23
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,	// 24 -> 31
         28,        -1,        28,        -1,        -1,        -1,        -1,        -1,	// 32 -> 39
         -1,        -1,        28,        30,        28,        -1,        -1,        -1,	// 40 -> 47
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,	// 48 -> 55
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,	// 56 -> 63
} ;

static uint8_t gpioToClkDiv [] =
{
         -1,        -1,        -1,        -1,        29,        31,        33,        -1,	//  0 ->  7
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, 	//  8 -> 15
         -1,        -1,        -1,        -1,        29,        31,        -1,        -1, 	// 16 -> 23
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,	// 24 -> 31
         29,        -1,        29,        -1,        -1,        -1,        -1,        -1,	// 32 -> 39
         -1,        -1,        29,        31,        29,        -1,        -1,        -1,	// 40 -> 47
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,	// 48 -> 55
         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,	// 56 -> 63
} ;

/* X3 Pin info */
#define X3_SYSCNT_BASE          (0xA6001000)
#define X3_GPIO_BASE            (0xA6003000)
#define PIN_MUX_BASE            (0xA6004000)
#define HB_GPIO_MODE            0

#define GPIO_GRP0_REG           0x000
#define GPIO_GRP1_REG           0x010
#define GPIO_GRP2_REG           0x020
#define GPIO_GRP3_REG           0x030
#define GPIO_GRP4_REG           0x040
#define GPIO_GRP5_REG           0x050
#define GPIO_GRP6_REG           0X060
#define GPIO_GRP7_REG           0x070

#define X3_IO_DIR_SHIFT         16

#define X3_GPIO0_CTRL_REG       0x8
#define X3_GPIO0_VALUE_REG      0xC
#define X3_GPIO1_CTRL_REG       0x18
#define X3_GPIO1_VALUE_REG      0x1c
#define X3_GPIO6_CTRL_REG       0x68
#define X3_GPIO6_VALUE_REG      0x6C

#define HB_PIN_MAX_NUMS         121

#define PIN_BASE_BOARD_SEL(x)   ((((x >> 14) & 0x1) << 0x1) | \
        ((x >> 12) & 0x1))
#define X3_MIPI_RESET_OUT_LOW(x)        (((x) | 0x10000000) & 0xffffefff)

#define PIN_CONFIG_GPIO(x)      ((x) | 0x3)
#define PIN_TYPE1_PULL_ENABLE_OFFSET   (6)
#define PIN_TYPE1_PULL_ENABLE    (1 << PIN_TYPE1_PULL_ENABLE_OFFSET)
#define PIN_TYPE1_PULL_DISABLE   (~(1 << PIN_TYPE1_PULL_ENABLE_OFFSET))
#define PIN_TYPE1_PULL_TYPE_OFFSET     (7)
#define PIN_TYPE1_PULL_UP        (1 << PIN_TYPE1_PULL_TYPE_OFFSET)
#define PIN_TYPE1_PULL_DOWN      (~(1 << PIN_TYPE1_PULL_TYPE_OFFSET))
#define PIN_TYPE2_PULL_UP_OFFSET (8)
#define PIN_TYPE2_PULL_UP_ENABLE   (1 << PIN_TYPE2_PULL_UP_OFFSET)
#define PIN_TYPE2_PULL_UP_DISABLE  (~(1 << PIN_TYPE2_PULL_UP_OFFSET))
#define PIN_TYPE2_PULL_DOWN_OFFSET   (7)
#define PIN_TYPE2_PULL_DOWN_ENABLE   (1 << PIN_TYPE2_PULL_DOWN_OFFSET)
#define PIN_TYPE2_PULL_DOWN_DISABLE  (~(1 << PIN_TYPE2_PULL_DOWN_OFFSET))

enum pull_state {
	NO_PULL = 0,
	PULL_UP,
	PULL_DOWN,
};

enum pin_dir {
	IN = 0,
	OUT,
};

enum xj3_pin_type {
	PIN_INVALID = 0,
	PULL_TYPE1,
	PULL_TYPE2,
};
/* to set xj3 pin */
struct pin_info {
	int pin_index;
	enum pull_state state;
};

static uint8_t xj3_all_pin_type[HB_PIN_MAX_NUMS] = {
	/* GPIO0[0-7] */
	PIN_INVALID, PULL_TYPE1, PULL_TYPE1, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO0[8-15] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE1, PULL_TYPE1, PIN_INVALID, PULL_TYPE2, PIN_INVALID, PULL_TYPE2,
	/* GPIO1[0-7] */
	PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO1[8-15] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO2[0-7] */
	PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1,
	/* GPIO2[8-15] */
	PULL_TYPE1, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO3[0-7] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO3[8-15] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO4[0-7] */
	PULL_TYPE1, PULL_TYPE1, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO4[8-15] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO5[0-7] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO5[8-15] */
	PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE1,
	/* GPIO6[0-7] */
	PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1,
	/* GPIO6[8-15] */
	PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE1, PULL_TYPE2,
	/* GPIO7[0-7] */
	PULL_TYPE2, PIN_INVALID, PULL_TYPE1, PULL_TYPE1, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2, PULL_TYPE2,
	/* GPIO7[8] */
	PULL_TYPE1,
};

/* Register offsets for the GPIO device */
#define X3_IO_CFG_OFFSET        0x0
#define X3_IO_PE_OFFSET         0x4
#define X3_IO_CTL_OFFSET        0x8
#define X3_IO_IN_VALUE_OFFSET   0xc

#define X3_GPIO_CFG_REG(BANK)     (X3_IO_CFG_OFFSET + (16 * BANK))
#define X3_GPIO_PE_REG(BANK)     (X3_IO_PE_OFFSET + (16 * BANK))
#define X3_GPIO_CTL_REG(BANK)     (X3_IO_CTL_OFFSET + (16 * BANK))
#define X3_GPIO_IN_VALUE_REG(BANK)     (X3_IO_IN_VALUE_OFFSET + (16 * BANK))

/*
 * Functions
 *********************************************************************************
 */

#define BIT(nr)                      (1UL << (nr))

static inline void reg32_write(volatile unsigned int *addr, uint32_t val)
{
  *addr = val;
}

static inline uint32_t reg32_read(volatile unsigned int *addr)
{
  return *addr;
}

static uint8_t xj3_pin_type(uint32_t pin)
{
	if (pin >= HB_PIN_MAX_NUMS)
		return PIN_INVALID;
	return xj3_all_pin_type[pin];
}

/**
 * xj3_gpio_get_bank_pin - Get the bank number and pin number within that bank
 * for a given pin in the GPIO device
 * @pin_num:    gpio pin number within the device
 * @bank_num:   an output parameter used to return the bank number of the gpio
 *              pin         
 * @bank_pin_num: an output parameter used to return pin number within a bank
 *                for the given gpio pin
 *
 * Returns the bank number and pin offset within the bank.
 */
static int xj3_gpio_get_bank_pin(unsigned int pin_num,
                        unsigned int *bank_num,
                        unsigned int *bank_pin_num)
{  
  *bank_num = (int)(pin_num / 16);
  *bank_pin_num = pin_num % 16;

  if (*bank_num >= 8) {
    // printf("Invalid bank and pin num: %d\n", pin_num);
    *bank_num = 0;
    *bank_pin_num = 0;
    return -1;
  }

  return 0;
}

static void xj3_set_pin_mux_gpio(char gpio_num)
{
	uint32_t reg = 0;
	uint32_t offset = 0;

	offset = gpio_num * 4;
	reg = reg32_read((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4));
  reg |= 3;
	reg32_write((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
  // printf("offset: 0x%x, pmux_reg: 0x%x reg: 0x%x\n", offset, (volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
}

static void xj3_set_pin_mux_pwm(char gpio_num, int val)
{
	uint32_t reg = 0;
	uint32_t offset = 0;

	offset = gpio_num * 4;
	reg = reg32_read((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4));
  reg &= ~(0x3);
  reg |= (val & 0x3);
	reg32_write((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
  // printf("offset: 0x%x, pmux_reg: 0x%x reg: 0x%x\n", offset, (volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
}

static void xj3_set_pin_drive(char gpio_num, uint32_t val)
{
	uint32_t reg = 0;
	uint32_t offset = 0;

	offset = gpio_num * 4;
	reg = reg32_read((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4));
  reg &= ~(0xf << 2);
  reg |= (val << 2);
	reg32_write((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
}

static void xj3_set_gpio_pull(char gpio_num, int state)
{
	uint32_t reg = 0;
	uint32_t offset = 0;
  uint32_t pin_type = xj3_pin_type(gpio_num);

	offset = gpio_num * 4;
	reg = reg32_read((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4));
  // printf("pull gpio_num: %d, pin_type: %d, reg: 0x%x\n", gpio_num, pin_type, reg);
	if (pin_type == PULL_TYPE1) {
		switch (state) {
		case PUD_OFF:
			reg &= PIN_TYPE1_PULL_DISABLE;
			break;
		case PUD_UP:
			reg |= PIN_TYPE1_PULL_ENABLE;
			reg |= PIN_TYPE1_PULL_UP;
			break;
		case PUD_DOWN:
			reg |= PIN_TYPE1_PULL_ENABLE;
			reg &= PIN_TYPE1_PULL_DOWN;
			break;
		default:
			break;
		}
		reg32_write((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
	} else if (pin_type == PULL_TYPE2) {
		switch (state) {
		case PUD_OFF:
			reg &= PIN_TYPE2_PULL_UP_DISABLE;
			reg &= PIN_TYPE2_PULL_DOWN_DISABLE;
			break;
		case PUD_UP:
			reg |= PIN_TYPE2_PULL_UP_ENABLE;
			reg &= PIN_TYPE2_PULL_DOWN_DISABLE;
			break;
		case PUD_DOWN:
			reg |= PIN_TYPE2_PULL_DOWN_ENABLE;
			reg &= PIN_TYPE2_PULL_UP_DISABLE;
			break;
		default:
			break;
		}
		reg32_write((volatile unsigned int *)(_wiringPiGpioPinMux + offset/4), reg);
	}
  // printf("pull gpio_num: %d, pin_type: %d, reg: 0x%x\n", gpio_num, pin_type, reg);
}

// void dump_pin_info(void)
// {
// 	uint32_t i = 0;
// 	uint32_t reg = 0;
// 	for (i = 0; i < 8; i++) {
// 		reg = _wiringPiGpio + 0x10 * i + 0x8;
// 		printf("reg[0x%x] = 0x%x\n", reg, reg32_read(reg));
// 		reg = _wiringPiGpio + 0x10 * i + 0xc;
// 		printf("reg[0x%x] = 0x%x\n", reg, reg32_read(reg));
// 	}
// 	for (i = 0; i < HB_PIN_MAX_NUMS; i++) {
// 		reg = PIN_MUX_BASE + i * 4;
// 		printf("pin[%d] status reg[0x%x] = 0x%x\n", i, reg, reg32_read(reg));
// 	}
// }

static int xj3_gpio_direction_input(int gpio_num)
{
  int ret;
  uint32_t reg;
  unsigned int bank_num, bank_pin_num;

  ret = xj3_gpio_get_bank_pin(gpio_num, &bank_num, &bank_pin_num);
  if (ret)
    return ret;

  xj3_set_pin_mux_gpio(gpio_num);

  /* clear the bit in direction mode reg to set the pin as input */
  reg = reg32_read((volatile unsigned int *)_wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4);
  reg &= ~BIT(bank_pin_num + X3_IO_DIR_SHIFT);
  reg32_write((volatile unsigned int *)_wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4, reg);

  // printf("gpio_num: %d, bank_num: 0x%x, bank_pin_num: 0x%x, reg: 0x%x\n", gpio_num, bank_num, bank_pin_num, reg);

  return 0;
}
 
static int xj3_gpio_direction_output(int gpio_num)
{
  int ret;
  uint32_t reg;
  unsigned int bank_num, bank_pin_num;

  ret = xj3_gpio_get_bank_pin(gpio_num, &bank_num, &bank_pin_num);
  if (ret)
    return ret;

  xj3_set_pin_mux_gpio(gpio_num);

  /* set the GPIO pin as output */
  reg = reg32_read((volatile unsigned int *)_wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4);
  reg |= BIT(bank_pin_num + X3_IO_DIR_SHIFT);
  reg32_write((volatile unsigned int *)_wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4, reg);

  // printf("gpio_num: %d, bank_num: 0x%x, bank_pin_num: 0x%x, reg: 0x%x\n", gpio_num, bank_num, bank_pin_num, reg);

  return 0;
}

static int xj3_get_pin_value(unsigned gpio_num)
{
  int ret;
  uint32_t data;
  unsigned int bank_num, bank_pin_num;

  ret = xj3_gpio_get_bank_pin(gpio_num, &bank_num, &bank_pin_num);
  if (ret)
    return ret;

  data = reg32_read((volatile unsigned int *)(_wiringPiGpio +
                        X3_GPIO_IN_VALUE_REG(bank_num)/4));

  // printf("gpio_num: %d bank_num: %d bank_pin_num: %d, bank_offset: %x, addr=0x%x, val=0x%x, val2=0x%x\n",
  //   gpio_num, bank_num, bank_pin_num, X3_GPIO_IN_VALUE_REG(bank_num), 
  //   (volatile unsigned int *)_wiringPiGpio + X3_GPIO_IN_VALUE_REG(bank_num)/4,
  //   data, *(volatile unsigned int *)(_wiringPiGpio + X3_GPIO_IN_VALUE_REG(bank_num)/4));

  return (data >> bank_pin_num) & 1;
}

static int xj3_set_pin_value(unsigned gpio_num, int value)
{
  int ret;
  uint32_t reg;
  unsigned int bank_num, bank_pin_num;

  ret = xj3_gpio_get_bank_pin(gpio_num, &bank_num, &bank_pin_num);
  if (ret)
    return ret;

  reg = reg32_read((volatile unsigned int *)(_wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4));

  if(value){
      reg |= BIT(bank_pin_num);
  }
  else{
      reg &= ~BIT(bank_pin_num);
  }

  // printf("set gpio_num: %d, reg: 0x%x, value: 0x%x, ctl_reg: 0x%x\n", gpio_num, reg, value,
  //   _wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4);
  
  // printf("set gpio_num: %d, bank_num: 0x%x, bank_pin_num: 0x%x, reg: 0x%x value: 0x%x\n",
  //   gpio_num, bank_num, bank_pin_num, reg, value);

  reg32_write((volatile unsigned int *)(_wiringPiGpio + X3_GPIO_CTL_REG(bank_num)/4), reg);

  return 0;
}

#define SYSFS_GPIO_DIR "/sys/class/gpio"
#define MAX_BUF	64
#define RET_OK 0
#define RET_ERR -1

/****************************************************************
 * gpio_export
 ****************************************************************/
int gpio_export(unsigned int gpio_num)
{
  int ret = 0;
	int fd, len;
	char buf[MAX_BUF];

	fd = open(SYSFS_GPIO_DIR "/export", O_WRONLY);
	if (fd < 0) {
		perror("gpio/export");
		return fd;
	}

	len = snprintf(buf, sizeof(buf), "%d", gpio_num);
	ret = write(fd, buf, len);
	close(fd);

  if (ret != len)
    return RET_ERR;

  return RET_OK;
}

/****************************************************************
 * gpio_unexport
 ****************************************************************/
int gpio_unexport(unsigned int gpio_num)
{
  int ret = 0;
	int fd, len;
	char buf[MAX_BUF];

	fd = open(SYSFS_GPIO_DIR "/unexport", O_WRONLY);
	if (fd < 0) {
		perror("gpio/export");
		return fd;
	}

	len = snprintf(buf, sizeof(buf), "%d", gpio_num);
	ret = write(fd, buf, len);
	close(fd);
	if (ret != len)
    return RET_ERR;

  return RET_OK;
}

/****************************************************************
 * gpio_set_dir
 ****************************************************************/
int gpio_set_dir(unsigned int gpio_num, unsigned int out_flag)
{
  int ret = 0;
	int fd, len;
	char buf[MAX_BUF];

	snprintf(buf, sizeof(buf), SYSFS_GPIO_DIR  "/gpio%d/direction", gpio_num);

	fd = open(buf, O_WRONLY);
	if (fd < 0) {
		perror("gpio/direction");
		return fd;
	}

	if (out_flag)
    len = snprintf(buf, sizeof(buf), "out");
	else
		len = snprintf(buf, sizeof(buf), "in");

  ret = write(fd, buf, len);
	close(fd);

	if (ret != len)
    return RET_ERR;

  return RET_OK;
}

/****************************************************************
 * gpio_set_value
 ****************************************************************/
int gpio_set_value(unsigned int gpio_num, unsigned int value)
{
  int ret = 0;
	int fd, len;
	char buf[MAX_BUF];

	snprintf(buf, sizeof(buf), SYSFS_GPIO_DIR "/gpio%d/value", gpio_num);

	fd = open(buf, O_WRONLY);
	if (fd < 0) {
		perror("gpio/set-value");
		return fd;
	}

  if (value)
    len = snprintf(buf, sizeof(buf), "1");
  else
    len = snprintf(buf, sizeof(buf), "0");

  ret = write(fd, buf, len);
	close(fd);

	if (ret != len)
    return RET_ERR;

  return RET_OK;
}

/****************************************************************
 * gpio_get_value
 ****************************************************************/
int gpio_get_value(unsigned int gpio_num, unsigned int *value)
{
  int ret = 0;
	int fd;
	char buf[MAX_BUF];
	char ch;

	snprintf(buf, sizeof(buf), SYSFS_GPIO_DIR "/gpio%d/value", gpio_num);

	fd = open(buf, O_RDONLY);
	if (fd < 0) {
		perror("gpio/get-value");
		return fd;
	}

	ret = read(fd, &ch, 1);

	if (ch != '0') {
		*value = 1;
	} else {
		*value = 0;
	}

	close(fd);

  if (ret != 1)
    return RET_ERR;

  return RET_OK;
}

/****************************************************************
 * gpio_set_edge
 ****************************************************************/
int gpio_set_edge(unsigned int gpio_num, char *edge)
{
  int ret = 0, len = 0;
	int fd;
	char buf[MAX_BUF];

	snprintf(buf, sizeof(buf), SYSFS_GPIO_DIR "/gpio%d/edge", gpio_num);

	fd = open(buf, O_WRONLY);
	if (fd < 0) {
		perror("gpio/set-edge");
		return fd;
	}

  len = strlen(edge) + 1;
	ret = write(fd, edge, len);
	close(fd);

	if (ret != len)
    return RET_ERR;

  return RET_OK;
}

/*
 * wiringPiFailure:
 *	Fail. Or not.
 *********************************************************************************
 */

int wiringPiFailure (int fatal, const char *message, ...)
{
  va_list argp ;
  char buffer [1024] ;

  if (!fatal && wiringPiReturnCodes)
    return -1 ;

  va_start (argp, message) ;
    vsnprintf (buffer, 1023, message, argp) ;
  va_end (argp) ;

  fprintf (stderr, "%s", buffer) ;
  exit (EXIT_FAILURE) ;

  return 0 ;
}


/*
 * setupCheck
 *	Another sanity check because some users forget to call the setup
 *	function. Mosty because they need feeding C drip by drip )-:
 *********************************************************************************
 */

static void setupCheck (const char *fName)
{
  if (!wiringPiSetuped)
  {
    fprintf (stderr, "%s: You have not called one of the wiringPiSetup\n"
	"  functions, so I'm aborting your program before it crashes anyway.\n", fName) ;
    exit (EXIT_FAILURE) ;
  }
}

/*
 * gpioMemCheck:
 *	See if we're using the /dev/gpiomem interface, if-so then some operations
 *	can't be done and will crash the Pi.
 *********************************************************************************
 */

static void usingGpioMemCheck (const char *what)
{
  if (usingGpioMem)
  {
    fprintf (stderr, "%s: Unable to do this when using /dev/gpiomem. Try sudo?\n", what) ;
    exit (EXIT_FAILURE) ;
  }
}



/*
 * piGpioLayout:
 *	Return a number representing the hardware revision of the board.
 *	This is not strictly the board revision but is used to check the
 *	layout of the GPIO connector - and there are 2 types that we are
 *	really interested in here. The very earliest Pi's and the
 *	ones that came after that which switched some pins ....
 *
 *	Revision 1 really means the early Model A and B's.
 *	Revision 2 is everything else - it covers the B, B+ and CM.
 *		... and the Pi 2 - which is a B+ ++  ...
 *		... and the Pi 0 - which is an A+ ...
 *
 *	The main difference between the revision 1 and 2 system that I use here
 *	is the mapping of the GPIO pins. From revision 2, the Pi Foundation changed
 *	3 GPIO pins on the (original) 26-way header - BCM_GPIO 22 was dropped and
 *	replaced with 27, and 0 + 1 - I2C bus 0 was changed to 2 + 3; I2C bus 1.
 *
 *	Additionally, here we set the piModel2 flag too. This is again, nothing to
 *	do with the actual model, but the major version numbers - the GPIO base
 *	hardware address changed at model 2 and above (not the Zero though)
 *
 *********************************************************************************
 */

static void piGpioLayoutOops (const char *why)
{
  fprintf (stderr, "Oops: Unable to determine board revision from /proc/cpuinfo\n") ;
  fprintf (stderr, " -> %s\n", why) ;
  fprintf (stderr, " ->  You'd best google the error to find out why.\n") ;
//fprintf (stderr, " ->  http://www.raspberrypi.org/phpBB3/viewtopic.php?p=184410#p184410\n") ;
  exit (EXIT_FAILURE) ;
}

int piGpioLayout (void)
{
  FILE *cpuFd ;
  char line [120] ;
  char *c ;
  static int  gpioLayout = -1 ;

  if (gpioLayout != -1)	// No point checking twice
    return gpioLayout ;

  if ((cpuFd = fopen ("/sys/class/socinfo/board_id", "r")) == NULL)
    piGpioLayoutOops ("Unable to open /sys/class/socinfo/board_id") ;

  fgets (line, 120, cpuFd);

  fclose (cpuFd) ;

  if (wiringPiDebug)
    printf ("piGpioLayout: board_id string: %s\n", line) ;

  c = line;
// Chomp trailing CR/NL

  for (c = &line [strlen (line) - 1] ; (*c == '\n') || (*c == '\r') ; --c)
    *c = 0 ;

// Chomp spaces
  c = line;
  ++c ;
  while (isspace (*c))
    ++c ;

  if (!isxdigit (*c))
    piGpioLayoutOops ("Bogus \"board_id\" line (no hex digit at start of revision)") ;

// Make sure its long enough

  if (strlen (c) < 4)
    piGpioLayoutOops ("Bogus board_id line (too small)") ;

// Isolate  last 3 characters: (in-case of overvolting or new encoding scheme)

  c = c + strlen (c) - 3 ;

  if (wiringPiDebug)
    printf ("piGpioLayout: last3Chars are: \"%s\"\n", c) ;

  if (strcmp (c, "304") == 0)
    gpioLayout = 3 ;
  else if (strcmp (c, "404") == 0)
    gpioLayout = 4 ;
  else if (strcmp (c, "504") == 0)
    gpioLayout = 5 ;
  else if (strcmp (c, "604") == 0)
    gpioLayout = 6 ;
  else if (strcmp (c, "804") == 0)
    gpioLayout = 8 ;
  else if (strcmp (c, "b04") == 0)
    gpioLayout = 0xb ;
  else
    gpioLayout = 3 ; // Covers everything else from the B revision 2 to the B+, the Pi v2, v3, zero and CM's.

  if (wiringPiDebug)
    printf ("piGpioLayoutOops: Returning Board type: %d\n", gpioLayout) ;

  return gpioLayout ;
}

/*
 * piBoardRev:
 *	Deprecated, but does the same as piGpioLayout
 *********************************************************************************
 */

int piBoardRev (void)
{
  return piGpioLayout () ;
}



/*
 * piBoardId:
 *	Return the real details of the board we have.
 *
 *	This is undocumented and really only intended for the GPIO command.
 *	Use at your own risk!
 *
 *	Seems there are some boards with 0000 in them (mistake in manufacture)
 *	So the distinction between boards that I can see is:
 *
 *		0000 - Error
 *		0001 - Not used
 *
 *	Original Pi boards:
 *		0002 - Model B,  Rev 1,   256MB, Egoman
 *		0003 - Model B,  Rev 1.1, 256MB, Egoman, Fuses/D14 removed.
 *
 *	Newer Pi's with remapped GPIO:
 *		0004 - Model B,  Rev 1.2, 256MB, Sony
 *		0005 - Model B,  Rev 1.2, 256MB, Egoman
 *		0006 - Model B,  Rev 1.2, 256MB, Egoman
 *
 *		0007 - Model A,  Rev 1.2, 256MB, Egoman
 *		0008 - Model A,  Rev 1.2, 256MB, Sony
 *		0009 - Model A,  Rev 1.2, 256MB, Egoman
 *
 *		000d - Model B,  Rev 1.2, 512MB, Egoman	(Red Pi, Blue Pi?)
 *		000e - Model B,  Rev 1.2, 512MB, Sony
 *		000f - Model B,  Rev 1.2, 512MB, Egoman
 *
 *		0010 - Model B+, Rev 1.2, 512MB, Sony
 *		0013 - Model B+  Rev 1.2, 512MB, Embest
 *		0016 - Model B+  Rev 1.2, 512MB, Sony
 *		0019 - Model B+  Rev 1.2, 512MB, Egoman
 *
 *		0011 - Pi CM,    Rev 1.1, 512MB, Sony
 *		0014 - Pi CM,    Rev 1.1, 512MB, Embest
 *		0017 - Pi CM,    Rev 1.1, 512MB, Sony
 *		001a - Pi CM,    Rev 1.1, 512MB, Egoman
 *
 *		0012 - Model A+  Rev 1.1, 256MB, Sony
 *		0015 - Model A+  Rev 1.1, 512MB, Embest
 *		0018 - Model A+  Rev 1.1, 256MB, Sony
 *		001b - Model A+  Rev 1.1, 256MB, Egoman
 *
 *	A small thorn is the olde style overvolting - that will add in
 *		1000000
 *
 *	The Pi compute module has an revision of 0011 or 0014 - since we only
 *	check the last digit, then it's 1, therefore it'll default to not 2 or
 *	3 for a	Rev 1, so will appear as a Rev 2. This is fine for the most part, but
 *	we'll properly detect the Compute Module later and adjust accordingly.
 *
 * And then things changed with the introduction of the v2...
 *
 * For Pi v2 and subsequent models - e.g. the Zero:
 *
 *   [USER:8] [NEW:1] [MEMSIZE:3] [MANUFACTURER:4] [PROCESSOR:4] [TYPE:8] [REV:4]
 *   NEW          23: will be 1 for the new scheme, 0 for the old scheme
 *   MEMSIZE      20: 0=256M 1=512M 2=1G
 *   MANUFACTURER 16: 0=SONY 1=EGOMAN 2=EMBEST
 *   PROCESSOR    12: 0=2835 1=2836
 *   TYPE         04: 0=MODELA 1=MODELB 2=MODELA+ 3=MODELB+ 4=Pi2 MODEL B 5=ALPHA 6=CM
 *   REV          00: 0=REV0 1=REV1 2=REV2
 *********************************************************************************
 */

void piBoardId (int *model, int *rev, int *mem, int *maker, int *warranty)
{
  FILE *cpuFd ;
  char line [120] ;
  char *c ;
  // unsigned int revision ;
  // int bRev, bType, bProc, bMfg, bMem, bWarranty ;

//	Will deal with the properly later on - for now, lets just get it going...
//  unsigned int modelNum ;

  (void)piGpioLayout () ;	// Call this first to make sure all's OK. Don't care about the result.

  if ((cpuFd = fopen ("/sys/class/socinfo/board_id", "r")) == NULL)
    piGpioLayoutOops ("Unable to open /sys/class/socinfo/board_id") ;

  fgets (line, 120, cpuFd);

  fclose (cpuFd) ;

// Chomp trailing CR/NL

  for (c = &line [strlen (line) - 1] ; (*c == '\n') || (*c == '\r') ; --c)
    *c = 0 ;

  if (wiringPiDebug)
    printf ("piBoardId: board_id string: %s\n", line) ;

// Need to work out if it's using the new or old encoding scheme:

// Chomp spaces
  c = line;
  ++c ;
  while (isspace (*c))
    ++c ;

  if (!isxdigit (*c))
    piGpioLayoutOops ("Bogus \"board_id\" line (no hex digit at start of revision)") ;

  // revision = (unsigned int)strtol (c, NULL, 16) ; // Hex number with no leading 0x

  if (wiringPiDebug)
    printf ("piBoardId: Old Way: board_id is: %s\n", c) ;

  if (!isdigit (*c))
    piGpioLayoutOops ("Bogus \"board_id\" line (no digit at start of revision)") ;

// Make sure its long enough

  if (strlen (c) < 3)
    piGpioLayoutOops ("Bogus \"board_id\" line (not long enough)") ;

// If longer than 3, we'll assume it's been overvolted

  *warranty = 0;

// Extract last 4 characters:

  c = c + strlen (c) - 3 ;

// Fill out the replys as appropriate

  /**/ if (strcmp (c, "304") == 0) { *model = PI_MODEL_SDB  ; *rev = PI_VERSION_3   ; *mem = 3 ; *maker = PI_MAKER_HORIZON  ; }
  else if (strcmp (c, "404") == 0) { *model = PI_MODEL_SDB  ; *rev = PI_VERSION_4   ; *mem = 3 ; *maker = PI_MAKER_HORIZON  ; }
  else if (strcmp (c, "504") == 0) { *model = PI_MODEL_RDKX3 ; *rev = PI_VERSION_1   ; *mem = 3 ; *maker = PI_MAKER_HORIZON    ; }
  else if (strcmp (c, "604") == 0) { *model = PI_MODEL_RDKX3V1_2 ; *rev = PI_VERSION_1   ; *mem = 3 ; *maker = PI_MAKER_HORIZON    ; }
  else if (strcmp (c, "804") == 0) { *model = PI_MODEL_RDKX3V2 ; *rev = PI_VERSION_1   ; *mem = 3 ; *maker = PI_MAKER_HORIZON    ; }
  else if (strcmp (c, "b04") == 0) { *model = PI_MODEL_RDKX3MD ; *rev = PI_VERSION_1   ; *mem = 3 ; *maker = PI_MAKER_HORIZON    ; }
  else                             { *model = PI_MODEL_SDB  ; *rev = PI_VERSION_3   ; *mem = 3 ; *maker = PI_MAKER_HORIZON ;               }
}



/*
 * wpiPinToGpio:
 *	Translate a wiringPi Pin number to native GPIO pin number.
 *	Provided for external support.
 *********************************************************************************
 */

int wpiPinToGpio (int wpiPin)
{
  return pinToGpio [wpiPin & 63] ;
}


/*
 * physPinToGpio:
 *	Translate a physical Pin number to native GPIO pin number.
 *	Provided for external support.
 *********************************************************************************
 */

int physPinToGpio (int physPin)
{
  return physToPin [physPin & 63] ;
}


/*
 * setPadDrive:
 *	Set the PAD driver value
 *********************************************************************************
 */

void setPadDrive (int pin, int value)
{
  if ((wiringPiMode == WPI_MODE_PINS) || (wiringPiMode == WPI_MODE_PHYS) || (wiringPiMode == WPI_MODE_GPIO))
  {
    if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
    {
      if (wiringPiMode == WPI_MODE_PINS)
        pin = pinToGpio [pin] ;
      else if (wiringPiMode == WPI_MODE_PHYS)
        pin = physToGpio [pin];
      else if (wiringPiMode != WPI_MODE_GPIO)
        return ;

      xj3_set_pin_drive(pin, value);

      if (wiringPiDebug)
      {
        printf ("setPadDrive: gpio: %d, value: %d\n", pin, value) ;
        // printf ("Read : %08X\n", *(pads + group + 11)) ;
      }
    }
  }
}


/*
 * getAlt:
 *	Returns the ALT bits for a given port. Only really of-use
 *	for the gpio readall command (I think)
 *********************************************************************************
 */

int getAlt (int pin)
{
  pin &= 120 ;

  /**/ if (wiringPiMode == WPI_MODE_PINS)
    pin = pinToGpio [pin] ;
  else if (wiringPiMode == WPI_MODE_PHYS)
    pin = physToGpio [pin] ;
  else if (wiringPiMode != WPI_MODE_GPIO)
    return 0 ;

  return 0 ;
}


/*
 * pwmSetMode:
 *	Select the native "balanced" mode, or standard mark:space mode
 *********************************************************************************
 */

void pwmSetMode (int mode)
{
  printf("pwmSetMode: %d\n", mode);
}


/*
 * pwmSetRange:
 *	Set the PWM range register. We set both range registers to the same
 *	value. If you want different in your own code, then write your own.
 *********************************************************************************
 */

void pwmSetRange (unsigned int range)
{
  if ((wiringPiMode == WPI_MODE_PINS) || (wiringPiMode == WPI_MODE_PHYS) || (wiringPiMode == WPI_MODE_GPIO))
  {
    *(pwm + PWM0_RANGE) = range ; delayMicroseconds (10) ;
    *(pwm + PWM1_RANGE) = range ; delayMicroseconds (10) ;
  }
}


/*
 * pwmSetClock:
 *	Set/Change the PWM clock. Originally my code, but changed
 *	(for the better!) by Chris Hall, <chris@kchall.plus.com>
 *	after further study of the manual and testing with a 'scope
 *********************************************************************************
 */

void pwmSetClock (int divisor)
{
  uint32_t pwm_control ;

  if (piGpioBase == GPIO_PERI_BASE_2711)
  {
    divisor = 540*divisor/192;
  }
  divisor &= 4095 ;

  if ((wiringPiMode == WPI_MODE_PINS) || (wiringPiMode == WPI_MODE_PHYS) || (wiringPiMode == WPI_MODE_GPIO))
  {
    if (wiringPiDebug)
      printf ("Setting to: %d. Current: 0x%08X\n", divisor, *(clk + PWMCLK_DIV)) ;

    pwm_control = *(pwm + PWM_CONTROL) ;		// preserve PWM_CONTROL

// We need to stop PWM prior to stopping PWM clock in MS mode otherwise BUSY
// stays high.

    *(pwm + PWM_CONTROL) = 0 ;				// Stop PWM

// Stop PWM clock before changing divisor. The delay after this does need to
// this big (95uS occasionally fails, 100uS OK), it's almost as though the BUSY
// flag is not working properly in balanced mode. Without the delay when DIV is
// adjusted the clock sometimes switches to very slow, once slow further DIV
// adjustments do nothing and it's difficult to get out of this mode.

    *(clk + PWMCLK_CNTL) = BCM_PASSWORD | 0x01 ;	// Stop PWM Clock
      delayMicroseconds (110) ;			// prevents clock going sloooow

    while ((*(clk + PWMCLK_CNTL) & 0x80) != 0)	// Wait for clock to be !BUSY
      delayMicroseconds (1) ;

    *(clk + PWMCLK_DIV)  = BCM_PASSWORD | (divisor << 12) ;

    *(clk + PWMCLK_CNTL) = BCM_PASSWORD | 0x11 ;	// Start PWM clock
    *(pwm + PWM_CONTROL) = pwm_control ;		// restore PWM_CONTROL

    if (wiringPiDebug)
      printf ("Set     to: %d. Now    : 0x%08X\n", divisor, *(clk + PWMCLK_DIV)) ;
  }
}

/* xj3 PWM */

// static int xj3PwmDutyCycles[121];

/*
 * xj3PwmSetPeriod:
 *	Set the PWM range register. We set both range registers to the same
 *	value. If you want different in your own code, then write your own.
 *********************************************************************************
 */
#define SYSFS_PWM_DIR "/sys/class/pwm"

int xj3PwmExport (struct xj3PwmInfoStruct pwminfo)
{
  int ret = 0;
  int fd, len;
  char buf[MAX_BUF];
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/export", pwminfo.pwm_chip_id) ;

  fd = open(fName, O_WRONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  len = snprintf(buf, sizeof(buf), "%d", pwminfo.pwm_id);
  ret = write(fd, buf, len);
  close(fd);

  if (ret != len)
    return RET_ERR;

  return RET_OK;
}

int xj3PwmUnExport (struct xj3PwmInfoStruct pwminfo)
{
  int ret = 0;
  int fd, len;
  char buf[MAX_BUF];
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/unexport", pwminfo.pwm_chip_id) ;

  fd = open(fName, O_WRONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  len = snprintf(buf, sizeof(buf), "%d", pwminfo.pwm_id);
  ret = write(fd, buf, len);
  close(fd);

  if (ret != len)
    return RET_ERR;

  return RET_OK;
}

int xj3PwmEnable (struct xj3PwmInfoStruct pwminfo, unsigned int enable)
{
  int ret = 0;
  int fd, len;
  char buf[MAX_BUF];
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/pwm%d/enable", pwminfo.pwm_chip_id, pwminfo.pwm_id) ;

  fd = open(fName, O_WRONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  if (1 == enable)
    len = snprintf(buf, sizeof(buf), "1");
  else
    len = snprintf(buf, sizeof(buf), "0");
  ret = write(fd, buf, len);
  close(fd);

  if (ret != len)
    return RET_ERR;

  return RET_OK;
}

unsigned int xj3PwmGetFreq (struct xj3PwmInfoStruct pwminfo)
{
  int fd;
  char buf[MAX_BUF] = {0};
  unsigned int period_ns = 0;
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/pwm%d/period", pwminfo.pwm_chip_id, pwminfo.pwm_id) ;

  fd = open(fName, O_RDONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  read(fd, buf, sizeof(buf));

  period_ns = atoi(buf);

  return period_ns;
}

unsigned int xj3PwmGetDutyCycle (struct xj3PwmInfoStruct pwminfo)
{
  int fd;
  char buf[MAX_BUF] = {0};
  unsigned int duty_cycle_ns = 0;
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/pwm%d/duty_cycle", pwminfo.pwm_chip_id, pwminfo.pwm_id) ;

  fd = open(fName, O_RDONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  read(fd, buf, sizeof(buf));

  duty_cycle_ns = atoi(buf);

  return duty_cycle_ns;
}

int xj3PwmSetDutyCycle (struct xj3PwmInfoStruct pwminfo, unsigned int duty_cycle)
{
  int ret = 0;
  int fd, len;
  unsigned int period_ns = 0;
  unsigned int duty_cycle_ns = 0;
  char buf[MAX_BUF];
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/pwm%d/duty_cycle", pwminfo.pwm_chip_id, pwminfo.pwm_id) ;

  fd = open(fName, O_WRONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  period_ns = xj3PwmGetFreq(pwminfo);

  duty_cycle_ns = (unsigned int)(period_ns * duty_cycle / 100);

  len = snprintf(buf, sizeof(buf), "%d", duty_cycle_ns);
  ret = write(fd, buf, len);
  close(fd);

  if (ret != len)
    return RET_ERR;

  return RET_OK;
}

int xj3PwmSetFreq (struct xj3PwmInfoStruct pwminfo, unsigned int freq)
{
  int ret = 0;
  int fd, len;
  char buf[MAX_BUF];
  unsigned int period_ns = 0;
  unsigned int duty_cycle_ns = 0;
  unsigned int duty_cycle = 0;
  char fName [64] ;
  sprintf (fName, SYSFS_PWM_DIR"/pwmchip%d/pwm%d/period", pwminfo.pwm_chip_id, pwminfo.pwm_id) ;

  fd = open(fName, O_WRONLY);
  if (fd < 0) {
    perror(fName);
    return fd;
  }

  period_ns = xj3PwmGetFreq(pwminfo);

  if (period_ns != 0)
  {
    duty_cycle_ns = xj3PwmGetDutyCycle(pwminfo);
    if (duty_cycle_ns >= period_ns)
      duty_cycle = 100;
    else
      duty_cycle = (int)(duty_cycle_ns * 100 / period_ns);
  }

  xj3PwmSetDutyCycle(pwminfo, 0);

  period_ns = (unsigned int)(1000000000.0 / freq);
  len = snprintf(buf, sizeof(buf), "%d", period_ns);
  ret = write(fd, buf, len);
  close(fd);

  xj3PwmSetDutyCycle(pwminfo, duty_cycle);

  if (ret != len)
    return RET_ERR;

  return RET_OK;
}

void pwmSetFreq (int pin, unsigned int frequency)
{
  struct xj3PwmInfoStruct pwminfo;
  setupCheck ("pwmSetFreq") ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
  {
    /**/ if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin] ;
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    if (xj3GetPwmInfo(pin, &pwminfo) == -1)	// Not a hardware capable PWM pin
        return ;

    xj3PwmSetFreq(pwminfo, frequency);
  }
}

void pwmSetDuty (int pin, unsigned int duty_cycle_ns)
{
  struct xj3PwmInfoStruct pwminfo;
  setupCheck ("pwmSetDuty") ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
  {
    /**/ if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin] ;
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    if (xj3GetPwmInfo(pin, &pwminfo) == -1)	// Not a hardware capable PWM pin
        return ;

    xj3PwmSetDutyCycle(pwminfo, duty_cycle_ns);
  }
}

/*
 * gpioClockSet:
 *	Set the frequency on a GPIO clock pin
 *********************************************************************************
 */

void gpioClockSet (int pin, int freq)
{
  int divi, divr, divf ;

  pin &= 63 ;

  /**/ if (wiringPiMode == WPI_MODE_PINS)
    pin = pinToGpio [pin] ;
  else if (wiringPiMode == WPI_MODE_PHYS)
    pin = physToGpio [pin] ;
  else if (wiringPiMode != WPI_MODE_GPIO)
    return ;

  divi = 19200000 / freq ;
  divr = 19200000 % freq ;
  divf = (int)((double)divr * 4096.0 / 19200000.0) ;

  if (divi > 4095)
    divi = 4095 ;

  *(clk + gpioToClkCon [pin]) = BCM_PASSWORD | GPIO_CLOCK_SOURCE ;		// Stop GPIO Clock
  while ((*(clk + gpioToClkCon [pin]) & 0x80) != 0)				// ... and wait
    ;

  *(clk + gpioToClkDiv [pin]) = BCM_PASSWORD | (divi << 12) | divf ;		// Set dividers
  *(clk + gpioToClkCon [pin]) = BCM_PASSWORD | 0x10 | GPIO_CLOCK_SOURCE ;	// Start Clock
}


/*
 * wiringPiFindNode:
 *      Locate our device node
 *********************************************************************************
 */

struct wiringPiNodeStruct *wiringPiFindNode (int pin)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;

  while (node != NULL)
    if ((pin >= node->pinBase) && (pin <= node->pinMax))
      return node ;
    else
      node = node->next ;

  return NULL ;
}


/*
 * wiringPiNewNode:
 *	Create a new GPIO node into the wiringPi handling system
 *********************************************************************************
 */

static         void pinModeDummy             (UNU struct wiringPiNodeStruct *node, UNU int pin, UNU int mode)  { return ; }
static         void pullUpDnControlDummy     (UNU struct wiringPiNodeStruct *node, UNU int pin, UNU int pud)   { return ; }
//static unsigned int digitalRead8Dummy        (UNU struct wiringPiNodeStruct *node, UNU int UNU pin)            { return 0 ; }
//static         void digitalWrite8Dummy       (UNU struct wiringPiNodeStruct *node, UNU int pin, UNU int value) { return ; }
static          int digitalReadDummy         (UNU struct wiringPiNodeStruct *node, UNU int UNU pin)            { return LOW ; }
static         void digitalWriteDummy        (UNU struct wiringPiNodeStruct *node, UNU int pin, UNU int value) { return ; }
static         void pwmWriteDummy            (UNU struct wiringPiNodeStruct *node, UNU int pin, UNU int value) { return ; }
static          int analogReadDummy          (UNU struct wiringPiNodeStruct *node, UNU int pin)            { return 0 ; }
static         void analogWriteDummy         (UNU struct wiringPiNodeStruct *node, UNU int pin, UNU int value) { return ; }

struct wiringPiNodeStruct *wiringPiNewNode (int pinBase, int numPins)
{
  int    pin ;
  struct wiringPiNodeStruct *node ;

// Minimum pin base is 64

  if (pinBase < 64)
    (void)wiringPiFailure (WPI_FATAL, "wiringPiNewNode: pinBase of %d is < 64\n", pinBase) ;

// Check all pins in-case there is overlap:

  for (pin = pinBase ; pin < (pinBase + numPins) ; ++pin)
    if (wiringPiFindNode (pin) != NULL)
      (void)wiringPiFailure (WPI_FATAL, "wiringPiNewNode: Pin %d overlaps with existing definition\n", pin) ;

  node = (struct wiringPiNodeStruct *)calloc (sizeof (struct wiringPiNodeStruct), 1) ;	// calloc zeros
  if (node == NULL)
    (void)wiringPiFailure (WPI_FATAL, "wiringPiNewNode: Unable to allocate memory: %s\n", strerror (errno)) ;

  node->pinBase          = pinBase ;
  node->pinMax           = pinBase + numPins - 1 ;
  node->pinMode          = pinModeDummy ;
  node->pullUpDnControl  = pullUpDnControlDummy ;
  node->digitalRead      = digitalReadDummy ;
//node->digitalRead8     = digitalRead8Dummy ;
  node->digitalWrite     = digitalWriteDummy ;
//node->digitalWrite8    = digitalWrite8Dummy ;
  node->pwmWrite         = pwmWriteDummy ;
  node->analogRead       = analogReadDummy ;
  node->analogWrite      = analogWriteDummy ;
  node->next             = wiringPiNodes ;
  wiringPiNodes          = node ;

  return node ;
}


#ifdef notYetReady
/*
 * pinED01:
 * pinED10:
 *	Enables edge-detect mode on a pin - from a 0 to a 1 or 1 to 0
 *	Pin must already be in input mode with appropriate pull up/downs set.
 *********************************************************************************
 */

void pinEnableED01Pi (int pin)
{
  pin = pinToGpio [pin & 63] ;
}
#endif


/*
 *********************************************************************************
 * Core Functions
 *********************************************************************************
 */

/*
 * pinModeAlt:
 *	This is an un-documented special to let you set any pin to any mode
 *********************************************************************************
 */

void pinModeAlt (int pin, int mode)
{
  int fSel, shift ;

  setupCheck ("pinModeAlt") ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-board pin
  {
    /**/ if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin] ;
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    fSel  = gpioToGPFSEL [pin] ;
    shift = gpioToShift  [pin] ;

    *(gpio + fSel) = (*(gpio + fSel) & ~(7 << shift)) | ((mode & 0x7) << shift) ;
  }
}


/*
 * pinMode:
 *	Sets the mode of a pin to be input, output or PWM output
 *********************************************************************************
 */

void pinMode (int pin, int mode)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;
  int origPin = pin ;
  struct xj3PwmInfoStruct pwminfo;

  setupCheck ("pinMode") ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-board pin
  {
    /**/ if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin] ;
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    softPwmStop  (origPin) ;
    softToneStop (origPin) ;

    // fSel    = gpioToGPFSEL [pin] ;
    // shift   = gpioToShift  [pin] ;

    /**/ if (mode == INPUT)
      xj3_gpio_direction_input(pin);
    else if (mode == OUTPUT)
      xj3_gpio_direction_output(pin);
    else if (mode == SOFT_PWM_OUTPUT)
      softPwmCreate (origPin, 0, 100) ;
    else if (mode == SOFT_TONE_OUTPUT)
      softToneCreate (origPin) ;
    else if (mode == PWM_TONE_OUTPUT)
    {
      pinMode (origPin, PWM_OUTPUT) ;	// Call myself to enable PWM mode
    }
    else if (mode == PWM_OUTPUT)
    {
      if (xj3GetPwmInfo(pin, &pwminfo) == -1)	// Not a hardware capable PWM pin
        return ;

      // Set pin to PWM mode

      xj3_set_pin_mux_pwm(pin, pwminfo.pin_mux_val);
      delayMicroseconds (110) ;		// See comments in pwmSetClockWPi

      xj3PwmExport (pwminfo) ;
      xj3PwmSetFreq (pwminfo, 48000) ;		// Default frequency is 48KHz
      xj3PwmSetDutyCycle (pwminfo, 50) ;  // 50%
      xj3PwmEnable(pwminfo, 1);
    }
  }
  else
  {
    if ((node = wiringPiFindNode (pin)) != NULL)
      node->pinMode (node, pin, mode) ;
    return ;
  }
}


/*
 * pullUpDownCtrl:
 *	Control the internal pull-up/down resistors on a GPIO pin.
 *********************************************************************************
 */

void pullUpDnControl (int pin, int pud)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;

  setupCheck ("pullUpDnControl") ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
  {
    /**/ if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin] ;
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    xj3_set_gpio_pull(pin, pud);
  }
  else						// Extension module
  {
    if ((node = wiringPiFindNode (pin)) != NULL)
      node->pullUpDnControl (node, pin, pud) ;
    return ;
  }
}

/*
 * digitalRead:
 *	Read the value of a given Pin, returning HIGH or LOW
 *********************************************************************************
 */

int digitalRead (int pin)
{
  unsigned int c = 0 ;
  // struct wiringPiNodeStruct *node = wiringPiNodes ;
  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
  {
    /**/ if (wiringPiMode == WPI_MODE_GPIO_SYS)	// Sys mode
    {
      if (pinToGpio[pin] == -1)
        return NOTD ;
      pin = pinToGpio[pin] ;
      if (gpio_export(pin))
        return LOW ;
      gpio_get_value(pin, &c) ;
      gpio_unexport(pin);
      return (c == 0) ? LOW : HIGH ;
    }
    else if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin];
    else if (wiringPiMode != WPI_MODE_GPIO)
      return NOTD ;

    xj3_gpio_direction_input(pin);

    if (xj3_get_pin_value(pin) != 0)
      return HIGH ;
    else
      return LOW ;
  }
  else
  {
      return NOTD ;
  }
}

/*
 * digitalWrite:
 *	Set an output bit
 *********************************************************************************
 */

void digitalWrite (int pin, int value)
{
  // struct wiringPiNodeStruct *node = wiringPiNodes ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
  {
    /**/ if (wiringPiMode == WPI_MODE_GPIO_SYS)	// Sys mode
    {
      if (pinToGpio[pin] != -1)
      {
        pin = pinToGpio[pin] ;
        if (gpio_export(pin))
          return;
        if (value == LOW)
          gpio_set_value(pin, 0) ;
        else
          gpio_set_value(pin, 1) ;
        gpio_unexport(pin);
      }
      return ;
    }
    else if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin];
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    xj3_gpio_direction_output(pin);

    if (value == LOW)
      xj3_set_pin_value(pin, 0);
    else
      xj3_set_pin_value(pin, 1);
  }
}


/*
 * digitalWrite8:
 *	Set an output 8-bit byte on the device from the given pin number
 *********************************************************************************

void digitalWrite8 (int pin, int value)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
    return ;
  else
  {
    if ((node = wiringPiFindNode (pin)) != NULL)
      node->digitalWrite8 (node, pin, value) ;
  }
}
 */


/*
 * pwmWrite:
 *	Set an output PWM value
 *********************************************************************************
 */

void pwmWrite (int pin, int value)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;

  setupCheck ("pwmWrite") ;

  if ((pin & PI_GPIO_MASK) == 0)		// On-Board Pin
  {
    /**/ if (wiringPiMode == WPI_MODE_PINS)
      pin = pinToGpio [pin] ;
    else if (wiringPiMode == WPI_MODE_PHYS)
      pin = physToGpio [pin] ;
    else if (wiringPiMode != WPI_MODE_GPIO)
      return ;

    *(pwm + gpioToPwmPort [pin]) = value ;
  }
  else
  {
    if ((node = wiringPiFindNode (pin)) != NULL)
      node->pwmWrite (node, pin, value) ;
  }
}


/*
 * analogRead:
 *	Read the analog value of a given Pin.
 *	There is no on-board Pi analog hardware,
 *	so this needs to go to a new node.
 *********************************************************************************
 */

int analogRead (int pin)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;

  if ((node = wiringPiFindNode (pin)) == NULL)
    return 0 ;
  else
    return node->analogRead (node, pin) ;
}


/*
 * analogWrite:
 *	Write the analog value to the given Pin.
 *	There is no on-board Pi analog hardware,
 *	so this needs to go to a new node.
 *********************************************************************************
 */

void analogWrite (int pin, int value)
{
  struct wiringPiNodeStruct *node = wiringPiNodes ;

  if ((node = wiringPiFindNode (pin)) == NULL)
    return ;

  node->analogWrite (node, pin, value) ;
}


/*
 * pwmToneWrite:
 *	Pi Specific.
 *      Output the given frequency on the Pi's PWM pin
 *********************************************************************************
 */

void pwmToneWrite (int pin, int freq)
{
  int range ;

  setupCheck ("pwmToneWrite") ;

  if (freq == 0)
    pwmWrite (pin, 0) ;             // Off
  else
  {
    range = 600000 / freq ;
    pwmSetRange (range) ;
    pwmWrite    (pin, freq / 2) ;
  }
}



/*
 * digitalWriteByte:
 * digitalReadByte:
 *	Pi Specific
 *	Write an 8-bit byte to the first 8 GPIO pins - try to do it as
 *	fast as possible.
 *	However it still needs 2 operations to set the bits, so any external
 *	hardware must not rely on seeing a change as there will be a change
 *	to set the outputs bits to zero, then another change to set the 1's
 *	Reading is just bit fiddling.
 *	These are wiringPi pin numbers 0..7, or BCM_GPIO pin numbers
 *	17, 18, 22, 23, 24, 24, 4 on a Pi v1 rev 0-3
 *	17, 18, 27, 23, 24, 24, 4 on a Pi v1 rev 3 onwards or B+, 2, 3, zero
 *********************************************************************************
 */

void digitalWriteByte (const int value)
{
  // uint32_t pinSet = 0 ;
  // uint32_t pinClr = 0 ;
  int mask = 1 ;
  int pin ;
  for (pin = 0 ; pin < 8 ; ++pin)
  {
    digitalWrite (pin, value & mask) ;
    mask <<= 1 ;
  }
  return ;
}

unsigned int digitalReadByte (void)
{
  int pin, x ;
  // uint32_t raw ;
  uint32_t data = 0 ;

  for (pin = 0 ; pin < 8 ; ++pin)
  {
    x = digitalRead (pin) ;
    data = (data << 1) | x ;
  }
  return data ;
}


/*
 * digitalWriteByte2:
 * digitalReadByte2:
 *	Pi Specific
 *	Write an 8-bit byte to the second set of 8 GPIO pins. This is marginally
 *	faster than the first lot as these are consecutive BCM_GPIO pin numbers.
 *	However they overlap with the original read/write bytes.
 *********************************************************************************
 */

void digitalWriteByte2 (const int value)
{
  register int mask = 1 ;
  register int pin ;

  /**/ if (wiringPiMode == WPI_MODE_GPIO_SYS)
  {
    for (pin = 20 ; pin < 28 ; ++pin)
    {
      digitalWrite (pin, value & mask) ;
      mask <<= 1 ;
    }
    return ;
  }
  else
  {
    *(gpio + gpioToGPCLR [0]) = (~value & 0xFF) << 20 ; // 0x0FF00000; ILJ > CHANGE: Old causes glitch
    *(gpio + gpioToGPSET [0]) = ( value & 0xFF) << 20 ;
  }
}

unsigned int digitalReadByte2 (void)
{
  int pin, x ;
  uint32_t data = 0 ;

  /**/ if (wiringPiMode == WPI_MODE_GPIO_SYS)
  {
    for (pin = 20 ; pin < 28 ; ++pin)
    {
      x = digitalRead (pin) ;
      data = (data << 1) | x ;
    }
  }
  else
    data = ((*(gpio + gpioToGPLEV [0])) >> 20) & 0xFF ; // First bank for these pins

  return data ;
}


/*
 * waitForInterrupt:
 *	Pi Specific.
 *	Wait for Interrupt on a GPIO pin.
 *	This is actually done via the /sys/class/gpio interface regardless of
 *	the wiringPi access mode in-use. Maybe sometime it might get a better
 *	way for a bit more efficiency.
 *********************************************************************************
 */

int waitForInterrupt (int pin, int mS)
{
  int fd, x ;
  char fName [64] ;
  uint8_t c ;
  struct pollfd polls ;

  /**/ if (wiringPiMode == WPI_MODE_PINS)
    pin = pinToGpio [pin] ;
  else if (wiringPiMode == WPI_MODE_PHYS)
    pin = physToGpio [pin] ;

  if ((fd = sysFds [pin]) == -1)
  {
    sprintf (fName, "/sys/class/gpio/gpio%d/value", pin) ;
    if ((sysFds [pin] = open (fName, O_RDWR)) < 0)
      return -2;

    fd = sysFds [pin];
  }

// Setup poll structure

  polls.fd     = fd ;
  polls.events = POLLPRI | POLLERR ;

// Wait for it ...

  // printf("Wait for it ... fName: %s pin=%d\n", fName, pin);

  x = poll (&polls, 1, mS) ;

// If no error, do a dummy read to clear the interrupt
//	A one character read appars to be enough.

  if (x > 0)
  {
    lseek (fd, 0, SEEK_SET) ;	// Rewind
    (void)read (fd, &c, 1) ;	// Read & clear
  }

  return x ;
}


/*
 * interruptHandler:
 *	This is a thread and gets started to wait for the interrupt we're
 *	hoping to catch. It will call the user-function when the interrupt
 *	fires.
 *********************************************************************************
 */

static void *interruptHandler (UNU void *arg)
{
  int myPin ;

  (void)piHiPri (55) ;	// Only effective if we run as root

  myPin   = pinPass ;
  pinPass = -1 ;

  for (;;)
    if (waitForInterrupt (myPin, -1) > 0)
      isrFunctions [myPin] () ;

  return NULL ;
}


/*
 * wiringPiISR:
 *	Pi Specific.
 *	Take the details and create an interrupt handler that will do a call-
 *	back to the user supplied function.
 *********************************************************************************
 */

int wiringPiISR (int pin, int mode, void (*function)(void))
{
  pthread_t threadId ;
  const char *modeS ;
  char fName   [64] ;
  char  pinS [8] ;
  pid_t pid ;
  int   count, i ;
  char  c ;
  int   bcmGpioPin ;

  if ((pin < 0) || (pin > 63))
    return wiringPiFailure (WPI_FATAL, "wiringPiISR: pin must be 0-63 (%d)\n", pin) ;

  /**/ if (wiringPiMode == WPI_MODE_UNINITIALISED)
    return wiringPiFailure (WPI_FATAL, "wiringPiISR: wiringPi has not been initialised. Unable to continue.\n") ;
  else if (wiringPiMode == WPI_MODE_PINS)
    bcmGpioPin = pinToGpio [pin] ;
  else if (wiringPiMode == WPI_MODE_PHYS)
    bcmGpioPin = physToGpio [pin] ;
  else
    bcmGpioPin = pin ;

// Now export the pin and set the right edge
//	We're going to use the gpio program to do this, so it assumes
//	a full installation of wiringPi. It's a bit 'clunky', but it
//	is a way that will work when we're running in "Sys" mode, as
//	a non-root user. (without sudo)

  if (mode != INT_EDGE_SETUP)
  {
    /**/ if (mode == INT_EDGE_FALLING)
      modeS = "falling" ;
    else if (mode == INT_EDGE_RISING)
      modeS = "rising" ;
    else
      modeS = "both" ;

    sprintf (pinS, "%d", bcmGpioPin) ;

    if ((pid = fork ()) < 0)	// Fail
      return wiringPiFailure (WPI_FATAL, "wiringPiISR: fork failed: %s\n", strerror (errno)) ;

    if (pid == 0)	// Child, exec
    {
      /**/ if (access ("/usr/local/bin/gpio", X_OK) == 0)
      {
	execl ("/usr/local/bin/gpio", "gpio", "edge", pinS, modeS, (char *)NULL) ;
	return wiringPiFailure (WPI_FATAL, "wiringPiISR: execl failed: %s\n", strerror (errno)) ;
      }
      else if (access ("/usr/bin/gpio", X_OK) == 0)
      {
	execl ("/usr/bin/gpio", "gpio", "edge", pinS, modeS, (char *)NULL) ;
	return wiringPiFailure (WPI_FATAL, "wiringPiISR: execl failed: %s\n", strerror (errno)) ;
      }
      else
	return wiringPiFailure (WPI_FATAL, "wiringPiISR: Can't find gpio program\n") ;
    }
    else		// Parent, wait
      waitpid (pid, NULL, 0) ;
  }

// Now pre-open the /sys/class node - but it may already be open if
//	we are in Sys mode...

  if (sysFds [bcmGpioPin] == -1)
  {
    sprintf (fName, "/sys/class/gpio/gpio%d/value", bcmGpioPin) ;
    if ((sysFds [bcmGpioPin] = open (fName, O_RDWR)) < 0)
      return wiringPiFailure (WPI_FATAL, "wiringPiISR: unable to open %s: %s\n", fName, strerror (errno)) ;
  }

// Clear any initial pending interrupt

  ioctl (sysFds [bcmGpioPin], FIONREAD, &count) ;
  for (i = 0 ; i < count ; ++i)
    read (sysFds [bcmGpioPin], &c, 1) ;

  isrFunctions [pin] = function ;

  pthread_mutex_lock (&pinMutex) ;
    pinPass = pin ;
    pthread_create (&threadId, NULL, interruptHandler, NULL) ;
    while (pinPass != -1)
      delay (1) ;
  pthread_mutex_unlock (&pinMutex) ;

  return 0 ;
}


/*
 * initialiseEpoch:
 *	Initialise our start-of-time variable to be the current unix
 *	time in milliseconds and microseconds.
 *********************************************************************************
 */

static void initialiseEpoch (void)
{
#ifdef	OLD_WAY
  struct timeval tv ;

  gettimeofday (&tv, NULL) ;
  epochMilli = (uint64_t)tv.tv_sec * (uint64_t)1000    + (uint64_t)(tv.tv_usec / 1000) ;
  epochMicro = (uint64_t)tv.tv_sec * (uint64_t)1000000 + (uint64_t)(tv.tv_usec) ;
#else
  struct timespec ts ;

  clock_gettime (CLOCK_MONOTONIC_RAW, &ts) ;
  epochMilli = (uint64_t)ts.tv_sec * (uint64_t)1000    + (uint64_t)(ts.tv_nsec / 1000000L) ;
  epochMicro = (uint64_t)ts.tv_sec * (uint64_t)1000000 + (uint64_t)(ts.tv_nsec /    1000L) ;
#endif
}


/*
 * delay:
 *	Wait for some number of milliseconds
 *********************************************************************************
 */

void delay (unsigned int howLong)
{
  struct timespec sleeper, dummy ;

  sleeper.tv_sec  = (time_t)(howLong / 1000) ;
  sleeper.tv_nsec = (long)(howLong % 1000) * 1000000 ;

  nanosleep (&sleeper, &dummy) ;
}


/*
 * delayMicroseconds:
 *	This is somewhat intersting. It seems that on the Pi, a single call
 *	to nanosleep takes some 80 to 130 microseconds anyway, so while
 *	obeying the standards (may take longer), it's not always what we
 *	want!
 *
 *	So what I'll do now is if the delay is less than 100uS we'll do it
 *	in a hard loop, watching a built-in counter on the ARM chip. This is
 *	somewhat sub-optimal in that it uses 100% CPU, something not an issue
 *	in a microcontroller, but under a multi-tasking, multi-user OS, it's
 *	wastefull, however we've no real choice )-:
 *
 *      Plan B: It seems all might not be well with that plan, so changing it
 *      to use gettimeofday () and poll on that instead...
 *********************************************************************************
 */

void delayMicrosecondsHard (unsigned int howLong)
{
  struct timeval tNow, tLong, tEnd ;

  gettimeofday (&tNow, NULL) ;
  tLong.tv_sec  = howLong / 1000000 ;
  tLong.tv_usec = howLong % 1000000 ;
  timeradd (&tNow, &tLong, &tEnd) ;

  while (timercmp (&tNow, &tEnd, <))
    gettimeofday (&tNow, NULL) ;
}

void delayMicroseconds (unsigned int howLong)
{
  struct timespec sleeper ;
  unsigned int uSecs = howLong % 1000000 ;
  unsigned int wSecs = howLong / 1000000 ;

  /**/ if (howLong ==   0)
    return ;
  else if (howLong  < 100)
    delayMicrosecondsHard (howLong) ;
  else
  {
    sleeper.tv_sec  = wSecs ;
    sleeper.tv_nsec = (long)(uSecs * 1000L) ;
    nanosleep (&sleeper, NULL) ;
  }
}


/*
 * millis:
 *	Return a number of milliseconds as an unsigned int.
 *	Wraps at 49 days.
 *********************************************************************************
 */

unsigned int millis (void)
{
  uint64_t now ;

#ifdef	OLD_WAY
  struct timeval tv ;

  gettimeofday (&tv, NULL) ;
  now  = (uint64_t)tv.tv_sec * (uint64_t)1000 + (uint64_t)(tv.tv_usec / 1000) ;

#else
  struct  timespec ts ;

  clock_gettime (CLOCK_MONOTONIC_RAW, &ts) ;
  now  = (uint64_t)ts.tv_sec * (uint64_t)1000 + (uint64_t)(ts.tv_nsec / 1000000L) ;
#endif

  return (uint32_t)(now - epochMilli) ;
}


/*
 * micros:
 *	Return a number of microseconds as an unsigned int.
 *	Wraps after 71 minutes.
 *********************************************************************************
 */

unsigned int micros (void)
{
  uint64_t now ;
#ifdef	OLD_WAY
  struct timeval tv ;

  gettimeofday (&tv, NULL) ;
  now  = (uint64_t)tv.tv_sec * (uint64_t)1000000 + (uint64_t)tv.tv_usec ;
#else
  struct  timespec ts ;

  clock_gettime (CLOCK_MONOTONIC_RAW, &ts) ;
  now  = (uint64_t)ts.tv_sec * (uint64_t)1000000 + (uint64_t)(ts.tv_nsec / 1000) ;
#endif


  return (uint32_t)(now - epochMicro) ;
}

/*
 * wiringPiVersion:
 *	Return our current version number
 *********************************************************************************
 */

void wiringPiVersion (int *major, int *minor)
{
  *major = VERSION_MAJOR ;
  *minor = VERSION_MINOR ;
}


/*
 * wiringPiSetup:
 *	Must be called once at the start of your program execution.
 *
 * Default setup: Initialises the system into wiringPi Pin mode and uses the
 *	memory mapped hardware directly.
 *
 * Changed now to revert to "gpio" mode if we're running on a Compute Module.
 *********************************************************************************
 */

int wiringPiSetup (void)
{
  int   fd ;
  int   model, rev, mem, maker, overVolted ;
  int   gpioLayout = 0;

  if (wiringPiSetuped)
    return 0 ;

  wiringPiSetuped = TRUE ;

  if (getenv (ENV_DEBUG) != NULL)
    wiringPiDebug = TRUE ;

  if (getenv (ENV_CODES) != NULL)
    wiringPiReturnCodes = TRUE ;

  if (wiringPiDebug)
    printf ("wiringPi: wiringPiSetup called\n") ;

// Get the board ID information. We're not really using the information here,
//	but it will give us information like the GPIO layout scheme (2 variants
//	on the older 26-pin Pi's) and the GPIO peripheral base address.
//	and if we're running on a compute module, then wiringPi pin numbers
//	don't really many anything, so force native BCM mode anyway.

  piBoardId (&model, &rev, &mem, &maker, &overVolted) ;

  if ((model == PI_MODEL_CM) ||
      (model == PI_MODEL_CM3) ||
      (model == PI_MODEL_CM3P))
    wiringPiMode = WPI_MODE_GPIO ;
  else
    wiringPiMode = WPI_MODE_PINS ;

  gpioLayout = piGpioLayout ();
  /**/ if (gpioLayout == 1)	// A, B, Rev 1, 1.1
  {
     pinToGpio =  pinToGpioR1 ;
    physToGpio = physToGpioR1 ;
  }
  else if (gpioLayout == 5 || gpioLayout == 6)
  {
     pinToGpio =  pinToGpioRdkX3 ;
    physToGpio = physToGpioRdkX3 ;
  }
  else if (gpioLayout == 8 || gpioLayout == 0xb)
  {
     pinToGpio =  pinToGpioRdkX3V2 ;
    physToGpio = physToGpioRdkX3V2 ;
  }
  else 					// A2, B2, A+, B+, CM, Pi2, Pi3, Zero, Zero W, Zero 2 W
  {
     pinToGpio =  pinToGpioR2 ;
    physToGpio = physToGpioR2 ;
  }

// ...

  switch (model)
  {
    case PI_MODEL_RDKX3:
      piGpioBase = 0xA6003000;
      piGpioPinMuxBase = 0xA6004000;
      break;
    default:
      piGpioBase = 0xA6003000;
      piGpioPinMuxBase = 0xA6004000;
      break;
  }

  if ((fd = open ("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC)) < 0)
  {
    if ((fd = open ("/dev/gpiomem", O_RDWR | O_SYNC | O_CLOEXEC) ) >= 0)	// We're using gpiomem
    {
      piGpioBase   = 0 ;
      usingGpioMem = TRUE ;
    }
    else
      return wiringPiFailure (WPI_ALMOST, "wiringPiSetup: Unable to open /dev/mem or /dev/gpiomem: %s.\n"
	"  Aborting your program because if it can not access the GPIO\n"
	"  hardware then it most certianly won't work\n"
	"  Try running with sudo?\n", strerror (errno)) ;
  }

// Set the offsets into the memory interface.
  GPIO_BASE	  = piGpioBase ;

// Map the individual hardware components

//	GPIO:
  off_t base = 0;
  off_t offset = 0;
  base = GPIO_BASE / 0x10000 * 0x10000;
  offset = GPIO_BASE % 0x10000;
  gpio = (uint32_t *)mmap(0, 16 * BLOCK_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, base) ;
  if (gpio == MAP_FAILED)
    return wiringPiFailure (WPI_ALMOST, "wiringPiSetup: mmap (GPIO) failed: %s\n", strerror (errno)) ;

// Export the base addresses for any external software that might need them

  _wiringPiGpio = (volatile unsigned int *)(gpio + offset/sizeof(uint32_t));
  _wiringPiGpioPinMux = (volatile unsigned int *)(gpio + 0x4000/sizeof(uint32_t));

  // printf("GPIO_BASE = %x, gpio = %x, _wiringPiGpio=%x\n", GPIO_BASE, gpio, _wiringPiGpio);

  initialiseEpoch () ;
  return 0 ;
}


/*
 * wiringPiSetupGpio:
 *	Must be called once at the start of your program execution.
 *
 * GPIO setup: Initialises the system into GPIO Pin mode and uses the
 *	memory mapped hardware directly.
 *********************************************************************************
 */

int wiringPiSetupGpio (void)
{
  (void)wiringPiSetup () ;

  if (wiringPiDebug)
    printf ("wiringPi: wiringPiSetupGpio called\n") ;

  wiringPiMode = WPI_MODE_GPIO ;

  return 0 ;
}


/*
 * wiringPiSetupPhys:
 *	Must be called once at the start of your program execution.
 *
 * Phys setup: Initialises the system into Physical Pin mode and uses the
 *	memory mapped hardware directly.
 *********************************************************************************
 */

int wiringPiSetupPhys (void)
{
  (void)wiringPiSetup () ;

  if (wiringPiDebug)
    printf ("wiringPi: wiringPiSetupPhys called\n") ;

  wiringPiMode = WPI_MODE_PHYS ;

  return 0 ;
}


/*
 * wiringPiSetupSys:
 *	Must be called once at the start of your program execution.
 *
 * Initialisation (again), however this time we are using the /sys/class/gpio
 *	interface to the GPIO systems - slightly slower, but always usable as
 *	a non-root user, assuming the devices are already exported and setup correctly.
 */

int wiringPiSetupSys (void)
{
  // int pin ;
  // char fName [128] ;
  int gpioLayout = 0;

  if (wiringPiSetuped)
    return 0 ;

  wiringPiSetuped = TRUE ;

  if (getenv (ENV_DEBUG) != NULL)
    wiringPiDebug = TRUE ;

  if (getenv (ENV_CODES) != NULL)
    wiringPiReturnCodes = TRUE ;

  if (wiringPiDebug)
    printf ("wiringPi: wiringPiSetupSys called\n") ;

  gpioLayout = piGpioLayout ();
  /**/ if (gpioLayout == 1)	// A, B, Rev 1, 1.1
  {
     pinToGpio =  pinToGpioR1 ;
    physToGpio = physToGpioR1 ;
  }
  else if (gpioLayout == 5 || gpioLayout == 6)
  {
     pinToGpio =  pinToGpioRdkX3 ;
    physToGpio = physToGpioRdkX3 ;
  }
  else if (gpioLayout == 8 || gpioLayout == 0xb)
  {
     pinToGpio =  pinToGpioRdkX3V2 ;
    physToGpio = physToGpioRdkX3V2 ;
  }
  else 					// A2, B2, A+, B+, CM, Pi2, Pi3, Zero, Zero W, Zero 2 W
  {
     pinToGpio =  pinToGpioR2 ;
    physToGpio = physToGpioR2 ;
  }

// Open and scan the directory, looking for exported GPIOs, and pre-open
//	the 'value' interface to speed things up for later

  // for (pin = 0 ; pin < 64 ; ++pin)
  // {
  //   if (pinToGpio[pin] != -1)
  //   {
  //     sprintf (fName, "/sys/class/gpio/gpio%d/value", pinToGpio[pin]) ;
  //     sysFds [pin] = open (fName, O_RDWR) ;
  //   }
  // }

  initialiseEpoch () ;

  wiringPiMode = WPI_MODE_GPIO_SYS ;

  return 0 ;
}
