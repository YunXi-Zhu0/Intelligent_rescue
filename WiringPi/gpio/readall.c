/*
 * readall.c:
 *	The readall functions - getting a bit big, so split them out.
 *	Copyright (c) 2012-2018 Gordon Henderson
 ***********************************************************************
 * This file is part of wiringPi:
 *	https://github.com/WiringPi/WiringPi/
 *
 *    wiringPi is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    wiringPi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with wiringPi.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <wiringPi.h>

extern int wpMode ;

#ifndef TRUE
#  define       TRUE    (1==1)
#  define       FALSE   (1==2)
#endif

static int *physToGpio ;
static char **physToNames ;

/*
 * doReadallExternal:
 *	A relatively crude way to read the pins on an external device.
 *	We don't know the input/output mode of pins, but we can tell
 *	if it's an analog pin or a digital one...
 *********************************************************************************
 */

static void doReadallExternal (void)
{
  int pin ;

  printf ("+------+---------+--------+\n") ;
  printf ("|  Pin | Digital | Analog |\n") ;
  printf ("+------+---------+--------+\n") ;

  for (pin = wiringPiNodes->pinBase ; pin <= wiringPiNodes->pinMax ; ++pin)
    printf ("| %4d |  %4d   |  %4d  |\n", pin, digitalRead (pin), analogRead (pin)) ;

  printf ("+------+---------+--------+\n") ;
}


/*
 * doReadall:
 *	Read all the GPIO pins
 *	We also want to use this to read the state of pins on an externally
 *	connected device, so we need to do some fiddling with the internal
 *	wiringPi node structures - since the gpio command can only use
 *	one external device at a time, we'll use that to our advantage...
 *********************************************************************************
 */

static char *alts [] =
{
  "IN", "OUT", "ALT5", "ALT4", "ALT0", "ALT1", "ALT2", "ALT3"
} ;

static int physToWpi [64] =
{
  -1,           // 0
  -1, -1,       // 1, 2
   8, -1,
   9, -1,
   7, 15,
  -1, 16,
   0,  1,
   2, -1,
   3,  4,
  -1,  5,
  12, -1,
  13,  6,
  14, 10,
  -1, 11,       // 25, 26
  30, 31,	// Actually I2C, but not used
  21, -1,
  22, 26,
  23, -1,
  24, 27,
  25, 28,
  -1, 29,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  -1, -1,
  17, 18,
  19, 20,
  -1, -1, -1, -1, -1, -1, -1, -1, -1
} ;

static char *physNames [64] =
{
  NULL,

  "   3.3v", "5v     ",
  "  SDA.1", "5v     ",
  "  SCL.1", "0v     ",
  "GPIO. 7", "TxD    ",
  "     0v", "RxD    ",
  "GPIO. 0", "GPIO. 1",
  "GPIO. 2", "0v     ",
  "GPIO. 3", "GPIO. 4",
  "   3.3v", "GPIO. 5",
  "   MOSI", "0v     ",
  "   MISO", "GPIO. 6",
  "   SCLK", "CE0    ",
  "     0v", "CE1    ",
  "  SDA.0", "SCL.0  ",
  "GPIO.21", "0v     ",
  "GPIO.22", "GPIO.26",
  "GPIO.23", "0v     ",
  "GPIO.24", "GPIO.27",
  "GPIO.25", "GPIO.28",
  "     0v", "GPIO.29",
       NULL, NULL,
       NULL, NULL,
       NULL, NULL,
       NULL, NULL,
       NULL, NULL,
  "GPIO.17", "GPIO.18",
  "GPIO.19", "GPIO.20",
   NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
} ;

static int physToGpioRdkX3 [64] =
{
   -1,           // 0
   -1,  -1,       // 1, 2
    9,  -1,
    8,  -1,
  101, 111,
   -1, 112,
    6, 102,
    5,  -1,
   30,  27,
   -1,   7,
   12,  -1,
   13,  29,
   14,  15,
   -1,  28,       // 25, 26
  106, 107,
  119,  -1,
  118,  25,
    4,  -1,
  103,   3,
  105, 104,
   -1, 108,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1, -1, -1, -1, -1, -1, -1, -1
} ;

static char *physNamesRdkX3 [64] =
{
  NULL,

  "     3.3v", "5v       ",
  "    SDA.0", "5v       ",
  "    SCL.0", "0v       ",
  "I2S0_MCLK", "TxD.3    ",
  "       0v", "RxD.3    ",
  "  GPIO. 6", "I2S0_BCLK",
  "  GPIO. 5", "0v       ",
  " GPIO. 30", "GPIO. 27 ",
  "     3.3v", "GPIO. 7  ",
  "SPI2_MOSI", "0v       ",
  "SPI2_MISO", "GPIO. 29 ",
  "SPI2_SCLK", "SPI2_CSN ",
  "       0v", "GPIO. 28 ",
  "I2S1_BCLK", "I2S1_LRCK",
  " GPIO.119", "0v       ",
  " GPIO.118", "PWM4     ",
  "     PWM0", "0v       ",
  "I2S0_LRCK", "GPIO. 3  ",
  " GPIO.105", "I2S0_SDIO",
  "       0v", "I2S1_SDIO",
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
     NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
} ;


static int physToGpioRdkX3V2 [64] =
{
   -1,           // 0
   -1,  -1,       // 1, 2
    9,  -1,
    8,  -1,
  101, 111,
   -1, 112,
   12, 102,
   13,  -1,
   30,  27,
   -1,  22,
    6,  -1,
    7,  29,
    3,   5,
   -1,  28,       // 25, 26
   15,  14,
  119,  -1,
  118,  25,
    4,  -1,
  103,  20,
  117, 108,
   -1, 104,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1,
   -1,  -1, -1, -1, -1, -1, -1, -1, -1
} ;

static char *physNamesRdkX3V2 [64] =
{
  NULL,

  "     3.3v", "5v       ",
  "    SDA.0", "5v       ",
  "    SCL.0", "0v       ",
  "I2S0_MCLK", "TxD.3    ",
  "       0v", "RxD.3    ",
  " GPIO. 17", "I2S0_BCLK",
  " GPIO. 27", "0v       ",
  " GPIO. 22", "GPIO. 23 ",
  "     3.3v", "GPIO. 24 ",
  "SPI1_MOSI", "0v       ",
  "SPI1_MISO", "GPIO. 25 ",
  "SPI1_SCLK", "SPI1_CSN ",
  "       0v", "GPIO.  7 ",
  "    SDA.3", "SCL.3    ",
  "  GPIO. 5", "0v       ",
  "  GPIO. 6", "PWM4     ",
  "     PWM0", "0v       ",
  "I2S0_LRCK", "GPIO. 16 ",
  "  GPIO.26", "I2S1_SDIO",
  "       0v", "I2S0_SDIO",
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
         NULL, NULL,
     NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
} ;

/*
 * readallPhys:
 *	Given a physical pin output the data on it and the next pin:
 *| BCM | wPi |   Name  | Mode | Val| Physical |Val | Mode | Name    | wPi | BCM |
 *********************************************************************************
 */

static void readallPhys (int physPin)
{
  int pin ;

  if (physPinToGpio (physPin) == -1)
    printf (" |     |    ") ;
  else
    printf (" | %3d | %3d", physPinToGpio (physPin), physToGpio [physPin]) ;

  printf (" | %s", physToNames [physPin]) ;

  printf (" | %2d", physPin) ;
  ++physPin ;
  printf (" || %-2d", physPin) ;


  printf (" | %-5s", physToNames [physPin]) ;

  if (physPinToGpio (physPin) == -1)
    printf (" |     |    ") ;
  else
    printf (" | %-3d | %-3d", physToGpio [physPin], physPinToGpio (physPin)) ;

  printf (" |\n") ;
}


/*
 * allReadall:
 *	Read all the pins regardless of the model. Primarily of use for
 *	the compute module, but handy for other fiddling...
 *********************************************************************************
 */

static void allReadall (void)
{
  int pin ;
  int val ;

  printf ("+-----+-------+      +-----+-------+\n") ;
  printf ("| Pin | Value |      | Pin | Value |\n") ;
  printf ("+-----+-------+      +-----+-------+\n") ;

  for (pin = 0 ; pin < 20 ; ++pin)
  {
    printf ("| %3d ", pin) ;
    val = digitalRead (pin);
    printf ("| %s  ", val == NOTD ? " -- ": val == HIGH ? "High" : "Low ") ;
    printf ("|      ") ;
    printf ("| %3d ", pin+20) ;
    val = digitalRead (pin + 20);
    printf ("| %s  ", val == NOTD ? " -- ": val == HIGH ? "High" : "Low ") ;
    printf ("|\n") ;
  }

  printf ("+-----+-------+      +-----+-------+\n") ;

}


/*
 * abReadall:
 *	Read all the pins on the model A or B.
 *********************************************************************************
 */

void abReadall (int model, int rev)
{
  int pin ;
  char *type ;

  if (model == PI_MODEL_A)
    type = " A" ;
  else
    if (rev == PI_VERSION_2)
      type = "B2" ;
    else
      type = "B1" ;

  printf (" +-----+-----+---------+------+---+-Model %s-+---+------+---------+-----+-----+\n", type) ;
  printf (" | BCM | wPi |   Name  | Mode | V | Physical | V | Mode | Name    | wPi | BCM |\n") ;
  printf (" +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+\n") ;
  for (pin = 1 ; pin <= 26 ; pin += 2)
    readallPhys (pin) ;

  if (rev == PI_VERSION_2) // B version 2
  {
    printf (" +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+\n") ;
    for (pin = 51 ; pin <= 54 ; pin += 2)
      readallPhys (pin) ;
  }

  printf (" +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+\n") ;
  printf (" | BCM | wPi |   Name  | Mode | V | Physical | V | Mode | Name    | wPi | BCM |\n") ;
  printf (" +-----+-----+---------+------+---+-Model %s-+---+------+---------+-----+-----+\n", type) ;
}


/*
 * piPlusReadall:
 *	Read all the pins on the model A+ or the B+ or actually, all 40-pin Pi's
 *********************************************************************************
 */

static void plus2header (int model)
{
  if (model == PI_MODEL_AP)
    printf (" +-----+-----+---------+------+---+---Pi A+--+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_BP)
    printf (" +-----+-----+---------+------+---+---Pi B+--+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_ZERO)
    printf (" +-----+-----+---------+------+---+-Pi Zero--+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_ZERO_W)
    printf (" +-----+-----+---------+------+---+-Pi ZeroW-+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_ZERO_2W)
    printf (" +-----+-----+---------+------+---+Pi Zero 2W+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_2)
    printf (" +-----+-----+---------+------+---+---Pi 2---+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_3B)
    printf (" +-----+-----+---------+------+---+---Pi 3B--+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_3BP)
    printf (" +-----+-----+---------+------+---+---Pi 3B+-+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_3AP)
    printf (" +-----+-----+---------+------+---+---Pi 3A+-+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_4B)
    printf (" +-----+-----+---------+------+---+---Pi 4B--+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_400)
    printf (" +-----+-----+---------+------+---+---Pi 400-+---+------+---------+-----+-----+\n") ;
  else if (model == PI_MODEL_RDKX3)
    printf (" +-----+-----+-----------+--RDK X3--+-----------+-----+-----+\n") ;
  else if (model == PI_MODEL_RDKX3V1_2)
    printf (" +-----+-----+-----------+RDK X3v1.2+-----------+-----+-----+\n") ;
  else if (model == PI_MODEL_RDKX3V2)
    printf (" +-----+-----+-----------+-RDK X3v2-+-----------+-----+-----+\n") ;
  else if (model == PI_MODEL_RDKX3MD)
    printf (" +-----+-----+-----------+-RDK X3MD-+-----------+-----+-----+\n") ;
  else if (model == PI_MODEL_SDB)
    printf (" +-----+-----+-----------+--X3 SDB--+-----------+-----+-----+\n") ;
  else
    printf (" +-----+-----+---------+------+---+---Pi ?---+---+------+---------+-----+-----+\n") ;
}


static void piPlusReadall (int model)
{
  int pin ;

  plus2header (model) ;

  printf (" | BCM | wPi |   Name  | Mode | V | Physical | V | Mode | Name    | wPi | BCM |\n") ;
  printf (" +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+\n") ;
  for (pin = 1 ; pin <= 40 ; pin += 2)
    readallPhys (pin) ;
  printf (" +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+\n") ;
  printf (" | BCM | wPi |   Name  | Mode | V | Physical | V | Mode | Name    | wPi | BCM |\n") ;

  plus2header (model) ;
}

void x3Readall (int model, int rev)
{
  int pin ;

  plus2header (model) ;

  printf (" | BCM | xPi |    Name   | Physical |   Name    | xPi | BCM |\n") ;
  printf (" +-----+-----+-----------+----++----+-----------+-----+-----+\n") ;
  for (pin = 1 ; pin <= 40 ; pin += 2)
    readallPhys (pin) ;
  printf (" +-----+-----+-----------+----++----+-----------+-----+-----+\n") ;
  printf (" | BCM | xPi |    Name   | Physical |   Name    | xPi | BCM |\n") ;

  plus2header (model) ;
}


/*
 * doReadall:
 *	Generic read all pins called from main program. Works out the Pi type
 *	and calls the appropriate function.
 *********************************************************************************
 */

void doReadall (void)
{
  int model, rev, mem, maker, overVolted ;

  if (wiringPiNodes != NULL)	// External readall
  {
    doReadallExternal () ;
    return ;
  }

  piBoardId (&model, &rev, &mem, &maker, &overVolted) ;

  /**/ if ((model == PI_MODEL_A) || (model == PI_MODEL_B))
    abReadall (model, rev) ;
  else if ((model == PI_MODEL_BP) || (model == PI_MODEL_AP) ||
	(model == PI_MODEL_2)    ||
	(model == PI_MODEL_3AP)  ||
	(model == PI_MODEL_3B)   || (model == PI_MODEL_3BP) ||
	(model == PI_MODEL_4B)   || (model == PI_MODEL_400) || (model == PI_MODEL_CM4) ||
	(model == PI_MODEL_ZERO) || (model == PI_MODEL_ZERO_W) || (model == PI_MODEL_ZERO_2W)) {
    physToGpio = physToWpi;
    physToNames = physNames;
    piPlusReadall (model) ;
  } else if ((model == PI_MODEL_CM) || (model == PI_MODEL_CM3) || (model == PI_MODEL_CM3P) )
    allReadall () ;
  else if (model == PI_MODEL_RDKX3 || model == PI_MODEL_SDB || model == PI_MODEL_RDKX3V1_2) {
    physToGpio = physToGpioRdkX3;
    physToNames = physNamesRdkX3;
    x3Readall(model, rev);
  } else if (model == PI_MODEL_RDKX3V2 || model == PI_MODEL_RDKX3MD) {
    physToGpio = physToGpioRdkX3V2;
    physToNames = physNamesRdkX3V2;
    x3Readall(model, rev);
  } else
    printf ("Oops - unable to determine board type... model: %d\n", model) ;
}


/*
 * doAllReadall:
 *	Force reading of all pins regardless of Pi model
 *********************************************************************************
 */

void doAllReadall (void)
{
  allReadall () ;
}


/*
 * doQmode:
 *	Query mode on a pin
 *********************************************************************************
 */

void doQmode (int argc, char *argv [])
{
  int pin ;

  if (argc != 3)
  {
    fprintf (stderr, "Usage: %s qmode pin\n", argv [0]) ;
    exit (EXIT_FAILURE) ;
  }

  pin = atoi (argv [2]) ;
  printf ("%s\n", alts [getAlt (pin)]) ;
}
