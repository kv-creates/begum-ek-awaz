# Wiring

All connections are 3.3 V logic. Power the board from USB or a 5 V bank.
No network wiring exists by design.

## Full schematic (ASCII)

```text
                    +-------------------+
                    |   ESP8266 D1 mini |
                    |                   |
  USB 5V ===========| 5V          3V3   |====+===+====
                    |                   |    |   |
                    |  GND          D1/GPIO5 |   |    .----------.
                    |                   |    |   +----| VCC      |
                    |  D2/GPIO4     D5/GPIO14|--------| SCL  SSD |
                    |                   |    |   +----| GND  1306|
                    |  D6/GPIO12    D7/GPIO13|   |    | SDA 0.96"|
                    |                   |    |   |    '----------'
                    |  A0           GPIO2   LED (built-in, active LOW)
                    |                   |
                    +-------------------+
                         |   |   |   |
      INMP441 I2S mic    |   |   |   |   OLED 0.96" I2C (0x3C)
      -------------      |   |   |   |   --------------------
      VDD ---------------+   |   |   +--- VCC (3V3)
      GND -------------------+   |   +--- GND
      L/R (to GND, left) ----|   |
      SD/DOUT ---------------+   |
      WS/LRCLK ------------------+
      SCK/BCLK ------------------+
```

## Pin table

| Net          | ESP8266 pin | Direction | Notes                          |
|--------------|-------------|-----------|--------------------------------|
| MIC VDD      | 3V3         | power     | INMP441 supply                 |
| MIC GND      | GND         | ground    | star-ground with OLED          |
| MIC L/R      | GND         | strap     | left-channel mono              |
| MIC DOUT     | D6 / GPIO12 | in        | 24-bit MSB-first, left slot    |
| MIC LRCLK    | D7 / GPIO13 | out       | 16 kHz word clock (bit-banged) |
| MIC BCLK     | D5 / GPIO14 | out       | ~500 kHz (bit-banged)          |
| OLED VCC     | 3V3         | power     | 0.96" SSD1306                  |
| OLED GND     | GND         | ground    | common ground                  |
| OLED SCL     | D1 / GPIO5  | out       | HW I2C, 400 kHz                |
| OLED SDA     | D2 / GPIO4  | bidir     | HW I2C                         |
| LED          | GPIO2       | out       | built-in, LOW = ON, 3 s pulse  |
| ALT MIC      | A0          | in        | MAX9814 OUT (USE_ANALOG_MIC)   |

## I2C bus

```text
 3V3 ----+---- VCC (OLED)
         |
 D1 -----+---- SCL (OLED)   400 kHz, internal pull-ups + OLED module pull-ups
 D2 -----+---- SDA (OLED)   address 0x3C, 128x64 page-buffer driver
 GND ----+---- GND (OLED)
```

Constructor: `U8G2_SSD1306_128X64_NONAME_1_HW_I2C` (the `_1_` page buffer
variant: 128-byte RAM, fits the ESP8266 budget).

## Analog fallback

Uncomment `USE_ANALOG_MIC` in `firmware/src/config.h`, then:

```text
 MAX9814 VCC -> 5V/USB, GND -> GND, OUT -> A0, GAIN -> VCC (max)
 ADC: 10-bit, centered at 512, scaled x64 to int16, paced at 62 us.
```
