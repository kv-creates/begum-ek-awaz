#ifndef BEGUM_OLED_DISPLAY_H
#define BEGUM_OLED_DISPLAY_H

// ============================================================================
// oled_display.h — U8g2 wrapper for 0.96" SSD1306 128x64 I2C.
// Wiring: VCC->3V3, GND->GND, SCL->GPIO5 (D1), SDA->GPIO4 (D2).
// States: Listening... / BEGUM DETECTED! + confidence / Cooldown... / status.
// ============================================================================

#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>
#include "config.h"

// Page-buffer constructor (lowest RAM: 128 bytes buffer, fits ESP8266).
static U8G2_SSD1306_128X64_NONAME_1_HW_I2C u8g2_oled(U8G2_R0, U8X8_PIN_NONE,
                                                    PIN_OLED_SCL, PIN_OLED_SDA);

inline void oled_init() {
  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
  Wire.setClock(400000L);
  u8g2_oled.begin();
  u8g2_oled.setContrast(200);
  u8g2_oled.setFont(u8g2_font_6x10_tf);
  u8g2_oled.firstPage();
  do {
    u8g2_oled.drawStr(0, 12, "BEGUM activator");
    u8g2_oled.drawStr(0, 26, "OLED: OK");
  } while (u8g2_oled.nextPage());
}

inline void oled_status(const char* line1, const char* line2) {
  u8g2_oled.firstPage();
  do {
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 12, "BEGUM");
    if (line1) u8g2_oled.drawStr(0, 28, line1);
    if (line2) u8g2_oled.drawStr(0, 42, line2);
  } while (u8g2_oled.nextPage());
}

inline void oled_listening() {
  u8g2_oled.firstPage();
  do {
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 12, "BEGUM");
    u8g2_oled.setFont(u8g2_font_10x20_tf);
    u8g2_oled.drawStr(0, 38, "Listening...");
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    // simple mic icon: rounded box + stand
    u8g2_oled.drawRBox(110, 24, 10, 16, 4);
    u8g2_oled.drawLine(112, 42, 112, 48);
    u8g2_oled.drawLine(118, 42, 118, 48);
    u8g2_oled.drawLine(109, 48, 121, 48);
    u8g2_oled.drawStr(0, 56, "say: bay-gum");
  } while (u8g2_oled.nextPage());
}

inline void oled_detected(float confidence) {
  char conf[24];
  int pct = (int)(confidence * 100.0f + 0.5f);
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  snprintf(conf, sizeof(conf), "conf: %d%%", pct);
  u8g2_oled.firstPage();
  do {
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 12, "! KEYWORD !");
    u8g2_oled.setFont(u8g2_font_7x13B_tf);
    u8g2_oled.drawStr(0, 32, "BEGUM");
    u8g2_oled.drawStr(0, 48, "DETECTED!");
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 62, conf);
  } while (u8g2_oled.nextPage());
}

inline void oled_cooldown(int seconds_left) {
  char buf[24];
  snprintf(buf, sizeof(buf), "Cooldown: %ds", seconds_left);
  u8g2_oled.firstPage();
  do {
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 12, "BEGUM");
    u8g2_oled.setFont(u8g2_font_10x20_tf);
    u8g2_oled.drawStr(0, 38, buf);
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 56, "please wait...");
  } while (u8g2_oled.nextPage());
}

inline void oled_boot(uint32_t freeHeap, float inferHz) {
  char l1[32], l2[32];
  snprintf(l1, sizeof(l1), "RAM %lu B", (unsigned long)freeHeap);
  snprintf(l2, sizeof(l2), "infer %.1f Hz", (double)inferHz);
  u8g2_oled.firstPage();
  do {
    u8g2_oled.setFont(u8g2_font_6x10_tf);
    u8g2_oled.drawStr(0, 12, "BEGUM boot OK");
    u8g2_oled.drawStr(0, 28, l1);
    u8g2_oled.drawStr(0, 42, l2);
    u8g2_oled.drawStr(0, 56, "Listening...");
  } while (u8g2_oled.nextPage());
}

#endif  // BEGUM_OLED_DISPLAY_H
