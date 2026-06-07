#include "sensor_hal.h"
#include "config.h"
#include <Arduino.h>
#include <SPI.h>
#include <OneWire.h>
#include <math.h>

// ADXL345 register map
#define ADXL_REG_DEVID      0x00
#define ADXL_REG_BW_RATE    0x2C
#define ADXL_REG_POWER_CTL  0x2D
#define ADXL_REG_DATA_FORMAT 0x31
#define ADXL_REG_DATAX0     0x32
#define ADXL_DEVID          0xE5
#define ADXL_READ_BIT       0x80
#define ADXL_MULTI_BIT      0x40
#define ADXL_BW_100HZ       0x0A
#define ADXL_RANGE_2G       0x00
#define ADXL_SCALE_2G       0.0039f  // g per LSB at 2g range

static OneWire ds(PIN_TEMP_DATA);

static void adxl_write(uint8_t reg, uint8_t val) {
    digitalWrite(PIN_ADXL_CS, LOW);
    SPI.transfer(reg & 0x3F);
    SPI.transfer(val);
    digitalWrite(PIN_ADXL_CS, HIGH);
}

static uint8_t adxl_read(uint8_t reg) {
    digitalWrite(PIN_ADXL_CS, LOW);
    SPI.transfer(reg | ADXL_READ_BIT);
    uint8_t val = SPI.transfer(0x00);
    digitalWrite(PIN_ADXL_CS, HIGH);
    return val;
}

static void adxl_read_burst(uint8_t reg, uint8_t *buf, uint8_t len) {
    digitalWrite(PIN_ADXL_CS, LOW);
    SPI.transfer(reg | ADXL_READ_BIT | ADXL_MULTI_BIT);
    for (uint8_t i = 0; i < len; i++) buf[i] = SPI.transfer(0x00);
    digitalWrite(PIN_ADXL_CS, HIGH);
}

static sensor_status_t read_vibration(float *x, float *y, float *z) {
    uint8_t raw[6];
    adxl_read_burst(ADXL_REG_DATAX0, raw, 6);

    int16_t ax = (int16_t)((raw[1] << 8) | raw[0]);
    int16_t ay = (int16_t)((raw[3] << 8) | raw[2]);
    int16_t az = (int16_t)((raw[5] << 8) | raw[4]);

    *x = ax * ADXL_SCALE_2G;
    *y = ay * ADXL_SCALE_2G;
    *z = az * ADXL_SCALE_2G;
    return SENSOR_OK;
}

static sensor_status_t read_temperature(float *temp_c) {
    byte addr[8];
    if (!ds.search(addr)) {
        ds.reset_search();
        return SENSOR_TIMEOUT;
    }
    if (OneWire::crc8(addr, 7) != addr[7]) return SENSOR_CRC_ERR;

    ds.reset();
    ds.select(addr);
    ds.write(0x44, 1);
    delay(750);

    ds.reset();
    ds.select(addr);
    ds.write(0xBE);

    byte data[9];
    for (uint8_t i = 0; i < 9; i++) data[i] = ds.read();
    if (OneWire::crc8(data, 8) != data[8]) return SENSOR_CRC_ERR;

    int16_t raw = (data[1] << 8) | data[0];
    *temp_c = (float)raw / 16.0f;
    ds.reset_search();
    return SENSOR_OK;
}

static sensor_status_t read_current(float *current_a) {
    // ACS712-5A: Vcc/2 = 2.5V at 0A, sensitivity 185 mV/A
    uint16_t adc_raw = analogRead(PIN_CURR_ADC);
    float voltage    = (adc_raw / 4095.0f) * 3.3f;
    *current_a       = (voltage - 1.65f) / 0.185f;
    return SENSOR_OK;
}

sensor_status_t sensor_hal_init(void) {
    SPI.begin();
    SPI.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE3));
    pinMode(PIN_ADXL_CS,   OUTPUT);
    pinMode(PIN_ADXL_INT1, INPUT);
    digitalWrite(PIN_ADXL_CS, HIGH);

    if (adxl_read(ADXL_REG_DEVID) != ADXL_DEVID) return SENSOR_ERR;

    adxl_write(ADXL_REG_DATA_FORMAT, ADXL_RANGE_2G | 0x08); // full resolution
    adxl_write(ADXL_REG_BW_RATE,    ADXL_BW_100HZ);
    adxl_write(ADXL_REG_POWER_CTL,  0x08);  // measure mode

    analogReadResolution(12);
    return SENSOR_OK;
}

bool sensor_hal_selftest(void) {
    return (adxl_read(ADXL_REG_DEVID) == ADXL_DEVID);
}

sensor_status_t sensor_hal_read(sensor_sample_t *s) {
    s->timestamp_ms = (uint64_t)millis();
    s->flags = 0;

    sensor_status_t st;

    st = read_vibration(&s->vib_x, &s->vib_y, &s->vib_z);
    if (st != SENSOR_OK) return st;
    if (fabsf(s->vib_x) > VIB_MAX_G || fabsf(s->vib_y) > VIB_MAX_G)
        s->flags |= 0x01;

    st = read_temperature(&s->temp_c);
    if (st != SENSOR_OK) return st;
    if (s->temp_c > TEMP_MAX_C) s->flags |= 0x02;

    st = read_current(&s->current_a);
    if (st != SENSOR_OK) return st;
    if (fabsf(s->current_a) > CURR_MAX_A) s->flags |= 0x04;

    return SENSOR_OK;
}

void sensor_hal_sleep(void) {
    adxl_write(ADXL_REG_POWER_CTL, 0x04); // standby
}

void sensor_hal_wake(void) {
    adxl_write(ADXL_REG_POWER_CTL, 0x08); // measure
    delay(10);
}
