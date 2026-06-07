#include <Arduino.h>
#include <WiFi.h>
#include <esp_task_wdt.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "sensor_hal.h"
#include "data_buffer.h"
#include "mqtt_client.h"

// Topic format: factory/{plant}/machine/{machine}/sensor/{type}
#define TOPIC_VIB   "factory/" PLANT_ID "/machine/" MACHINE_ID "/sensor/vibration"
#define TOPIC_TEMP  "factory/" PLANT_ID "/machine/" MACHINE_ID "/sensor/temperature"
#define TOPIC_CURR  "factory/" PLANT_ID "/machine/" MACHINE_ID "/sensor/current"
#define TOPIC_ALERT "factory/" PLANT_ID "/machine/" MACHINE_ID "/alerts/oob"

typedef enum {
    STATE_BOOT = 0,
    STATE_SELF_TEST,
    STATE_INIT,
    STATE_IDLE,
    STATE_SAMPLING,
    STATE_PROCESSING,
    STATE_PUBLISHING,
    STATE_ERROR,
    STATE_SLEEP
} node_state_t;

static ring_buffer_t  ring;
static node_state_t   state       = STATE_BOOT;
static uint32_t       seq         = 0;
static hw_timer_t    *sample_timer = NULL;
static volatile bool  timer_fired  = false;

// ISR: runs at SAMPLE_RATE_HZ, pushes samples into ring
void IRAM_ATTR sample_isr(void) {
    timer_fired = true;
}

static void build_payload(char *buf, size_t len,
                           const char *sensor, float value,
                           uint64_t ts_ms, uint8_t flags) {
    snprintf(buf, len,
        "{\"node_id\":\"%s\",\"ts_ms\":%llu,\"sensor\":\"%s\","
        "\"value\":%.4f,\"seq\":%lu,\"flags\":%u}",
        NODE_ID, (unsigned long long)ts_ms,
        sensor, value, (unsigned long)seq++, flags);
}

static void publish_sample(const sensor_sample_t *s) {
    char payload[256];
    float vib_mag = sqrtf(s->vib_x*s->vib_x +
                          s->vib_y*s->vib_y +
                          s->vib_z*s->vib_z);

    build_payload(payload, sizeof(payload), "vibration",  vib_mag,      s->timestamp_ms, s->flags);
    mqtt_publish(TOPIC_VIB,  payload, MQTT_QOS);

    build_payload(payload, sizeof(payload), "temperature", s->temp_c,   s->timestamp_ms, s->flags);
    mqtt_publish(TOPIC_TEMP, payload, MQTT_QOS);

    build_payload(payload, sizeof(payload), "current",    s->current_a, s->timestamp_ms, s->flags);
    mqtt_publish(TOPIC_CURR, payload, MQTT_QOS);

    if (s->flags) {
        snprintf(payload, sizeof(payload),
            "{\"node_id\":\"%s\",\"ts_ms\":%llu,\"oob_flags\":%u}",
            NODE_ID, (unsigned long long)s->timestamp_ms, s->flags);
        mqtt_publish(TOPIC_ALERT, payload, MQTT_QOS);
    }
}

void setup(void) {
    Serial.begin(UART_BAUD);
    state = STATE_SELF_TEST;

    if (sensor_hal_init() != SENSOR_OK || !sensor_hal_selftest()) {
        state = STATE_ERROR;
        Serial.println("[ERR] sensor init failed");
        return;
    }

    ring_init(&ring);

    WiFi.begin("YOUR_SSID", "YOUR_PASSWORD");
    uint8_t attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts++ < 20) delay(500);
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WARN] WiFi failed, running offline");
    }

    mqtt_client_init(MQTT_HOST, MQTT_PORT);

    // sample timer: fires every 1000/SAMPLE_RATE_HZ ms
    sample_timer = timerBegin(0, 80, true);           // 80 prescaler = 1MHz
    timerAttachInterrupt(sample_timer, &sample_isr, true);
    timerAlarmWrite(sample_timer, 1000000 / SAMPLE_RATE_HZ, true);
    timerAlarmEnable(sample_timer);

    esp_task_wdt_init(WDT_TIMEOUT_MS / 1000, true);
    esp_task_wdt_add(NULL);

    state = STATE_IDLE;
    Serial.println("[OK] node ready");
}

void loop(void) {
    esp_task_wdt_reset();
    mqtt_loop();

    switch (state) {
        case STATE_IDLE:
            if (timer_fired) {
                timer_fired = false;
                state = STATE_SAMPLING;
            }
            break;

        case STATE_SAMPLING: {
            sensor_sample_t s;
            sensor_status_t st = sensor_hal_read(&s);
            if (st == SENSOR_OK) {
                if (!ring_push(&ring, &s)) {
                    Serial.println("[WARN] ring full, sample dropped");
                }
            } else {
                Serial.print("[ERR] sensor read: ");
                Serial.println(st);
            }
            state = ring_count(&ring) >= 10 ? STATE_PROCESSING : STATE_IDLE;
            break;
        }

        case STATE_PROCESSING:
            // drain ring in batches; could add feature extraction here
            state = STATE_PUBLISHING;
            break;

        case STATE_PUBLISHING: {
            sensor_sample_t s;
            while (ring_pop(&ring, &s)) {
                publish_sample(&s);
            }
            state = STATE_IDLE;
            break;
        }

        case STATE_ERROR:
            // blink LED, wait for WDT reset
            digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
            delay(250);
            break;

        default:
            state = STATE_IDLE;
            break;
    }
}
