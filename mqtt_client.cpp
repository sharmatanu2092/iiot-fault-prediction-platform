#include "mqtt_client.h"
#include "config.h"
#include <Arduino.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <string.h>
#include <stdio.h>

#define OFFLINE_BUF_SIZE    MQTT_OFFLINE_BUFFER

typedef struct {
    char topic[64];
    char payload[256];
    uint8_t qos;
} offline_entry_t;

static WiFiClientSecure  tls_client;
static PubSubClient      mqtt_client(tls_client);
static offline_entry_t   offline_buf[OFFLINE_BUF_SIZE];
static uint16_t          offline_head = 0;
static uint16_t          offline_tail = 0;
static uint32_t          reconnect_delay_ms = MQTT_RECONNECT_BASE_MS;
static uint32_t          last_reconnect_ms  = 0;

static bool offline_empty(void) { return offline_head == offline_tail; }
static bool offline_full(void) {
    return ((offline_head + 1) % OFFLINE_BUF_SIZE) == offline_tail;
}

static void offline_push(const char *topic, const char *payload, uint8_t qos) {
    if (offline_full()) return; // oldest entry lost, not worth crashing over
    strncpy(offline_buf[offline_head].topic,   topic,   63);
    strncpy(offline_buf[offline_head].payload, payload, 255);
    offline_buf[offline_head].qos = qos;
    offline_head = (offline_head + 1) % OFFLINE_BUF_SIZE;
}

static void flush_offline_buffer(void) {
    while (!offline_empty() && mqtt_client.connected()) {
        offline_entry_t *e = &offline_buf[offline_tail];
        mqtt_client.publish(e->topic, e->payload, e->qos);
        offline_tail = (offline_tail + 1) % OFFLINE_BUF_SIZE;
        delay(10);
    }
}

void mqtt_client_init(const char *host, uint16_t port) {
    tls_client.setInsecure(); // replace with setCACert() in production
    mqtt_client.setServer(host, port);
    mqtt_client.setKeepAlive(MQTT_KEEPALIVE_S);
}

mqtt_status_t mqtt_connect(void) {
    char client_id[32];
    snprintf(client_id, sizeof(client_id), "node-%s", NODE_ID);

    if (mqtt_client.connect(client_id)) {
        reconnect_delay_ms = MQTT_RECONNECT_BASE_MS;
        flush_offline_buffer();
        return MQTT_CONN_OK;
    }
    return MQTT_CONN_FAIL;
}

mqtt_status_t mqtt_publish(const char *topic, const char *payload, uint8_t qos) {
    if (!mqtt_client.connected()) {
        offline_push(topic, payload, qos);
        return MQTT_OFFLINE_BUFFERED;
    }
    return mqtt_client.publish(topic, payload, qos)
           ? MQTT_PUBLISH_OK : MQTT_PUBLISH_FAIL;
}

void mqtt_loop(void) {
    if (!mqtt_client.connected()) {
        uint32_t now = millis();
        if ((now - last_reconnect_ms) >= reconnect_delay_ms) {
            last_reconnect_ms = now;
            if (mqtt_connect() != MQTT_CONN_OK) {
                // exponential backoff, cap at max
                reconnect_delay_ms = (reconnect_delay_ms * 2 > MQTT_RECONNECT_MAX_MS)
                                     ? MQTT_RECONNECT_MAX_MS
                                     : reconnect_delay_ms * 2;
            }
        }
        return;
    }
    mqtt_client.loop();
}

bool mqtt_connected(void) {
    return mqtt_client.connected();
}

void mqtt_disconnect(void) {
    mqtt_client.disconnect();
}

uint16_t mqtt_offline_count(void) {
    return (offline_head - offline_tail + OFFLINE_BUF_SIZE) % OFFLINE_BUF_SIZE;
}
