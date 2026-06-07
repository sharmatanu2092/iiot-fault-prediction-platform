#include "data_buffer.h"
#include <string.h>

// Single-producer single-consumer lock-free ring.
// head written only by ISR (producer).
// tail written only by main task (consumer).
// Both are volatile so the compiler does not cache them in registers.

void ring_init(ring_buffer_t *rb) {
    rb->head    = 0;
    rb->tail    = 0;
    rb->dropped = 0;
}

bool ring_full(const ring_buffer_t *rb) {
    return ((rb->head - rb->tail) & (RING_CAPACITY - 1)) == (RING_CAPACITY - 1);
}

bool ring_empty(const ring_buffer_t *rb) {
    return rb->head == rb->tail;
}

uint16_t ring_count(const ring_buffer_t *rb) {
    return (rb->head - rb->tail) & (RING_CAPACITY - 1);
}

// Called from ISR context -- no blocking, no malloc.
bool ring_push(ring_buffer_t *rb, const sensor_sample_t *s) {
    if (ring_full(rb)) {
        rb->dropped++;
        return false;
    }
    uint16_t slot = rb->head & (RING_CAPACITY - 1);
    memcpy(&rb->buf[slot], s, sizeof(sensor_sample_t));
    // publish head after data is written so consumer never sees partial
    __asm__ volatile("" ::: "memory");
    rb->head = (rb->head + 1) & 0xFFFF;
    return true;
}

// Called from main task only.
bool ring_pop(ring_buffer_t *rb, sensor_sample_t *s) {
    if (ring_empty(rb)) return false;
    uint16_t slot = rb->tail & (RING_CAPACITY - 1);
    memcpy(s, &rb->buf[slot], sizeof(sensor_sample_t));
    __asm__ volatile("" ::: "memory");
    rb->tail = (rb->tail + 1) & 0xFFFF;
    return true;
}

void ring_flush(ring_buffer_t *rb) {
    rb->tail = rb->head;
}
