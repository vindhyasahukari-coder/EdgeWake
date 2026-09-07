#ifndef I2S_CAPTURE_H
#define I2S_CAPTURE_H

#include <stdint.h>
#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"

// Audio settings matching Python config
#define SAMPLE_RATE     16000
#define NUM_CHANNELS    1
#define SAMPLE_BIT_SIZE 16

// I2S Pins for INMP441 on ESP32-S3 (can be changed based on actual wiring)
#define I2S_WS_PIN      15
#define I2S_SD_PIN      16
#define I2S_SCK_PIN     17
#define I2S_PORT        I2S_NUM_0

// Buffer sizing
#define BUFFER_DURATION_SEC 3
#define BUFFER_SIZE_BYTES   (SAMPLE_RATE * NUM_CHANNELS * (SAMPLE_BIT_SIZE / 8) * BUFFER_DURATION_SEC)
#define CHUNK_SIZE_SAMPLES  (SAMPLE_RATE / 5) // 0.2s chunks
#define CHUNK_SIZE_BYTES    (CHUNK_SIZE_SAMPLES * (SAMPLE_BIT_SIZE / 8))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the I2S peripheral and the ring buffer.
 */
void i2s_capture_init();

/**
 * @brief FreeRTOS task that continuously reads I2S DMA and pushes to ring buffer.
 */
void i2s_capture_task(void *pvParameters);

/**
 * @brief Get the latest samples from the ring buffer.
 * 
 * @param out_buffer Pointer to destination array (must be pre-allocated).
 * @param num_samples Number of int16_t samples to retrieve.
 * @return size_t Number of bytes successfully read.
 */
size_t get_latest_audio(int16_t *out_buffer, size_t num_samples);

#ifdef __cplusplus
}
#endif

#endif // I2S_CAPTURE_H
