#include "i2s_capture.h"
#include "driver/i2s.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "I2S_CAPTURE";

// FreeRTOS RingBuffer handle
static RingbufHandle_t audio_ring_buffer = NULL;

void i2s_capture_init() {
    ESP_LOGI(TAG, "Initializing I2S and Ring Buffer...");

    // 1. Create the FreeRTOS Ring Buffer
    // We use RINGBUF_TYPE_BYTEBUF for simple byte streaming
    audio_ring_buffer = xRingbufferCreate(BUFFER_SIZE_BYTES, RINGBUF_TYPE_BYTEBUF);
    if (audio_ring_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to create ring buffer!");
        return;
    }

    // 2. Configure I2S driver
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // Default interrupt priority
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN
    };

    // 3. Install and start I2S driver
    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed installing I2S driver: %d", err);
        return;
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed setting I2S pins: %d", err);
        return;
    }

    ESP_LOGI(TAG, "I2S Initialization Complete.");
}

void i2s_capture_task(void *pvParameters) {
    uint8_t dma_read_buffer[CHUNK_SIZE_BYTES];
    size_t bytes_read = 0;

    while (1) {
        // Wait for DMA to fill the buffer, block until data is ready
        esp_err_t err = i2s_read(I2S_PORT, dma_read_buffer, CHUNK_SIZE_BYTES, &bytes_read, portMAX_DELAY);
        
        if (err == ESP_OK && bytes_read > 0) {
            // Push to ring buffer. 
            // If the buffer is full, we must manually dequeue old data to make room (sliding window).
            size_t free_space = xRingbufferGetCurFreeSize(audio_ring_buffer);
            
            if (free_space < bytes_read) {
                // Buffer full, drop oldest data
                size_t bytes_to_drop = bytes_read - free_space;
                size_t item_size;
                void *old_data = xRingbufferReceiveUpTo(audio_ring_buffer, &item_size, 0, bytes_to_drop);
                if (old_data) {
                    vRingbufferReturnItem(audio_ring_buffer, old_data);
                }
            }
            
            // Send new data to ring buffer
            xRingbufferSend(audio_ring_buffer, dma_read_buffer, bytes_read, portMAX_DELAY);
        }
    }
}

size_t get_latest_audio(int16_t *out_buffer, size_t num_samples) {
    size_t req_bytes = num_samples * sizeof(int16_t);
    size_t item_size;
    
    // In a real sliding window, you would read without dequeuing, or dequeue and re-queue.
    // FreeRTOS ringbuf doesn't have a native "peek". A common pattern is reading all data
    // into a linear buffer, grabbing the tail end, and putting the remainder back.
    // For simplicity in this demo, we do a block read.
    
    void *data = xRingbufferReceiveUpTo(audio_ring_buffer, &item_size, 0, req_bytes);
    
    if (data != NULL) {
        memcpy(out_buffer, data, item_size);
        vRingbufferReturnItem(audio_ring_buffer, data);
        return item_size;
    }
    
    return 0;
}
