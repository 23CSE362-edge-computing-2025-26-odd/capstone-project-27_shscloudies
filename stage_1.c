/* stm32_ecg_baremetal.c
 * Bare-metal STM32F4 implementation:
 * - ADC sampling (PA0)
 * - Bandpass + Notch (biquads)
 * - Haar wavelet denoise (1-level)
 * - QRS detection and BPM
 * - Window switching: 5s (1250) normal, 3s (750) alert
 * - LED behavior: PC13, PC14, PB9
 * - UART2 streaming of abnormal windows to Raspberry Pi
 *
 * NOTE: Replace filter coefficients with coefficients designed for your Fs.
 */

#include "stm32f4xx.h"
#include <stdio.h>
#include <string.h>

/* ----------------- Configuration ----------------- */
#define FS               250                    // sampling frequency (Hz)
#define NORMAL_WINDOW    (5 * FS)               // 5 s -> 1250 samples
#define ALERT_WINDOW     (3 * FS)               // 3 s -> 750 samples
#define NO_DATA_MS       2000                   // 2 seconds no-data threshold
#define UART_BAUDRATE    115200

/* ----------------- Globals ----------------- */
volatile uint32_t msTicks = 0;                   // 1 ms tick (SysTick)
static float ecg_window[NORMAL_WINDOW];          // max window buffer (global to avoid stack overflow)
static int window_size = NORMAL_WINDOW;

volatile uint32_t last_sample_time_ms = 0;       // last time a sample was collected
int abnormal = 0;

/* ----------------- SysTick ----------------- */
void SysTick_Handler(void) {
    msTicks++;
}
static void delay_ms(uint32_t ms) {
    uint32_t start = msTicks;
    while ((msTicks - start) < ms) { __NOP(); }
}

/* ----------------- UART2 (PA2 TX) ----------------- */
/* Uses USART2 TX only, not enabling RX interrupt here */
static void UART2_Init(void) {
    // enable GPIOA and USART2 clocks
    RCC->AHB1ENR |= (1U << 0);   // GPIOA
    RCC->APB1ENR |= (1U << 17);  // USART2

    // PA2 -> AF7 (USART2 TX)
    GPIOA->MODER &= ~(3U << (2*2));
    GPIOA->MODER |=  (2U << (2*2));   // AF mode for PA2
    GPIOA->AFR[0] &= ~(0xF << (2*4));
    GPIOA->AFR[0] |=  (7U << (2*4));  // AF7 = USART2

    // Configure USART2: 16 MHz PCLK1 assumed, BRR precomputed for 16MHz/115200
    // If SystemCoreClock differs, compute BRR accordingly.
    USART2->BRR = 0x0683; // value for 16MHz / 115200 ~ 0x683 (139)
    USART2->CR1 = 0;      // clear
    USART2->CR1 |= (1U << 3);  // TE = 1 (transmitter enable)
    USART2->CR1 |= (1U << 13); // UE = 1 (USART enable)
}
static void UART2_SendString(const char *s) {
    while (*s) {
        // wait until TXE set
        while (!(USART2->SR & (1U << 7))) { __NOP(); }
        USART2->DR = (uint8_t)(*s++);
    }
}
static void UART2_SendFloatAsInt16Line(float v) {
    // scale then send as integer line (faster and compact)
    int val = (int)(v * 1000.0f); // scale (mV or V as per your prefer)
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d\n", val);
    // send
    for (int i = 0; i < n; ++i) {
        while (!(USART2->SR & (1U << 7))) { __NOP(); }
        USART2->DR = (uint8_t)buf[i];
    }
}
static void UART2_SendHeader(const char *reason, int fs, int samples) {
    char hdr[128];
    int n = snprintf(hdr, sizeof(hdr),
        "{ \"type\":\"quick_alert\", \"reason\":\"%s\", \"fs\":%d, \"samples\":%d }\n",
        reason, fs, samples);
    UART2_SendString(hdr);
}

/* ----------------- ADC1 (PA0, channel 0) ----------------- */
static void ADC1_Init(void) {
    // enable GPIOA and ADC1 clocks
    RCC->AHB1ENR |= (1U << 0);   // GPIOA
    RCC->APB2ENR |= (1U << 8);   // ADC1

    // PA0 analog mode
    GPIOA->MODER |= (3U << (0*2)); // analog

    // ADC common and ADC1 config
    ADC->CCR = 0; // default
    ADC1->CR1 = 0;
    ADC1->CR2 = 0;
    ADC1->SQR3 = 0;       // channel 0 in regular sequence
    // sampling time selection (channel 0 in SMPR2 bits 2:0)
    // choose 84 cycles as safe sample time (can be adjusted)
    ADC1->SMPR2 &= ~(7U << (0*3));
    ADC1->SMPR2 |=  (5U << (0*3)); // 5 -> 84 cycles (approx) (refer manual)
    ADC1->CR2 |= (1U << 0); // ADON = 1 (enable ADC)
    // small delay after ADC on
    for (volatile int i=0;i<10000;i++) __NOP();
}
static uint16_t ADC1_Read(void) {
    ADC1->CR2 |= (1U << 30); // SWSTART
    // wait for EOC (end of conversion)
    while (!(ADC1->SR & (1U << 1))) { __NOP(); }
    uint16_t v = (uint16_t)ADC1->DR;
    ADC1->SR &= ~(1U << 1); // clear EOC (read DR usually clears)
    return v;
}

/* ----------------- GPIO init for LEDs (PC13, PC14, PB9) ----------------- */
static void GPIO_LED_Init(void) {
    // enable GPIOC and GPIOB
    RCC->AHB1ENR |= (1U << 2) | (1U << 1); // GPIOC, GPIOB
    // PC13, PC14 outputs
    GPIOC->MODER &= ~((3U << (13*2)) | (3U << (14*2)));
    GPIOC->MODER |=  ((1U << (13*2)) | (1U << (14*2)));
    // PB9 output
    GPIOB->MODER &= ~(3U << (9*2));
    GPIOB->MODER |=  (1U << (9*2));
}

/* ----------------- Biquad filter (Direct Form II Transposed) ----------------- */
typedef struct { float b0,b1,b2,a1,a2; float z1,z2; } Biquad;
static Biquad bp_bq, notch_bq;

static inline float biquad_process(Biquad *b, float x) {
    float y = b->b0 * x + b->z1;
    b->z1 = b->b1 * x - b->a1 * y + b->z2;
    b->z2 = b->b2 * x - b->a2 * y;
    return y;
}

/* Placeholder coefficients: replace with properly designed coefficients for Fs=250Hz */
static void filters_init(void) {
    // Bandpass approx (0.5 - 40 Hz) placeholder
    bp_bq.b0 =  0.0675f; bp_bq.b1 = 0.0f;    bp_bq.b2 = -0.0675f;
    bp_bq.a1 = -1.1380f; bp_bq.a2 = 0.4866f;
    bp_bq.z1 = bp_bq.z2 = 0.0f;

    // Notch around 50 Hz placeholder
    notch_bq.b0 = 0.98799f; notch_bq.b1 = -1.8741f; notch_bq.b2 = 0.98799f;
    notch_bq.a1 = -1.8741f; notch_bq.a2 = 0.97599f;
    notch_bq.z1 = notch_bq.z2 = 0.0f;
}

/* ----------------- Haar DWT 1-level denoise (in-place) ----------------- */
static void haar_denoise_inplace(float *x, int N, float thr) {
    // N must be even. Use static workspace sized to max half window
    static float approx[NORMAL_WINDOW/2];
    static float detail[NORMAL_WINDOW/2];

    int half = N / 2;
    for (int i=0;i<half;i++) {
        approx[i] = (x[2*i] + x[2*i+1]) * 0.70710678f; // 1/sqrt(2)
        detail[i] = (x[2*i] - x[2*i+1]) * 0.70710678f;
    }
    // soft threshold
    for (int i=0;i<half;i++) {
        float d = detail[i];
        if (d > thr) detail[i] = d - thr;
        else if (d < -thr) detail[i] = d + thr;
        else detail[i] = 0.0f;
    }
    // reconstruct
    for (int i=0;i<half;i++) {
        x[2*i]   = (approx[i] + detail[i]) * 0.70710678f;
        x[2*i+1] = (approx[i] - detail[i]) * 0.70710678f;
    }
}

/* ----------------- QRS / Peak detection (simple threshold + refractory) ----------------- */
static int detect_qrs_simple(float *sig, int N) {
    // adaptive threshold: mean + k * std
    float mean = 0.0f, var = 0.0f;
    for (int i=0;i<N;i++) mean += sig[i];
    mean /= N;
    for (int i=0;i<N;i++) {
        float d = sig[i] - mean;
        var += d*d;
    }
    float std = sqrtf(var / N);
    float thresh = mean + 0.6f * std + 0.05f; // heuristic

    int count = 0;
    int refractory = FS / 5; // 200 ms
    int last_idx = -refractory;
    for (int i=1;i<N-1;i++) {
        if (sig[i] > thresh && sig[i] > sig[i-1] && sig[i] >= sig[i+1]) {
            if ((i - last_idx) > refractory) {
                count++;
                last_idx = i;
            }
        }
    }
    return count;
}

/* ----------------- Process window: filtering, denoising, detection, UART streaming ----------------- */
static void process_window_and_act(void) {
    // Use local filtered buffer sized to current window_size
    static float filtered_buf[NORMAL_WINDOW];

    // 1) Bandpass + notch
    for (int i=0;i<window_size;i++) {
        float v = ecg_window[i];
        v = biquad_process(&bp_bq, v);
        v = biquad_process(&notch_bq, v);
        filtered_buf[i] = v;
    }

    // 2) Haar denoise
    haar_denoise_inplace(filtered_buf, window_size, 0.02f); // threshold tuneable

    // 3) detect beats and compute bpm
    int beats = detect_qrs_simple(filtered_buf, window_size);
    // bpm = beats per window * (60 / window_seconds)
    float window_sec = (float)window_size / (float)FS;
    int bpm = 0;
    if (window_sec > 0.0f) bpm = (int)((beats * 60.0f) / window_sec + 0.5f);

    // 4) decision logic
    if (beats == 0 || bpm < 30 || bpm > 180) {
        // abnormal
        abnormal = 1;
        window_size = ALERT_WINDOW; // shorten window for faster reaction
        // LED PC14 ON (solid)
        GPIOC->ODR |= (1U << 14);

        // 5) stream header + samples to Raspberry Pi
        UART2_SendHeader("abnormal_detected", FS, window_size);
        for (int i=0;i<window_size;i++) {
            UART2_SendFloatAsInt16Line(filtered_buf[i]);
        }
        UART2_SendString("[END]\n");
    } else {
        // normal
        abnormal = 0;
        window_size = NORMAL_WINDOW;
        // LED PC14 off
        GPIOC->ODR &= ~(1U << 14);
    }
}

/* ----------------- Blink LEDs when no data for 2 seconds ----------------- */
static void blink_all_leds_once(void) {
    // blink PC13, PC14, PB9 in sequence
    GPIOC->ODR ^= (1U << 13); // toggle PC13
    delay_ms(200);
    GPIOC->ODR ^= (1U << 14); // toggle PC14
    delay_ms(200);
    GPIOB->ODR ^= (1U << 9);  // toggle PB9
    delay_ms(200);
    // turn them off (ensure off)
    GPIOC->ODR &= ~((1U << 13) | (1U << 14));
    GPIOB->ODR &= ~(1U << 9);
}

/* ----------------- Main ----------------- */
int main(void) {
    // configure SysTick for 1 ms ticks (assumes SystemCoreClock is set appropriately)
    SysTick_Config(SystemCoreClock / 1000U);

    // init peripherals
    UART2_Init();
    ADC1_Init();
    GPIO_LED_Init();
    filters_init();

    // initialize last sample time
    last_sample_time_ms = msTicks;

    // Main loop: continuously collect window_size samples and process
    while (1) {
        // collect samples for current window_size
        for (int i=0; i<window_size; i++) {
            // read ADC sample
            uint16_t raw = ADC1_Read();
            // convert to voltage (0..3.3) or to any scale you want
            float volt = (3.3f * (float)raw) / 4095.0f;
            ecg_window[i] = volt;
            last_sample_time_ms = msTicks;

            // Respect sampling rate ~ FS. Use precise ms delay
            delay_ms(1000U / FS);

            // If no-signal situation: check outside sample loop as well
            // (last_sample_time_ms updated above)
        }

        // if last sample older than NO_DATA_MS -> blink alert LEDs sequence
        if ((msTicks - last_sample_time_ms) > NO_DATA_MS) {
            // Blink as long as no data
            // Here we perform a few blinks then continue to next loop
            blink_all_leds_once();
            continue; // go back and attempt to collect window again
        }

        // Process the freshly-collected window (filter, denoise, detect)
        process_window_and_act();

        // After abnormal detection and streaming, we remain in loop:
        // The Raspberry Pi may confirm/issue commands later via UART (not implemented here).
    }

    // should never reach here
    return 0;
}
