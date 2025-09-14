/* stm32_ecg_baremetal_nonblocking.c
 * Non-blocking STM32F4 ECG simulation:
 * - ADC sampling (PA0 or simulated)
 * - Bandpass + Notch (biquads)
 * - Haar wavelet denoise (1-level)
 * - QRS detection and BPM
 * - LED behavior: PC13, PC14, PB9
 * - UART2 streaming of abnormal windows to Raspberry Pi
 */

#include "stm32f4xx.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


/* ----------------- Configuration ----------------- */
#define FS               250                    // sampling frequency (Hz)
#define NORMAL_WINDOW    (5 * FS)               // 5 s -> 1250 samples
#define ALERT_WINDOW     (3 * FS)               // 3 s -> 750 samples
#define UART_BAUDRATE    115200
#define LED_BLINK_MS     200                    // ms

/* ----------------- Globals ----------------- */
volatile uint32_t msTicks = 0;                   // 1 ms tick (SysTick)
static float ecg_window[NORMAL_WINDOW];          // ECG buffer
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

/* ----------------- UART2 ----------------- */
static void UART2_Init(void) {
    RCC->AHB1ENR |= (1U << 0);   // GPIOA
    RCC->APB1ENR |= (1U << 17);  // USART2

    GPIOA->MODER &= ~(3U << (2*2));
    GPIOA->MODER |=  (2U << (2*2)); // AF mode
    GPIOA->AFR[0] &= ~(0xF << (2*4));
    GPIOA->AFR[0] |=  (7U << (2*4)); // AF7 = USART2

    USART2->BRR = 0x0683; // 16MHz / 115200
    USART2->CR1 = 0;
    USART2->CR1 |= (1U << 3) | (1U << 13); // TE + UE
}

static void UART2_SendString(const char *s) {
    while (*s) {
        while (!(USART2->SR & (1U << 7))) { __NOP(); }
        USART2->DR = (uint8_t)(*s++);
    }
}

static void UART2_SendFloatAsInt16Line(float v) {
    int val = (int)(v * 1000.0f);
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d\n", val);
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

/* ----------------- GPIO LEDs ----------------- */
static void GPIO_LED_Init(void) {
    RCC->AHB1ENR |= (1U << 2) | (1U << 1); // GPIOC, GPIOB
    GPIOC->MODER &= ~((3U << (13*2)) | (3U << (14*2)));
    GPIOC->MODER |=  ((1U << (13*2)) | (1U << (14*2)));
    GPIOB->MODER &= ~(3U << (9*2));
    GPIOB->MODER |=  (1U << (9*2));
}

/* ----------------- Biquad filter ----------------- */
typedef struct { float b0,b1,b2,a1,a2; float z1,z2; } Biquad;
static Biquad bp_bq, notch_bq;

static inline float biquad_process(Biquad *b, float x) {
    float y = b->b0 * x + b->z1;
    b->z1 = b->b1 * x - b->a1 * y + b->z2;
    b->z2 = b->b2 * x - b->a2 * y;
    return y;
}

static void filters_init(void) {
    bp_bq.b0 =  0.0675f; bp_bq.b1 = 0.0f; bp_bq.b2 = -0.0675f;
    bp_bq.a1 = -1.1380f; bp_bq.a2 = 0.4866f;
    bp_bq.z1 = bp_bq.z2 = 0.0f;

    notch_bq.b0 = 0.98799f; notch_bq.b1 = -1.8741f; notch_bq.b2 = 0.98799f;
    notch_bq.a1 = -1.8741f; notch_bq.a2 = 0.97599f;
    notch_bq.z1 = notch_bq.z2 = 0.0f;
}

/* ----------------- Haar DWT ----------------- */
static void haar_denoise_inplace(float *x, int N, float thr) {
    static float approx[NORMAL_WINDOW/2], detail[NORMAL_WINDOW/2];
    int half = N/2;
    for (int i=0;i<half;i++) {
        approx[i] = (x[2*i] + x[2*i+1])*0.70710678f;
        detail[i] = (x[2*i] - x[2*i+1])*0.70710678f;
    }
    for (int i=0;i<half;i++) {
        float d = detail[i];
        if (d > thr) detail[i] = d - thr;
        else if (d < -thr) detail[i] = d + thr;
        else detail[i] = 0.0f;
    }
    for (int i=0;i<half;i++) {
        x[2*i] = (approx[i]+detail[i])*0.70710678f;
        x[2*i+1] = (approx[i]-detail[i])*0.70710678f;
    }
}

/* ----------------- QRS Detection ----------------- */
static int detect_qrs_simple(float *sig, int N) {
    float mean = 0.0f, var = 0.0f;
    for (int i=0;i<N;i++) mean += sig[i];
    mean /= N;
    for (int i=0;i<N;i++) { float d = sig[i]-mean; var += d*d; }
    float std = sqrtf(var / N);
    float thresh = mean + 0.6f*std + 0.05f;

    int count=0, refractory = FS/5, last_idx=-refractory;
    for (int i=1;i<N-1;i++) {
        if (sig[i]>thresh && sig[i]>sig[i-1] && sig[i]>=sig[i+1]) {
            if ((i-last_idx) > refractory) { count++; last_idx=i; }
        }
    }
    return count;
}

/* ----------------- ECG Window Processing ----------------- */
static void process_window(float *window, int size) {
    static float filtered_buf[NORMAL_WINDOW];
    for (int i=0;i<size;i++) {
        float v = window[i];
        v = biquad_process(&bp_bq, v);
        v = biquad_process(&notch_bq, v);
        filtered_buf[i] = v;
    }
    haar_denoise_inplace(filtered_buf, size, 0.02f);
    int beats = detect_qrs_simple(filtered_buf, size);
    float sec = (float)size/FS;
    int bpm = (sec>0)? (int)((beats*60.0f)/sec+0.5f) : 0;

    if (beats==0 || bpm<30 || bpm>180) {
        abnormal = 1;
        window_size = ALERT_WINDOW;
        GPIOC->ODR |= (1U<<14);
        UART2_SendHeader("abnormal_detected", FS, window_size);
        for (int i=0;i<window_size;i++) UART2_SendFloatAsInt16Line(filtered_buf[i]);
        UART2_SendString("[END]\n");
    } else {
        abnormal = 0;
        window_size = NORMAL_WINDOW;
        GPIOC->ODR &= ~(1U<<14);
    }
}

/* ----------------- ADC Read (Simulated) ----------------- */
static float ADC_Read_Simulated(void) {
    static float t=0.0f;
    t += 1.0f/FS;
    return 1.0f + 0.5f*sinf(2.0f*3.14159f*1.2f*t) + 0.05f*((rand()%100)/100.0f-0.5f);
}

/* ----------------- Main ----------------- */
int main(void) {
    SysTick_Config(SystemCoreClock / 1000U);
    UART2_Init();
    GPIO_LED_Init();
    filters_init();
    last_sample_time_ms = msTicks;

    int ecg_idx = 0;
    uint32_t last_led_ms = 0;

    while (1) {
        // ----------------- LED blinking -----------------
        if ((msTicks - last_led_ms) > LED_BLINK_MS) {
            GPIOC->ODR ^= (1U<<13);
            GPIOB->ODR ^= (1U<<9);
            last_led_ms = msTicks;
        }

        // ----------------- Sample ECG -----------------
        if ((msTicks - last_sample_time_ms) >= (1000U / FS)) {
            float volt = ADC_Read_Simulated();
            ecg_window[ecg_idx++] = volt;
            if (ecg_idx >= window_size) {
                process_window(ecg_window, window_size);
                ecg_idx = 0;
            }
            last_sample_time_ms = msTicks;
        }
    }
    return 0;
}