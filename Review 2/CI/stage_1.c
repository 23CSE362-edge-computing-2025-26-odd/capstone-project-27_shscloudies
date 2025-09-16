/* stm32_ecg_final_seq.c
 * STM32F4 ECG Processing - Final Version with Ordered Simulation
 * - ECG simulated generator: Normal → Abnormal → Normal → NoData → Normal
 * - Preprocessing (bandpass + notch + wavelet)
 * - QRS detection + BPM
 * - Adaptive window (5s normal / 3s abnormal)
 * - LED logic:
 *   * 0–5s warm-up: all OFF
 *   * 5–7s startup: all ON
 *   * Normal: all OFF
 *   * Abnormal: PC14 ON
 *   * No data ≥2s: PC13 + PB9 ON
 */

#include "stm32f4xx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---------------- Config ---------------- */
#define FS 250
#define NORMAL_WINDOW (5 * FS)
#define ALERT_WINDOW (3 * FS)
#define STARTUP_MS 2000
#define WARMUP_MS 5000
#define NODATA_TIMEOUT 2000
#define SWITCH_INTERVAL 5000 // 5s per mode

/* ---------------- Globals ---------------- */
volatile uint32_t msTicks = 0;
static float ecg_window[NORMAL_WINDOW];
static int window_size = NORMAL_WINDOW;
volatile uint32_t last_sample_time_ms = 0;
volatile uint32_t last_data_time_ms = 0;
int abnormal = 0;

typedef enum
{
    ECG_NORMAL1,
    ECG_ABNORMAL,
    ECG_NORMAL2,
    ECG_NODATA
} ECGMode;
ECGMode sim_mode = ECG_NORMAL1;
uint32_t sim_switch_time = 0;

/* ---------------- SysTick ---------------- */
void SysTick_Handler(void) { msTicks++; }
static void delay_ms(uint32_t ms)
{
    uint32_t start = msTicks;
    while ((msTicks - start) < ms)
    {
        __NOP();
    }
}

/* ---------------- GPIO LEDs ---------------- */
static void GPIO_LED_Init(void)
{
    RCC->AHB1ENR |= (1U << 2) | (1U << 1); // GPIOC + GPIOB
    GPIOC->MODER &= ~((3U << (13 * 2)) | (3U << (14 * 2)));
    GPIOC->MODER |= ((1U << (13 * 2)) | (1U << (14 * 2)));
    GPIOB->MODER &= ~(3U << (9 * 2));
    GPIOB->MODER |= (1U << (9 * 2));

    // All OFF initially
    GPIOC->ODR |= (1U << 13);  // PC13 OFF (active low)
    GPIOC->ODR &= ~(1U << 14); // PC14 OFF
    GPIOB->ODR &= ~(1U << 9);  // PB9 OFF
}

/* ---------------- ECG Template ---------------- */
static float ECG_Template(float t)
{
    return (
        0.1f * expf(-powf((t - 0.2f) / 0.05f, 2)) +
        -0.15f * expf(-powf((t - 0.3f) / 0.015f, 2)) +
        1.2f * expf(-powf((t - 0.32f) / 0.01f, 2)) +
        -0.25f * expf(-powf((t - 0.34f) / 0.015f, 2)) +
        0.3f * expf(-powf((t - 0.55f) / 0.1f, 2)));
}

/* ---------------- ECG Simulator ---------------- */
static float ECG_ReadSimulated(void)
{
    static float t = 0.0f;

    // Switch mode every 5s in desired order
    if (msTicks - sim_switch_time > SWITCH_INTERVAL)
    {
        if (sim_mode == ECG_NORMAL1)
            sim_mode = ECG_ABNORMAL;
        else if (sim_mode == ECG_ABNORMAL)
            sim_mode = ECG_NORMAL2;
        else if (sim_mode == ECG_NORMAL2)
            sim_mode = ECG_NODATA;
        else
            sim_mode = ECG_NORMAL1;
        sim_switch_time = msTicks;
    }

    if (sim_mode == ECG_NODATA)
    {
        return 0.0f; // flatline
    }
    else if (sim_mode == ECG_ABNORMAL)
    {
        t += 1.0f / FS;
        return 0.5f * sinf(2 * 3.14159f * 3.0f * t); // fast abnormal
    }
    else
    { // normal1 or normal2
        t += 1.0f / FS;
        float base = ECG_Template(fmodf(t, 0.8f));
        base += 0.02f * ((rand() % 100) / 100.0f - 0.5f);
        return base;
    }
}

/* ---------------- Filters ---------------- */
typedef struct
{
    float b0, b1, b2, a1, a2;
    float z1, z2;
} Biquad;
static Biquad bp_bq, notch_bq;

static inline float biquad_process(Biquad *b, float x)
{
    float y = b->b0 * x + b->z1;
    b->z1 = b->b1 * x - b->a1 * y + b->z2;
    b->z2 = b->b2 * x - b->a2 * y;
    return y;
}
static void filters_init(void)
{
    bp_bq = (Biquad){0.0675f, 0.0f, -0.0675f, -1.1380f, 0.4866f, 0, 0};
    notch_bq = (Biquad){0.98799f, -1.8741f, 0.98799f, -1.8741f, 0.97599f, 0, 0};
}

/* ---------------- Haar Wavelet ---------------- */
static void haar_denoise(float *x, int N, float thr)
{
    static float approx[NORMAL_WINDOW / 2], detail[NORMAL_WINDOW / 2];
    int half = N / 2;
    for (int i = 0; i < half; i++)
    {
        approx[i] = (x[2 * i] + x[2 * i + 1]) * 0.7071f;
        detail[i] = (x[2 * i] - x[2 * i + 1]) * 0.7071f;
    }
    for (int i = 0; i < half; i++)
    {
        float d = detail[i];
        if (d > thr)
            detail[i] = d - thr;
        else if (d < -thr)
            detail[i] = d + thr;
        else
            detail[i] = 0.0f;
    }
    for (int i = 0; i < half; i++)
    {
        x[2 * i] = (approx[i] + detail[i]) * 0.7071f;
        x[2 * i + 1] = (approx[i] - detail[i]) * 0.7071f;
    }
}

/* ---------------- QRS Detection ---------------- */
static int detect_qrs(float *sig, int N)
{
    float mean = 0, var = 0;
    for (int i = 0; i < N; i++)
        mean += sig[i];
    mean /= N;
    for (int i = 0; i < N; i++)
    {
        float d = sig[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / N);
    float thresh = mean + 0.6f * std + 0.05f;
    int beats = 0, last = -FS / 5;
    for (int i = 1; i < N - 1; i++)
    {
        if (sig[i] > thresh && sig[i] > sig[i - 1] && sig[i] >= sig[i + 1])
        {
            if ((i - last) > (FS / 5))
            {
                beats++;
                last = i;
            }
        }
    }
    return beats;
}

/* ---------------- ECG Processing ---------------- */
static void process_window(float *win, int N)
{
    static float filtered[NORMAL_WINDOW];
    for (int i = 0; i < N; i++)
    {
        float v = win[i];
        v = biquad_process(&bp_bq, v);
        v = biquad_process(&notch_bq, v);
        filtered[i] = v;
    }
    haar_denoise(filtered, N, 0.02f);

    int beats = detect_qrs(filtered, N);
    float sec = (float)N / FS;
    int bpm = (sec > 0) ? (int)((beats * 60.0f) / sec + 0.5f) : 0;

    if (beats == 0 || bpm < 30 || bpm > 180)
    {
        abnormal = 1;
        window_size = ALERT_WINDOW;
    }
    else
    {
        abnormal = 0;
        window_size = NORMAL_WINDOW;
    }
}

/* ---------------- LED Logic ---------------- */
static void update_leds(void)
{
    if ((msTicks - last_data_time_ms) > NODATA_TIMEOUT)
    {
        GPIOC->ODR &= ~(1U << 13); // PC13 ON (active low)
        GPIOC->ODR &= ~(1U << 14); // PC14 OFF
        GPIOB->ODR |= (1U << 9);   // PB9 ON
    }
    else if (abnormal)
    {
        GPIOC->ODR |= (1U << 13); // PC13 OFF
        GPIOC->ODR |= (1U << 14); // PC14 ON
        GPIOB->ODR &= ~(1U << 9); // PB9 OFF
    }
    else
    {
        GPIOC->ODR |= (1U << 13);  // PC13 OFF
        GPIOC->ODR &= ~(1U << 14); // PC14 OFF
        GPIOB->ODR &= ~(1U << 9);  // PB9 OFF
    }
}

/* ---------------- Main ---------------- */
int main(void)
{
    SysTick_Config(SystemCoreClock / 1000U);
    GPIO_LED_Init();
    filters_init();

    // Warm-up 0–5s: all OFF
    delay_ms(WARMUP_MS);

    // Startup indication 5–7s: all ON
    GPIOC->ODR &= ~(1U << 13);
    GPIOC->ODR |= (1U << 14);
    GPIOB->ODR |= (1U << 9);
    delay_ms(STARTUP_MS);

    // Back to OFF
    GPIOC->ODR |= (1U << 13);
    GPIOC->ODR &= ~(1U << 14);
    GPIOB->ODR &= ~(1U << 9);

    last_sample_time_ms = msTicks;
    last_data_time_ms = msTicks;
    sim_switch_time = msTicks;

    int idx = 0;

    while (1)
    {
        if ((msTicks - last_sample_time_ms) >= (1000 / FS))
        {
            float v = ECG_ReadSimulated();
            ecg_window[idx++] = v;
            last_sample_time_ms = msTicks;
            if (v != 0.0f)
                last_data_time_ms = msTicks;

            if (idx >= window_size)
            {
                process_window(ecg_window, window_size);
                idx = 0;
            }
        }
        update_leds();
    }
}

/*
 * ECG module output is connected to *PA0* (ADC1, Channel 0).
 * 12-bit ADC, Vref = 3.3V.
 * ECG signal biased around 1.65V (typical for AD8232).
 */

/*
 * stm32_ecg_realtime.c
 * STM32F4 ECG Processing - Real-Time ADC Input
 * - Reads ECG from ADC1 Channel 0 (PA0)
 * - Preprocessing (bandpass + notch + wavelet)
 * - QRS detection + BPM
 * - Adaptive window (5s normal / 3s abnormal)
 * - LED logic:
 * * 0–5s warm-up: all OFF
 * * 5–7s startup: all ON
 * * Normal: all OFF
 * * Abnormal: PC14 ON
 * * No data ≥2s: PC13 + PB9 ON
 */
/*
#include "stm32f4xx.h" // Include STM32F4 standard peripheral library
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For floating-point math functions like expf, powf, sqrtf, and fabs
*/
/* ---------------- Config ---------------- */
/*#define FS                250 // Sampling frequency in Hz
#define NORMAL_WINDOW     (5*FS) // Normal processing window size (5 seconds)
#define ALERT_WINDOW      (3*FS) // Abnormal (alert) processing window size (3 seconds)
#define STARTUP_MS        2000 // Duration of startup LED sequence (2 seconds)
#define WARMUP_MS         5000 // Duration of initial warm-up (5 seconds)
#define NODATA_TIMEOUT    2000 // Timeout for no data detection in ms (2 seconds)
/*
/* ---------------- Globals ---------------- */
/*volatile uint32_t msTicks = 0; // Global millisecond tick counter, updated by SysTick
static float ecg_window[NORMAL_WINDOW]; // Array to store ECG samples for processing
static int window_size = NORMAL_WINDOW; // Current size of the processing window
volatile uint32_t last_sample_time_ms = 0; // Timestamp of the last sample read
volatile uint32_t last_data_time_ms   = 0; // Timestamp of the last non-zero data
int abnormal = 0; // Flag to indicate an abnormal condition

/* ---------------- SysTick ---------------- */
// SysTick_Handler is an interrupt handler automatically called by the SysTick timer
/*void SysTick_Handler(void) { msTicks++; } // Increment global millisecond counter on each SysTick interrupt
// Simple busy-wait delay function
static void delay_ms(uint32_t ms) {
    uint32_t start = msTicks;
    while ((msTicks - start) < ms) { __NOP(); } // Use a no-operation to prevent optimization
}

/* ---------------- GPIO LEDs ---------------- */
// Initializes the GPIO pins for the LEDs
/*static void GPIO_LED_Init(void) {
    RCC->AHB1ENR |= (1U<<2)|(1U<<1); // Enable clock for GPIOC and GPIOB
    // Clear and set the MODER registers to configure PC13 and PC14 as general purpose output (01)
    GPIOC->MODER &= ~((3U<<(13*2))|(3U<<(14*2)));
    GPIOC->MODER |=  ((1U<<(13*2))|(1U<<(14*2)));
    // Clear and set MODER for PB9 as general purpose output (01)
    GPIOB->MODER &= ~(3U<<(9*2));
    GPIOB->MODER |=  (1U<<(9*2));

    // Set initial LED states to OFF
    GPIOC->ODR |= (1U<<13);   // PC13 OFF (active low)
    GPIOC->ODR &= ~(1U<<14);  // PC14 OFF (active high)
    GPIOB->ODR &= ~(1U<<9);   // PB9 OFF (active high)
}

/* ---------------- ADC Init ---------------- */
// Initializes the ADC peripheral
/*static void ADC1_Init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;     // Enable clock for ADC1
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;    // Enable clock for GPIOA

    // Configure GPIOA pin 0 (PA0) to analog mode (MODER = 11)
    GPIOA->MODER |= (3U << (0 * 2));

    ADC1->CR2 = 0; // Reset control register
    ADC1->SQR3 = 0; // Set ADC sequence to sample channel 0 first and only
    // Set sample time for channel 0 to maximum (480 cycles) for accuracy
    ADC1->SMPR2 |= (7U << 0);
    ADC1->CR2 |= ADC_CR2_ADON; // Enable the ADC peripheral
}

// Performs a single ADC conversion
static uint16_t ADC1_Read(void) {
    ADC1->CR2 |= ADC_CR2_SWSTART; // Start ADC conversion using software
    while (!(ADC1->SR & ADC_SR_EOC)); // Wait for End of Conversion flag to be set
    return (uint16_t)ADC1->DR; // Return the 16-bit ADC data register value
}

// Reads ECG data from the ADC and performs voltage conversion and centering
static float ECG_ReadADC(void) {
    uint16_t raw = ADC1_Read(); // Get raw 12-bit ADC value (0-4095)
    // Convert the raw value to a voltage, assuming Vref = 3.3V
    float voltage = (3.3f * raw) / 4095.0f;
    // Center the signal around 0 by subtracting the typical bias voltage
    return voltage - 1.65f;
}*/

/* ---------------- Filters ---------------- */
// Structure for a Biquad IIR filter
/*typedef struct { float b0,b1,b2,a1,a2; float z1,z2; } Biquad;
static Biquad bp_bq, notch_bq; // Instances for bandpass and notch filters

// Processes a single sample through a Biquad filter using Direct Form II
static inline float biquad_process(Biquad *b, float x) {
    float y = b->b0*x + b->z1;
    b->z1 = b->b1*x - b->a1*y + b->z2;
    b->z2 = b->b2*x - b->a2*y;
    return y;
}
// Initializes filter coefficients
static void filters_init(void) {
    // Bandpass filter coefficients
    bp_bq = (Biquad){0.0675f, 0.0f, -0.0675f, -1.1380f, 0.4866f, 0,0};
    // Notch filter coefficients (e.g., for 60Hz hum)
    notch_bq = (Biquad){0.98799f, -1.8741f, 0.98799f, -1.8741f, 0.97599f, 0,0};
}*/

/* ---------------- Haar Wavelet ---------------- */
// Performs a single-level Haar wavelet denoise
/*static void haar_denoise(float *x, int N, float thr) {
    static float approx[NORMAL_WINDOW/2], detail[NORMAL_WINDOW/2]; // Wavelet coefficients
    int half = N/2;
    // Forward transform: calculate approximation and detail coefficients
    for (int i=0;i<half;i++) {
        approx[i] = (x[2*i] + x[2*i+1])*0.7071f;
        detail[i] = (x[2*i] - x[2*i+1])*0.7071f;
    }
    // Thresholding: apply soft-thresholding to detail coefficients
    for (int i=0;i<half;i++) {
        float d = detail[i];
        if (d > thr) detail[i] = d - thr;
        else if (d < -thr) detail[i] = d + thr;
        else detail[i] = 0.0f;
    }
    // Inverse transform: reconstruct the signal from the modified coefficients
    for (int i=0;i<half;i++) {
        x[2*i]   = (approx[i]+detail[i])*0.7071f;
        x[2*i+1] = (approx[i]-detail[i])*0.7071f;
    }
}*/

/* ---------------- QRS Detection ---------------- */
// Detects QRS complexes (heartbeats) in a given signal window
/*static int detect_qrs(float *sig,int N){
    float mean=0,var=0;
    // Calculate mean and variance of the signal
    for(int i=0;i<N;i++) mean+=sig[i];
    mean/=N;
    for(int i=0;i<N;i++){ float d=sig[i]-mean; var+=d*d; }
    float std=sqrtf(var/N); // Standard deviation
    // Set a dynamic threshold based on signal mean and standard deviation
    float thresh=mean+0.6f*std+0.05f;
    int beats=0,last=-FS/5; // Initialize beat count and last beat time
    // Iterate through the signal to find peaks above the threshold
    for(int i=1;i<N-1;i++){
        // A peak is found if the sample is above the threshold and a local maximum
        if(sig[i]>thresh && sig[i]>sig[i-1] && sig[i]>=sig[i+1]){
            // Check for refractory period to avoid double-counting
            if((i-last)>(FS/5)){ beats++; last=i; }
        }
    }
    return beats; // Return the number of detected beats
}*/

/* ---------------- ECG Processing ---------------- */
// Main signal processing function
/*static void process_window(float *win,int N){
    static float filtered[NORMAL_WINDOW];
    // Apply bandpass and notch filters to the entire window
    for (int i=0;i<N;i++) {
        float v = win[i];
        v = biquad_process(&bp_bq, v);
        v = biquad_process(&notch_bq, v);
        filtered[i] = v;
    }
    // Apply Haar wavelet denoising
    haar_denoise(filtered, N, 0.02f);

    int beats=detect_qrs(filtered,N); // Detect beats in the filtered signal
    float sec=(float)N/FS; // Calculate window duration in seconds
    int bpm=(sec>0)?(int)((beats*60.0f)/sec+0.5f):0; // Calculate BPM

    // Check for abnormal conditions (no beats, or BPM out of range)
    if(beats==0 || bpm<30 || bpm>180){
        abnormal=1; // Set abnormal flag
        window_size=ALERT_WINDOW; // Switch to shorter window for quicker alerts
    } else {
        abnormal=0; // Clear abnormal flag
        window_size=NORMAL_WINDOW; // Switch back to normal window
    }
}*/

/* ---------------- LED Logic ---------------- */
// Updates LED states based on system status
/*static void update_leds(void){
    // Check for "no data" condition
    if((msTicks-last_data_time_ms)>NODATA_TIMEOUT){
        GPIOC->ODR &= ~(1U<<13); // PC13 ON (active low)
        GPIOC->ODR &= ~(1U<<14); // PC14 OFF
        GPIOB->ODR |= (1U<<9);   // PB9 ON
    }
    // Check for "abnormal" condition
    else if(abnormal){
        GPIOC->ODR |= (1U<<13);  // PC13 OFF
        GPIOC->ODR |= (1U<<14);  // PC14 ON
        GPIOB->ODR &= ~(1U<<9);  // PB9 OFF
    }
    // Normal state
    else {
        GPIOC->ODR |= (1U<<13);  // PC13 OFF
        GPIOC->ODR &= ~(1U<<14);// PC14 OFF
        GPIOB->ODR &= ~(1U<<9); // PB9 OFF
    }
}*/

/* ---------------- Main ---------------- */
// Main function of the program
/*int main(void){
    SysTick_Config(SystemCoreClock/1000U); // Configure SysTick to generate interrupts every 1ms
    GPIO_LED_Init(); // Initialize LEDs
    filters_init(); // Initialize filter coefficients
    ADC1_Init(); // Initialize the ADC peripheral

    // Warm-up phase: LEDs are all OFF
    delay_ms(WARMUP_MS);

    // Startup phase: LEDs are all ON to indicate initialization
    GPIOC->ODR &= ~(1U<<13);
    GPIOC->ODR |=  (1U<<14);
    GPIOB->ODR |=  (1U<<9);
    delay_ms(STARTUP_MS);

    // After startup, set LEDs back to OFF
    GPIOC->ODR |= (1U<<13);
    GPIOC->ODR &= ~(1U<<14);
    GPIOB->ODR &= ~(1U<<9);

    // Initialize timestamps
    last_sample_time_ms=msTicks;
    last_data_time_ms=msTicks;

    int idx=0; // Index for the ECG window buffer

    // Main processing loop
    while(1){
        // Check if it's time to take a new sample based on the sampling frequency
        if((msTicks-last_sample_time_ms)>=(1000/FS)){
            float v=ECG_ReadADC();  // Read a real ECG sample from the ADC
            ecg_window[idx++]=v; // Store the sample in the processing window
            last_sample_time_ms=msTicks; // Update the last sample timestamp
            // Update the last data timestamp only if a significant signal is present
            if(fabs(v) > 0.01f) last_data_time_ms=msTicks;

            // Check if the current window is full
            if(idx>=window_size){
                process_window(ecg_window,window_size); // Process the full window of data
                idx=0; // Reset the window index
            }
        }
        update_leds(); // Update the LED status
    }
}*/