/* stm32_ecg_final_led_swapped.c
 * Final STM32F4 ECG Processing
 * - ECG simulated generator (normal, flatline, abnormal)
 * - Preprocessing: bandpass + notch + wavelet
 * - QRS detection + BPM
 * - Adaptive windowing: 5s normal / 3s abnormal
 * - LED logic:
 *   * First 5s warm-up (all LEDs OFF)
 *   * Then startup indication: all 3 LEDs ON for 2s
 *   * PC13 + PB9 ON if 2s no data
 *   * PC14 ON if abnormal ECG
 *   * All OFF if normal ECG
 * - Mode switches every 5s
 */

#include "stm32f4xx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---------------- Config ---------------- */
#define FS               250
#define NORMAL_WINDOW    (5*FS)   // 5s = 1250 samples
#define ALERT_WINDOW     (3*FS)   // 3s = 750 samples
#define STARTUP_MS       2000     // all LEDs ON after warm-up
#define WARMUP_MS        5000     // 5s warm-up
#define NODATA_TIMEOUT   2000     // 2s no data = alert
#define SWITCH_INTERVAL  5000     // 5s per mode

/* ---------------- Globals ---------------- */
volatile uint32_t msTicks = 0;
static float ecg_window[NORMAL_WINDOW];
static int window_size = NORMAL_WINDOW;
volatile uint32_t last_sample_time_ms = 0;
volatile uint32_t last_data_time_ms   = 0;
int abnormal = 0;

typedef enum { ECG_NORMAL, ECG_FLATLINE, ECG_ABNORMAL } ECGMode;
ECGMode sim_mode = ECG_NORMAL;
uint32_t sim_switch_time = 0;

/* ---------------- SysTick ---------------- */
void SysTick_Handler(void) { msTicks++; }

/* ---------------- Delay ---------------- */
static void delay_ms(uint32_t ms) {
    uint32_t start = msTicks;
    while ((msTicks - start) < ms) { __NOP(); }
}

/* ---------------- GPIO LEDs ---------------- */
static void GPIO_LED_Init(void) {
    RCC->AHB1ENR |= (1U<<2)|(1U<<1); // GPIOC + GPIOB
    GPIOC->MODER &= ~((3U<<(13*2))|(3U<<(14*2)));
    GPIOC->MODER |=  ((1U<<(13*2))|(1U<<(14*2)));
    GPIOB->MODER &= ~(3U<<(9*2));
    GPIOB->MODER |=  (1U<<(9*2));

    // All OFF at startup
    GPIOC->ODR |= (1U<<13);   // PC13 off (active low)
    GPIOC->ODR &= ~(1U<<14);  // PC14 off
    GPIOB->ODR &= ~(1U<<9);   // PB9 off
}

/* ---------------- ECG Template ---------------- */
static float ECG_Template(float t) {
    return (
        0.1f*expf(-powf((t-0.2f)/0.05f,2)) +
       -0.15f*expf(-powf((t-0.3f)/0.015f,2)) +
        1.2f*expf(-powf((t-0.32f)/0.01f,2)) +
       -0.25f*expf(-powf((t-0.34f)/0.015f,2)) +
        0.3f*expf(-powf((t-0.55f)/0.1f,2))
    );
}

/* ---------------- ECG Simulator ---------------- */
static float ECG_ReadSimulated(void) {
    static float t = 0.0f;

    // Switch scenarios every 5s
    if (msTicks - sim_switch_time > SWITCH_INTERVAL) {
        if (sim_mode == ECG_NORMAL) sim_mode = ECG_FLATLINE;
        else if (sim_mode == ECG_FLATLINE) sim_mode = ECG_NORMAL;
        else if (sim_mode == ECG_NORMAL) sim_mode = ECG_ABNORMAL;
        sim_switch_time = msTicks;
    }

    if (sim_mode == ECG_FLATLINE) {
        return 0.0f; // no data
    }
    else if (sim_mode == ECG_ABNORMAL) {
        t += 1.0f/FS;
        return 0.5f*sinf(2*3.14159f*3.0f*t); // tachy-like
    }
    else { // ECG_NORMAL
        t += 1.0f/FS;
        float base = ECG_Template(fmodf(t,0.8f)); // ~75 BPM
        base += 0.02f*((rand()%100)/100.0f-0.5f);
        return base;
    }
}

/* ---------------- Filters ---------------- */
typedef struct { float b0,b1,b2,a1,a2; float z1,z2; } Biquad;
static Biquad bp_bq, notch_bq;

static inline float biquad_process(Biquad *b, float x) {
    float y = b->b0*x + b->z1;
    b->z1 = b->b1*x - b->a1*y + b->z2;
    b->z2 = b->b2*x - b->a2*y;
    return y;
}

static void filters_init(void) {
    // Example coefficients (replace with designed ones!)
    bp_bq = (Biquad){0.0675f, 0.0f, -0.0675f, -1.1380f, 0.4866f, 0,0};
    notch_bq = (Biquad){0.98799f, -1.8741f, 0.98799f, -1.8741f, 0.97599f, 0,0};
}

/* ---------------- Haar DWT ---------------- */
static void haar_denoise(float *x, int N, float thr) {
    static float approx[NORMAL_WINDOW/2], detail[NORMAL_WINDOW/2];
    int half = N/2;
    for (int i=0;i<half;i++) {
        approx[i] = (x[2*i] + x[2*i+1])*0.7071f;
        detail[i] = (x[2*i] - x[2*i+1])*0.7071f;
    }
    for (int i=0;i<half;i++) {
        float d = detail[i];
        if (d > thr) detail[i] = d - thr;
        else if (d < -thr) detail[i] = d + thr;
        else detail[i] = 0.0f;
    }
    for (int i=0;i<half;i++) {
        x[2*i]   = (approx[i]+detail[i])*0.7071f;
        x[2*i+1] = (approx[i]-detail[i])*0.7071f;
    }
}

/* ---------------- QRS Detection ---------------- */
static int detect_qrs(float *sig,int N){
    float mean=0,var=0;
    for(int i=0;i<N;i++) mean+=sig[i];
    mean/=N;
    for(int i=0;i<N;i++){ float d=sig[i]-mean; var+=d*d; }
    float std=sqrtf(var/N);
    float thresh=mean+0.6f*std+0.05f;
    int beats=0,last=-FS/5;
    for(int i=1;i<N-1;i++){
        if(sig[i]>thresh && sig[i]>sig[i-1] && sig[i]>=sig[i+1]){
            if((i-last)>(FS/5)){ beats++; last=i; }
        }
    }
    return beats;
}

/* ---------------- ECG Processing ---------------- */
static void process_window(float *win,int N){
    static float filtered[NORMAL_WINDOW];
    for (int i=0;i<N;i++) {
        float v = win[i];
        v = biquad_process(&bp_bq, v);
        v = biquad_process(&notch_bq, v);
        filtered[i] = v;
    }
    haar_denoise(filtered, N, 0.02f);

    int beats=detect_qrs(filtered,N);
    float sec=(float)N/FS;
    int bpm=(sec>0)?(int)((beats*60.0f)/sec+0.5f):0;

    if(beats==0 || bpm<30 || bpm>180){
        abnormal=1;
        window_size=ALERT_WINDOW;
    } else {
        abnormal=0;
        window_size=NORMAL_WINDOW;
    }
}

/* ---------------- LED Logic ---------------- */
static void update_leds(void){
    if((msTicks-last_data_time_ms)>NODATA_TIMEOUT){
        // No data ≥2s → PC13 + PB9 ON
        GPIOC->ODR &= ~(1U<<13); // PC13 ON (active low)
        GPIOC->ODR &= ~(1U<<14); // PC14 OFF
        GPIOB->ODR |= (1U<<9);   // PB9 ON
    }
    else if(abnormal){
        // Abnormal → PC14 ON
        GPIOC->ODR |= (1U<<13);  // PC13 OFF
        GPIOC->ODR |= (1U<<14);  // PC14 ON
        GPIOB->ODR &= ~(1U<<9);  // PB9 OFF
    }
    else {
        // Normal → all OFF
        GPIOC->ODR |= (1U<<13);
        GPIOC->ODR &= ~(1U<<14);
        GPIOB->ODR &= ~(1U<<9);
    }
}

/* ---------------- Main ---------------- */
int main(void){
    SysTick_Config(SystemCoreClock/1000U);
    GPIO_LED_Init();
    filters_init();

    // ---- Warm-up first (5s, all OFF) ----
    GPIOC->ODR |= (1U<<13);
    GPIOC->ODR &= ~(1U<<14);
    GPIOB->ODR &= ~(1U<<9);
    delay_ms(WARMUP_MS);

    // ---- Startup indication: all ON for 2s ----
    GPIOC->ODR &= ~(1U<<13); // PC13 ON (active low)
    GPIOC->ODR |=  (1U<<14); // PC14 ON
    GPIOB->ODR |=  (1U<<9);  // PB9 ON
    delay_ms(STARTUP_MS);

    // Turn OFF after indication
    GPIOC->ODR |= (1U<<13);
    GPIOC->ODR &= ~(1U<<14);
    GPIOB->ODR &= ~(1U<<9);

    last_sample_time_ms=msTicks;
    last_data_time_ms=msTicks;
    sim_switch_time=msTicks;

    int idx=0;

    while(1){
        if((msTicks-last_sample_time_ms)>=(1000/FS)){
            float v=ECG_ReadSimulated();
            ecg_window[idx++]=v;
            last_sample_time_ms=msTicks;
            if(v!=0.0f) last_data_time_ms=msTicks;

            if(idx>=window_size){
                process_window(ecg_window,window_size);
                idx=0;
            }
        }
        update_leds();
    }
}