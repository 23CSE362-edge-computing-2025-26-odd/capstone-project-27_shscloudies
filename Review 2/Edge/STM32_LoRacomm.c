#include <stm32f4xx.h>

#define LED_PIN 13
#define LED_PORT GPIOC

void SPI1_Init(void){
    RCC->APB2ENR |= (1U<<12); 
    RCC->AHB1ENR |= (1U<<0);  

   
    GPIOA->MODER &= ~((3U<<10)|(3U<<12)|(3U<<14));
    GPIOA->MODER |=  ((2U<<10)|(2U<<12)|(2U<<14));
    GPIOA->AFR[0] &= ~((0xF<<20)|(0xF<<24)|(0xF<<28));
    GPIOA->AFR[0] |=  ((5<<20)|(5<<24)|(5<<28));

    SPI1->CR1 = (1U<<2)|(1U<<1); 
    SPI1->CR1 |= (1U<<6);       
}

uint8_t SPI1_Transmit(uint8_t data){
    SPI1->DR = data;
    while(!(SPI1->SR & (1U<<1)));
    while(SPI1->SR & (1U<<7));   
    return SPI1->DR;
}

int main(void){
    RCC->AHB1ENR |= (1U<<2); // GPIOC
    GPIOC->MODER &= ~(3U<<(2*LED_PIN));
    GPIOC->MODER |=  (1U<<(2*LED_PIN));

    SPI1_Init();

    while(1){
        SPI1_Transmit(0x55);      
        GPIOC->ODR ^= (1U<<LED_PIN); 
        for(volatile int i=0;i<500000;i++);
    }
}
