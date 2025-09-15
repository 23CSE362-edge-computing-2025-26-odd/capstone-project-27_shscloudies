#include <stm32f4xx.h>

#define LORA_NSS_PIN    8   // PB8
#define LED_PIN         13  // PC13

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

uint8_t SPI1_TransmitReceive(uint8_t data){
    SPI1->DR = data;
    while(!(SPI1->SR & (1U<<0))); 
    return SPI1->DR;
}

uint8_t LoRa_ReadReg(uint8_t addr){
    uint8_t value;
    GPIOB->ODR &= ~(1U<<LORA_NSS_PIN);
    SPI1_TransmitReceive(addr & 0x7F); 
    value = SPI1_TransmitReceive(0x00); 
    GPIOB->ODR |= (1U<<LORA_NSS_PIN);  
    return value;
}

int main(void){
    RCC->AHB1ENR |= (1U<<1)|(1U<<2); 

 
    GPIOC->MODER &= ~(3U<<(2*LED_PIN));
    GPIOC->MODER |=  (1U<<(2*LED_PIN));
    
    
    GPIOB->MODER &= ~(3U<<(2*LORA_NSS_PIN));
    GPIOB->MODER |=  (1U<<(2*LORA_NSS_PIN));
    GPIOB->ODR |= (1U<<LORA_NSS_PIN); 
    SPI1_Init();

    uint8_t version = LoRa_ReadReg(0x42); 

    if(version == 0x12){ 
        GPIOC->ODR &= ~(1U<<LED_PIN); 
    } else {
        GPIOC->ODR |= (1U<<LED_PIN);  
    }

    while(1){}
}
