#ifndef __USART_H
#define __USART_H

#include "stm32f10x.h"
#define USART1_REC_LEN 5
extern uint8_t USART1_RX_BUF[USART1_REC_LEN];
extern uint8_t USART1_RX_LEN ;
void USART1_Init(uint32_t bound);
void RS232_Send_Data(u8 *buf, u8 len);
#endif
