#ifndef __MOTOR_H
#define __MOTOR_H

#include "stm32f10x.h"

#define ST_PORT        GPIOA
#define ST_PIN         GPIO_Pin_4//st

// 方向控制引脚 - 按实际轮子位置命名
#define FR_RF_PORT      GPIOA  // 右前轮方向控制
#define FR_RF_PIN       GPIO_Pin_0

#define FR_LR_PORT      GPIOA  // 左后轮方向控制
#define FR_LR_PIN       GPIO_Pin_1

#define FR_RR_PORT      GPIOA  // 右后轮方向控制
#define FR_RR_PIN       GPIO_Pin_2

#define FR_LF_PORT      GPIOA  // 左前轮方向控制
#define FR_LF_PIN       GPIO_Pin_3

// PWM控制引脚 - 按实际轮子位置命名
#define PWM_RF_PORT     GPIOA  // 右前轮PWM控制
#define PWM_RF_PIN      GPIO_Pin_6

#define PWM_LR_PORT     GPIOA  // 左后轮PWM控制
#define PWM_LR_PIN      GPIO_Pin_7

#define PWM_RR_PORT     GPIOB  // 右后轮PWM控制
#define PWM_RR_PIN      GPIO_Pin_0

#define PWM_LF_PORT     GPIOB  // 左前轮PWM控制
#define PWM_LF_PIN      GPIO_Pin_1


#define DIR_FORWARD 0
#define DIR_BACK 1
#define DIR_LEFT 2
#define DIR_RIGHT 3



void Motor_GPIO_Init(void);
void Motor_PWM_Init(void);
void Motor_SetSpeed(uint16_t speed);
void Motor_SetDirection(uint8_t dir);
void Motor_SetDirectionWithSpeed(uint8_t dir, uint16_t speed);
void Motor_Enable(uint8_t enable);
void Motor_Test_Single(uint8_t motor);
void Motor_SetDifferentialSpeed(uint16_t left_speed, uint16_t right_speed);
void Motor_Test_Differential(void);
#endif
