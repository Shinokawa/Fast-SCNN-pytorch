#include "motor.h"
#include "stm32f10x.h"
uint16_t g_speed = 500;

void Motor_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;

    //GPIO
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
    //FR
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

//ST
GPIO_InitStructure.GPIO_Pin = ST_PIN;
GPIO_Init(ST_PORT, &GPIO_InitStructure);

    // 右前轮方向控制
    GPIO_InitStructure.GPIO_Pin = FR_RF_PIN;
    GPIO_Init(FR_RF_PORT, &GPIO_InitStructure);

    // 左后轮方向控制
    GPIO_InitStructure.GPIO_Pin = FR_LR_PIN;
    GPIO_Init(FR_LR_PORT, &GPIO_InitStructure);

    // 右后轮方向控制
    GPIO_InitStructure.GPIO_Pin = FR_RR_PIN;
    GPIO_Init(FR_RR_PORT, &GPIO_InitStructure);

    // 左前轮方向控制
    GPIO_InitStructure.GPIO_Pin = FR_LF_PIN;
    GPIO_Init(FR_LF_PORT, &GPIO_InitStructure);
		
		
}

void Motor_PWM_Init(void)
{
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_OCInitTypeDef TIM_OCInitStructure;
    GPIO_InitTypeDef GPIO_InitStructure;
 
	    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);

GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

// CH1 PA6 - 右前轮PWM
// CH2 PA7 - 左后轮PWM
GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_7;
GPIO_Init(GPIOA, &GPIO_InitStructure);

// CH3 PB0 - 右后轮PWM
// CH4 PB1 - 左前轮PWM
GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1;
GPIO_Init(GPIOB, &GPIO_InitStructure);
	
    TIM_TimeBaseStructure.TIM_Period = 1000 - 1;        
    TIM_TimeBaseStructure.TIM_Prescaler = 72 - 1;       
    TIM_TimeBaseStructure.TIM_ClockDivision = 0;
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

 
    TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
    TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
    TIM_OCInitStructure.TIM_Pulse = g_speed;
    TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;

    //CH1 -> PA6 (右前轮PWM)
    TIM_OC1Init(TIM3, &TIM_OCInitStructure);
    TIM_OC1PreloadConfig(TIM3, TIM_OCPreload_Enable);

    //CH2 -> PA7 (左后轮PWM)
    TIM_OC2Init(TIM3, &TIM_OCInitStructure);
    TIM_OC2PreloadConfig(TIM3, TIM_OCPreload_Enable);

    //CH3 -> PB0 (右后轮PWM)
    TIM_OC3Init(TIM3, &TIM_OCInitStructure);
    TIM_OC3PreloadConfig(TIM3, TIM_OCPreload_Enable);

    //CH4 -> PB1 (左前轮PWM)
    TIM_OC4Init(TIM3, &TIM_OCInitStructure);
    TIM_OC4PreloadConfig(TIM3, TIM_OCPreload_Enable);

    TIM_Cmd(TIM3, ENABLE);
}
void Motor_SetSpeed(uint16_t speed)
{
	if (speed > 1000) speed = 1000;
	g_speed = speed;

	TIM_SetCompare1(TIM3, speed);
  TIM_SetCompare2(TIM3, speed);
  TIM_SetCompare3(TIM3, speed);
  TIM_SetCompare4(TIM3, speed);
}
void Motor_Enable(uint8_t enable)
{
    if (enable)
        GPIO_SetBits(ST_PORT, ST_PIN);
    else
        GPIO_ResetBits(ST_PORT, ST_PIN);
}

void Motor_SetDirection(uint8_t dir)
{
    switch(dir)
    {
        case DIR_FORWARD:
            // 前进：所有轮子都前进
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);      
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN);      
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);   
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);    
            // 所有轮子相同速度
            TIM_SetCompare1(TIM3, g_speed);  // 右前轮
            TIM_SetCompare2(TIM3, g_speed);  // 左后轮
            TIM_SetCompare3(TIM3, g_speed);  // 右后轮
            TIM_SetCompare4(TIM3, g_speed);  // 左前轮
            break;
            
        case DIR_BACK:
            // 后退：所有轮子都后退
            GPIO_ResetBits(FR_RF_PORT, FR_RF_PIN);   
            GPIO_ResetBits(FR_LR_PORT, FR_LR_PIN);    
            GPIO_SetBits(FR_RR_PORT, FR_RR_PIN);     
            GPIO_SetBits(FR_LF_PORT, FR_LF_PIN);  
            // 所有轮子相同速度
            TIM_SetCompare1(TIM3, g_speed);  // 右前轮
            TIM_SetCompare2(TIM3, g_speed);  // 左后轮
            TIM_SetCompare3(TIM3, g_speed);  // 右后轮
            TIM_SetCompare4(TIM3, g_speed);  // 左前轮
            break;
            
        case DIR_LEFT:
        {
            // 左转：差速转向，右侧轮子速度快，左侧轮子速度慢
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);   
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN); 
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN); 
            // 右侧轮子速度快，左侧轮子速度慢
            uint16_t left_speed = g_speed * 1 / 4;  // 左侧轮子25%速度
            uint16_t right_speed = g_speed;         // 右侧轮子100%速度
            TIM_SetCompare1(TIM3, right_speed);  // 右前轮
            TIM_SetCompare2(TIM3, left_speed);   // 左后轮
            TIM_SetCompare3(TIM3, right_speed);  // 右后轮
            TIM_SetCompare4(TIM3, left_speed);   // 左前轮
            break;
        }
            
        case DIR_RIGHT:
        {
            // 右转：差速转向，左侧轮子速度快，右侧轮子速度慢
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN);
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);
            // 左侧轮子速度快，右侧轮子速度慢
            uint16_t left_speed_fast = g_speed;         // 左侧轮子100%速度
            uint16_t right_speed_slow = g_speed * 1 / 4; // 右侧轮子25%速度
            TIM_SetCompare1(TIM3, right_speed_slow);  // 右前轮
            TIM_SetCompare2(TIM3, left_speed_fast);   // 左后轮
            TIM_SetCompare3(TIM3, right_speed_slow);  // 右后轮
            TIM_SetCompare4(TIM3, left_speed_fast);   // 左前轮
            break;
        }
            
        default:
            break;
    }
}

void Motor_SetDirectionWithSpeed(uint8_t dir, uint16_t speed)
{
    // 限制速度范围
    if (speed > 1000) speed = 1000;
    
    switch(dir)
    {
        case DIR_FORWARD:
            // 前进：所有轮子都前进
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);      
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN);      
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);   
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);    
            // 所有轮子相同速度
            TIM_SetCompare1(TIM3, speed);  // 右前轮
            TIM_SetCompare2(TIM3, speed);  // 左后轮
            TIM_SetCompare3(TIM3, speed);  // 右后轮
            TIM_SetCompare4(TIM3, speed);  // 左前轮
            break;
            
        case DIR_BACK:
            // 后退：所有轮子都后退
            GPIO_ResetBits(FR_RF_PORT, FR_RF_PIN);   
            GPIO_ResetBits(FR_LR_PORT, FR_LR_PIN);    
            GPIO_SetBits(FR_RR_PORT, FR_RR_PIN);     
            GPIO_SetBits(FR_LF_PORT, FR_LF_PIN);  
            // 所有轮子相同速度
            TIM_SetCompare1(TIM3, speed);  // 右前轮
            TIM_SetCompare2(TIM3, speed);  // 左后轮
            TIM_SetCompare3(TIM3, speed);  // 右后轮
            TIM_SetCompare4(TIM3, speed);  // 左前轮
            break;
            
        case DIR_LEFT:
        {
            // 左转：差速转向，右侧轮子速度快，左侧轮子停止
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);   
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN); 
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN); 
            // 右侧轮子前进，左侧轮子慢速
            uint16_t left_speed = speed * 0 / 10;  // 左侧轮子20%速度
            uint16_t right_speed = speed;         // 右侧轮子100%速度
            TIM_SetCompare1(TIM3, right_speed);  // 右前轮
            TIM_SetCompare2(TIM3, left_speed);   // 左后轮
            TIM_SetCompare3(TIM3, right_speed);  // 右后轮
            TIM_SetCompare4(TIM3, left_speed);   // 左前轮
            break;
        }
            
        case DIR_RIGHT:
        {
            // 右转：差速转向，左侧轮子前进，右侧轮子停止
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN);
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);
            // 左侧轮子前进，右侧轮子慢速
            uint16_t left_speed_fast = speed;         // 左侧轮子100%速度
            uint16_t right_speed_slow = speed * 0 / 10; // 右侧轮子20%速度
            TIM_SetCompare1(TIM3, right_speed_slow);  // 右前轮
            TIM_SetCompare2(TIM3, left_speed_fast);   // 左后轮
            TIM_SetCompare3(TIM3, right_speed_slow);  // 右后轮
            TIM_SetCompare4(TIM3, left_speed_fast);   // 左前轮
            break;
        }
            
        default:
            break;
    }
}

void Motor_SetDifferentialSpeed(uint16_t left_speed, uint16_t right_speed)
{
    // 限制速度范围
    if (left_speed > 1000) left_speed = 1000;
    if (right_speed > 1000) right_speed = 1000;
    
    // 设置所有轮子前进方向
    GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);   
    GPIO_SetBits(FR_LR_PORT, FR_LR_PIN); 
    GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
    GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN); 
    
    // 设置差速
    TIM_SetCompare1(TIM3, right_speed);  // 右前轮
    TIM_SetCompare2(TIM3, left_speed);   // 左后轮
    TIM_SetCompare3(TIM3, right_speed);  // 右后轮
    TIM_SetCompare4(TIM3, left_speed);   // 左前轮
}
void Motor_Test_Single(uint8_t motor)
{
    // 先停止所有电机
    Motor_SetSpeed(0);
    
    // 设置中等速度
    uint16_t test_speed = 500;
    
    switch(motor)
    {
        case 1: // 右前轮
            // 设置右前轮前进
            GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);
            // 停止其他轮子
            GPIO_ResetBits(FR_LR_PORT, FR_LR_PIN);
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);
            // 只给右前轮PWM
            TIM_SetCompare1(TIM3, test_speed);
            TIM_SetCompare2(TIM3, 0);
            TIM_SetCompare3(TIM3, 0);
            TIM_SetCompare4(TIM3, 0);
            break;
            
        case 2: // 左后轮
            // 设置左后轮前进
            GPIO_SetBits(FR_LR_PORT, FR_LR_PIN);
            // 停止其他轮子
            GPIO_ResetBits(FR_RF_PORT, FR_RF_PIN);
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);
            // 只给左后轮PWM
            TIM_SetCompare1(TIM3, 0);
            TIM_SetCompare2(TIM3, test_speed);
            TIM_SetCompare3(TIM3, 0);
            TIM_SetCompare4(TIM3, 0);
            break;
            
        case 3: // 右后轮
            // 设置右后轮前进
            GPIO_SetBits(FR_RR_PORT, FR_RR_PIN);
            // 停止其他轮子
            GPIO_ResetBits(FR_RF_PORT, FR_RF_PIN);
            GPIO_ResetBits(FR_LR_PORT, FR_LR_PIN);
            GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN);
            // 只给右后轮PWM
            TIM_SetCompare1(TIM3, 0);
            TIM_SetCompare2(TIM3, 0);
            TIM_SetCompare3(TIM3, test_speed);
            TIM_SetCompare4(TIM3, 0);
            break;
            
        case 4: // 左前轮
            // 设置左前轮前进
            GPIO_SetBits(FR_LF_PORT, FR_LF_PIN);
            // 停止其他轮子
            GPIO_ResetBits(FR_RF_PORT, FR_RF_PIN);
            GPIO_ResetBits(FR_LR_PORT, FR_LR_PIN);
            GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
            // 只给左前轮PWM
            TIM_SetCompare1(TIM3, 0);
            TIM_SetCompare2(TIM3, 0);
            TIM_SetCompare3(TIM3, 0);
            TIM_SetCompare4(TIM3, test_speed);
            break;
            
        default:
            break;
    }
}

void Motor_Test_Differential(void)
{
    // 测试差速转向
    uint16_t test_speed = 500;
    
    // 设置所有轮子前进方向
    GPIO_SetBits(FR_RF_PORT, FR_RF_PIN);   
    GPIO_SetBits(FR_LR_PORT, FR_LR_PIN); 
    GPIO_ResetBits(FR_RR_PORT, FR_RR_PIN);
    GPIO_ResetBits(FR_LF_PORT, FR_LF_PIN); 
    
    // 设置明显的差速：左侧50%，右侧100%
    uint16_t left_speed = test_speed / 2;   // 50%
    uint16_t right_speed = test_speed;      // 100%
    
    TIM_SetCompare1(TIM3, right_speed);  // 右前轮
    TIM_SetCompare2(TIM3, left_speed);   // 左后轮
    TIM_SetCompare3(TIM3, right_speed);  // 右后轮
    TIM_SetCompare4(TIM3, left_speed);   // 左前轮
}
