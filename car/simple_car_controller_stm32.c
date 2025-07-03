#include "stm32f10x.h"
#include "motor.h"
#include "usart.h"
#include <string.h>

// 通信协议定义
#define PROTOCOL_HEADER 0xAA
#define PROTOCOL_TAIL   0x55

// 全局变量
int16_t g_left_wheel_speed = 0;   // 左轮组速度
int16_t g_right_wheel_speed = 0;  // 右轮组速度
uint8_t g_motor_enabled = 0;

// 函数声明
void ProcessSpeedCommand(int16_t left_speed, int16_t right_speed);
void SetWheelSpeeds(int16_t left_speed, int16_t right_speed);
uint8_t CalculateChecksum(uint8_t* data, uint8_t len);

int main(void)
{
    // 初始化硬件
    Motor_GPIO_Init();
    Motor_PWM_Init();
    USART1_Init(115200);  // 使用115200波特率
    
    // 初始化状态
    Motor_Enable(1);
    g_motor_enabled = 1;
    Motor_SetDirection(DIR_STOP);
    
    // 发送初始化完成信号
    RS232_Send_Data((u8*)"Simple Car Controller Ready\r\n", 28);
    
    while(1)
    {
        // 检查是否有新的串口数据
        if (USART1_RX_LEN >= 7)  // 最小包长度：头部(1) + 左轮速度(2) + 右轮速度(2) + 校验(1) + 尾部(1)
        {
            uint8_t* rx_buf = USART1_RX_BUF;
            
            // 检查协议头
            if (rx_buf[0] == PROTOCOL_HEADER && rx_buf[6] == PROTOCOL_TAIL)
            {
                // 解析左右轮速度（小端序）
                int16_t left_speed = (rx_buf[2] << 8) | rx_buf[1];
                int16_t right_speed = (rx_buf[4] << 8) | rx_buf[3];
                
                // 验证校验和
                uint8_t expected_checksum = CalculateChecksum(&rx_buf[1], 4);
                uint8_t received_checksum = rx_buf[5];
                
                if (expected_checksum == received_checksum)
                {
                    // 处理速度命令
                    ProcessSpeedCommand(left_speed, right_speed);
                }
                else
                {
                    // 校验和错误
                    RS232_Send_Data((u8*)"Checksum Error\r\n", 16);
                }
            }
            else
            {
                // 协议头或尾部错误
                RS232_Send_Data((u8*)"Protocol Error\r\n", 16);
            }
            
            // 清空接收缓冲区
            USART1_RX_LEN = 0;
        }
        
        // 处理紧急停止超时
        // 如果超过500ms没有收到命令，自动停止
        static uint32_t last_command_time = 0;
        if (HAL_GetTick() - last_command_time > 500 && 
            (g_left_wheel_speed != 0 || g_right_wheel_speed != 0))
        {
            SetWheelSpeeds(0, 0);
        }
    }
}

void ProcessSpeedCommand(int16_t left_speed, int16_t right_speed)
{
    // 限制速度范围
    if (left_speed > 1000) left_speed = 1000;
    if (left_speed < -1000) left_speed = -1000;
    if (right_speed > 1000) right_speed = 1000;
    if (right_speed < -1000) right_speed = -1000;
    
    // 设置轮子速度
    SetWheelSpeeds(left_speed, right_speed);
    
    // 更新最后命令时间
    last_command_time = HAL_GetTick();
}

void SetWheelSpeeds(int16_t left_speed, int16_t right_speed)
{
    // 更新全局变量
    g_left_wheel_speed = left_speed;
    g_right_wheel_speed = right_speed;
    
    // 设置左轮组（左前和左后）
    uint16_t left_pwm = abs(left_speed);
    if (left_speed >= 0)
    {
        // 前进
        GPIO_SetBits(FRA_PORT, FRA_PIN);
        GPIO_SetBits(FRB_PORT, FRB_PIN);
        GPIO_ResetBits(FRC_PORT, FRC_PIN);
        GPIO_ResetBits(FRD_PORT, FRD_PIN);
    }
    else
    {
        // 后退
        GPIO_ResetBits(FRA_PORT, FRA_PIN);
        GPIO_ResetBits(FRB_PORT, FRB_PIN);
        GPIO_ResetBits(FRC_PORT, FRC_PIN);
        GPIO_ResetBits(FRD_PORT, FRD_PIN);
    }
    
    // 设置右轮组（右前和右后）
    uint16_t right_pwm = abs(right_speed);
    if (right_speed >= 0)
    {
        // 前进（方向引脚已经在上面设置了）
    }
    else
    {
        // 后退（方向引脚已经在上面设置了）
    }
    
    // 设置PWM值
    TIM_SetCompare1(TIM3, left_pwm);   // 左前轮
    TIM_SetCompare2(TIM3, left_pwm);   // 左后轮
    TIM_SetCompare3(TIM3, right_pwm);  // 右前轮
    TIM_SetCompare4(TIM3, right_pwm);  // 右后轮
    
    // 发送确认信息
    char status_msg[64];
    sprintf(status_msg, "Speed: L=%d R=%d\r\n", left_speed, right_speed);
    RS232_Send_Data((u8*)status_msg, strlen(status_msg));
}

uint8_t CalculateChecksum(uint8_t* data, uint8_t len)
{
    uint8_t checksum = 0;
    for (int i = 0; i < len; i++)
    {
        checksum += data[i];
    }
    return checksum;
}

// 简单的延时函数
static void Delay_ms(uint32_t ms)
{
    uint32_t i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 7200; j++);
} 