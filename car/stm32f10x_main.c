#include "stm32f10x.h"
#include "motor.h"
#include "usart.h"

uint16_t NowSpeed = 500;
uint8_t CurrentDirection = 0xFF; // 0xFF表示停止状态

#include "stm32f10x.h"
#include "motor.h"
#include "usart.h"

uint16_t NowSpeed = 500;
uint8_t CurrentDirection = 0xFF; // 0xFF表示停止状态

// 新增：解析自定义协议
void TryParseWheelSpeedPacket(void) {
    if (USART1_RX_LEN == 7 && USART1_RX_BUF[0] == 0xAA && USART1_RX_BUF[6] == 0x55) {
        int16_t left = (USART1_RX_BUF[2] << 8) | USART1_RX_BUF[1];
        int16_t right = (USART1_RX_BUF[4] << 8) | USART1_RX_BUF[3];
        uint8_t checksum = (USART1_RX_BUF[1] + USART1_RX_BUF[2] + USART1_RX_BUF[3] + USART1_RX_BUF[4]) & 0xFF;
        if (checksum == USART1_RX_BUF[5]) {
            // 方向控制
            if (left >= 0 && right >= 0) {
                // 前进
                Motor_SetDifferentialSpeed(left, right);
                // 设置所有轮子前进方向
                // ...（如原有前进方向引脚设置）
            } else if (left <= 0 && right <= 0) {
                // 后退
                Motor_SetDifferentialSpeed(-left, -right);
                // 设置所有轮子后退方向
                // ...（如原有后退方向引脚设置）
            } else {
                // 差速转向（如原地转等）
                // 左正右负：原地右转，左负右正：原地左转
                Motor_SetDifferentialSpeed(abs(left), abs(right));
                // 设置左右轮方向
                // ...（根据left/right正负分别设置左右轮方向引脚）
            }
            USART1_RX_LEN = 0;
        }
    }
}

int main(void)
{
    Motor_GPIO_Init();
    Motor_PWM_Init();
    Motor_Enable(1);
    USART1_Init(9600);

    Motor_SetSpeed(0);
    RS232_Send_Data((u8*)"System Ready - Stop Mode\r\n", 25);
    RS232_Send_Data((u8*)"Commands: w(forward), s(back), a(left), d(right), h(speed+), l(speed-), p(stop), t(test)\r\n", 80);

    while(1)
    {
        // 优先处理自定义协议
        TryParseWheelSpeedPacket();

        // 兼容wasd
        if (USART1_RX_LEN)
        {
            u8 cmd = USART1_RX_BUF[0];
            // ...（原有wasd switch代码）
            USART1_RX_LEN=0;
        }
    }
}

int main(void)
{
    Motor_GPIO_Init();
    Motor_PWM_Init();
    Motor_Enable(1);
    USART1_Init(9600);
    
    // 默认停止状态
    Motor_SetSpeed(0);
    RS232_Send_Data((u8*)"System Ready - Stop Mode\r\n", 25);
    RS232_Send_Data((u8*)"Commands: w(forward), s(back), a(left), d(right), h(speed+), l(speed-), p(stop), t(test)\r\n", 80);
    
    while(1)
    {
        if (USART1_RX_LEN)
        {
            
            u8 cmd = USART1_RX_BUF[0];

            switch(cmd)
            {
                case 'w':
                    Motor_Enable(1);
                    Motor_SetDirectionWithSpeed(DIR_FORWARD, NowSpeed);
                    CurrentDirection = DIR_FORWARD;
                    RS232_Send_Data((u8*)"Forward\r\n",9);
                    break;
                
                case 's':
                    Motor_Enable(1);
                    Motor_SetDirectionWithSpeed(DIR_BACK, NowSpeed);
                    CurrentDirection = DIR_BACK;
                    RS232_Send_Data((u8*)"Back\r\n",6);
                    break;
                
                case 'a':
                    Motor_Enable(1);
                    Motor_SetDirectionWithSpeed(DIR_LEFT, NowSpeed);
                    CurrentDirection = DIR_LEFT;
                    RS232_Send_Data((u8*)"Left Turn - Right:100%, Left:20%\r\n",32);
                    break;
                
                case 'd':
                    Motor_Enable(1);
                    Motor_SetDirectionWithSpeed(DIR_RIGHT, NowSpeed);
                    CurrentDirection = DIR_RIGHT;
                    RS232_Send_Data((u8*)"Right Turn - Left:100%, Right:20%\r\n",33);
                    break;
                
                case 'l'://lower speed
                    if (NowSpeed >= 50) NowSpeed -= 50;
                    // 如果当前在运动，重新应用当前方向和速度
                    if (CurrentDirection != 0xFF) {
                        Motor_SetDirectionWithSpeed(CurrentDirection, NowSpeed);
                    }
                    RS232_Send_Data((u8*)"Speed Down - Current:",20);
                    break;

                case 'h'://higher speed
                    if (NowSpeed < 950) NowSpeed += 50;
                    // 如果当前在运动，重新应用当前方向和速度
                    if (CurrentDirection != 0xFF) {
                        Motor_SetDirectionWithSpeed(CurrentDirection, NowSpeed);
                    }
                    RS232_Send_Data((u8*)"Speed Up - Current:",18);
                    break;
                
                case 'p'://stop
                    Motor_SetSpeed(0);
                    CurrentDirection = 0xFF; // 设置为停止状态
                    RS232_Send_Data((u8*)"Stop\r\n",6);
                    break;
                
                case 't'://test differential
                    Motor_Enable(1);
                    Motor_Test_Differential();
                    RS232_Send_Data((u8*)"Test Differential - Left:50%, Right:100%\r\n",40);
                    break;
                
                default:
                    break;
            }
            USART1_RX_LEN=0;
        }        
    }
}
