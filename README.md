#wma decode
##1. overview
    将FFMPEG当中的wma decode部分进行移植,使得wmadecode使用于使用定点运算的平台.
    目标平台:stm32
##2. target
    初步目标是和STM32 audio engine -WMA decoder library技术指标相当
![xOpenLee](https://github.com/xOpenLee/wmaDecode/blob/master/image/wmaDecodeTarget.png)
##3. The ultimate goal
    Spirit DSP wma decode 
![xOpenLee](https://github.com/xOpenLee/wmaDecode/blob/master/image/finalTarget.png)
