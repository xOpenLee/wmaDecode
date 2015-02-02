#wma decode
##1. overview
    将RockBox当中的wma decode部分进行移植,使得wmadecode使用于使用定点运算的平台.
    目标平台:stm32
##2. target
    初步目标是和STM32 audio engine -WMA decoder library技术指标相当
![xOpenLee](https://github.com/xOpenLee/wmaDecode/blob/master/image/wmaDecodeTarget.png)
##3. The ultimate goal
    Spirit DSP wma decode 
![xOpenLee](https://github.com/xOpenLee/wmaDecode/blob/master/image/finalTarget.png)
##4. 优化方案
    优化方向参考以下论文进行.
    a.A Low- cost Multi-standard Audio Decoder Architecture Design,Tao Zhang, Fengping Yu, Haojun Quan
    b.Optimizing open source decoding program of WMA based on S3C2410 ZHANG Hai-bin,LI Hui
    c.Integer Optimization of WMA Audio Based on Embedded ARM Environment SU Yang , HU Shao-jiang, GUO Qian
###4.1. 浮点数转定点数
    需要将一个float 转换成定点数来表示,需要将定点数的存储空间看成是有浮点数的整数和小数部分组成.其中
    整数占的位数为i,小数占的位数为f,符号位为最高一位.因此,浮点数转定点数可以表示成
    F(i,f) = r*2^f,其中,i为整数位数,f为小数位数,r为浮点数.论文当中这个公式是值得讨论的.
    作者引用的链接是http://trac.bookofhook.com/bookofhook/trac.cgi/wiki/IntroductionToFixedPointMath
    浮点数的表示范围: -2^i <= F(i,f)<= 2^i - 2^-f,得出浮点数的取值范围与i的位数有关,精度于f的位数有关,而且是正相关.
    浮点数的四则运算:
    加法: FA(i,f) + FB(i, f) = rA*2^f + rB*2^f = (rA + rB)*2^f
    减法: FA(i,f) - FB(i, f) = rA*2^f - rB*2^f = (rA - rB)*2^f
    乘法: FA(i,f) * FB(i, f) = rA*2^f * rB*2^f = (rA * rB)*2^f*2^f,需要除以2^f才能得到正确的值
    除法: FA(i,f) / FB(i, f) = rA*2^f / rB*2^f = (rA / rB),需要乘以2^f才能得到正确的值
    乘除法都需要数据对齐,而且在除法当中需要先移位后进行除法操作

###4.2. 将除法进行转化
###4.3. 利用MDCT解决wma当中的sin cos sqrt运算
###4.4. 依据stm32硬件架构进行汇编级指令优化
