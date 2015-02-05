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
###4.2.以时间换空间
    嵌入式设备和x86架构的普通电脑一样,嵌入式设备目的是剪裁资源使得能用最少的资源或者成本完成特定的功能.
    目标:首要减少ram,其次减少rom,在芯片当中,ram的面积是rom的3-4倍,成本ram比多rom多,具体数据无.
###4.2. 将除法进行转化
    一次32为的乘法最多需要6个周期,除法根据执行情况和输入操作数需要20-100个周期.需要将a/b=a*(1/b)=a*c转化
    需要将除以2的次方的转成移位操作,如a/2^n => a>>n
###4.3. 利用MDCT解决wma当中的sin cos sqrt运算
    wma中的大量sin cos sqrt函数需要指令周期数几百,需要通过特勒级数将起展开,转化成定点数的四则运算.
###4.4. 依据stm32硬件架构进行汇编级指令优化
##5. 使用mmap 
    进行文件操作,就不需要每次去read文件ptr = mmap(NULL, *size, PROT_READ|PROT_WRITE,
    MAP_PRIVATE, fd, 0)直接操作内存地址.
    通过#define AVERROR(E) (-(e)) 实现将返回值控制在负数
    三目运算: max = a > b? a:b;//表达式1?表达式2:表达式3;若表达式1为真
    ,则整个三目运算的结果是返回表达式2,反之.因为= 优先级低于三目运算,所以,此式正确.
    opaque 是对用户隐藏了数据结构,或者用户定义数据结构.
