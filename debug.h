/*************************************************************************
	> File Name: debug.h
	> Author: xOpenLee
	> Mail: 750haige@gmail.com 
	> Created Time: Sunday, January 25, 2015 PM04:04:47 HKT
 ************************************************************************/

#ifndef _DEBUG_H
#define _DEBUG_H

#include <stdio.h>

#define WMA_PRINTF(f...) printf(f)

/*printf debug info*/

#define WMA_DEBUG
#ifdef WMA_DEBUG
#define DEBUG(f...) WMA_PRINTF(f) 
#else 
#define DEBUG(f...) do{}while(0)
#endif

/*debug position*/
#define WMA_PDEBUG
#ifdef WMA_PDEBUG
#define PDEBUG() \
        do{\
           WMA_PRINTF("###INFO: at File = %s, Func = %s, Line = %d\r\n",\
                            __FILE__, __func__, __LINE__); \
        }while(0)
#else
#define  PDEBUG() do{}while(0)
#endif

/*debug position*/
#define WMA_VAR_DEBUG
#ifdef WMA_VAR_DEBUG
#define VAR_DEBUG(var) \
        do{\
           WMA_PRINTF("###INFO: Line = %d, varName = %s, varAddr = %p, varVal = 0x%x\r\n",\
                     __LINE__, #var, &var, var); \
        }while(0)
#else
#define  VAR_DEBUG(var) do{}while(0)
#endif

/*ctrl error msg*/
#define  WMA_ERR(error) \
           do {\
           WMA_PRINTF("###ERR: error = %d, at File = %s, Func = %s, Line = %d\r\n",\
                     error, __FILE__, __func__, __LINE__); \
           }while(0)

#endif /*End of _DEBUG_H*/
