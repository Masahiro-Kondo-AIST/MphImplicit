#ifndef __LOG_H__INCLUDED__
#define __LOG_H__INCLUDED__

#include <stdio.h>


void log_open(const char* filename);
int  log_printf(const char* format, ...);
void log_close();

#endif// __LOG_H__INCLUDED__
