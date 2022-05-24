#include <cstdio>
#include <cstdarg>

#include "log.h"
#include "errorfunc.h"

static FILE* log_fp;

void log_open(const char* filename)
{
  log_fp = fopen(filename,"w");
}

int log_printf(const char* format, ...)
{
    int reval;
    va_list args1,args2;
    va_start(args1,format);
    reval = vfprintf(log_fp,format,args1);
    va_end(args1);
    va_start(args2,format);
    reval = vfprintf(stderr,format,args2);
    va_end(args2);
    return reval;
}

void log_close()
{
  fclose(log_fp);
}

