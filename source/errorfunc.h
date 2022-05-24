#ifndef __ERRORFUNC_H__INCLUDED__
#define __ERRORFUNC_H__INCLUDED__

#include <cstdio>

#ifndef __NOCHANGE__
#define malloc(s) err_malloc((s),__FILE__,__LINE__)
#define fopen(f,c) err_fopen((f),(c),__FILE__,__LINE__)
#endif//__NOCHANGE__

void* err_malloc(size_t size, const char *file, int line);
FILE* err_fopen(const char *filename,const char* mode,const char *file,int line);


#endif// __ERRORFUNC_H__INCLUDED__
