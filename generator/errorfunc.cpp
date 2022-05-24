#include <cstdio>
#include <cstdlib>

#define __NOCHANGE__
#include "errorfunc.h"


void* err_malloc(size_t size,const char* file,int line)
{
  void* ptr;
  ptr=malloc(size);
  if(ptr==NULL){
    fprintf(stderr, "Memmory Allocation Error\n");
    fprintf(stderr, "file:%s  line:%d\n",file,line);
    exit(1);
  }
  return ptr;
}

FILE* err_fopen(const char* filename,const char* mode, const char* file,int line)
{
  FILE* fp;
  fp=fopen(filename,mode);
  if(fp==NULL){
    fprintf(stderr, "File Open Error\n");
    fprintf(stderr, "filename:%s  mode:%s\n",filename,mode);
    fprintf(stderr, "file:%s  line:%d\n",file,line);
    exit(1);
  }
  return fp;
}
