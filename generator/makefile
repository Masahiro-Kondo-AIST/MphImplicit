TARGET = GeneratorForMph

CC = g++
OBJE = errorfunc.o log.o typedefs.o generator.o
#CFLAGS  = -O3 -Wall
#CFLAGS  = -O3 -Wall -Wno-uninitialized
CFLAGS  = -g -Wall
LDFLAGS =  -lm


$(TARGET): $(OBJE)
	$(CC) $(CFLAGS) -o $@ $(OBJE) $(LDFLAGS) 

.cpp.o:
	$(CC) $(CFLAGS) -c $< 

clean:
	/bin/rm -f *.o *~ $(TARGET)  $(TARGET).exe 

errorfunc.o	:errorfunc.h
log.o		:errorfunc.h log.h 
typedefs.o	:typedefs.h
