TARGET = MphImplicit

#CC = pgc++
CC = g++
OBJE = main.o errorfunc.o log.o
#CFLAGS  = -O3 -acc -Minfo=accel -ta=tesla,cc70 -Mcuda 
#CFLAGS  = -O3 -acc -Minfo=accel 
#CFLAGS  = -O3 -fopenmp 
CFLAGS  = -O3
#CFLAGS  = -g -Wall
#LDFLAGS =  -lm -lcublas -lcusparse
LDFLAGS =  -lm


$(TARGET): $(OBJE)
	$(CC) $(CFLAGS) $(WFLAGS) -o $@ $(OBJE) $(LDFLAGS) 

.cpp.o:
	$(CC) $(CFLAGS) $(WFLAGS) -c $< 

clean:
	/bin/rm -f *.o *~ $(TARGET)  $(TARGET).exe 


main.o			:log.h
errorfunc.o		:errorfunc.h
log.o			:errorfunc.h log.h
