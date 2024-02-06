TARGET          = MphImplicit
TARGET_3D       = MphImplicit_3d
TARGET_GCC      = Mph_gcc
TARGET_GCC3D    = Mph_gcc3d
TARGET_ICC      = Mph_icc
TARGET_ICC3D    = Mph_icc3d
TARGET_ACC      = Mph_acc
TARGET_ACC3D    = Mph_acc3d
TARGET_CUDA     = Mph_cuda
TARGET_CUDA3D   = Mph_cuda3d

SRC = main.cpp errorfunc.cpp log.cpp

DFLAGS = -D TWO_DIMENSIONAL


###### For single CPU calculation compiled with g++ ######
CC = g++
OBJE = $(patsubst %.cpp,%.o,$(SRC))
OBJE_3D = $(patsubst %.cpp,%_3d.o,$(SRC))

CFLAGS  = -O3
LDFLAGS =  -lm

$(TARGET): $(OBJE)
	$(CC) $(CFLAGS) $(DFLAGS) -o $@ $(OBJE) $(LDFLAGS) 

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) -c $< 

$(TARGET_3D): $(OBJE_3D)
	$(CC) $(CFLAGS)  -o $@ $(OBJE_3D) $(LDFLAGS) 

%_3d.o : %.cpp
	$(CC) $(CFLAGS)  -o $@ -c $<



###### For OpenMP calculation compiled with g++ ######
GCC = g++
OBJE_GCC   = $(patsubst %.cpp,%_gcc.o,$(SRC))
OBJE_GCC3D = $(patsubst %.cpp,%_gcc3d.o,$(SRC))
CFLAGS_GCC = -O3 -fopenmp

$(TARGET_GCC): $(OBJE_GCC)
	$(GCC) $(CFLAGS_GCC) $(DFLAGS) -o $@ $(OBJE_GCC) $(LDFLAGS) 

%_gcc.o : %.cpp
	$(GCC) $(CFLAGS_GCC) $(DFLAGS) -o $@ -c $<

$(TARGET_GCC3D): $(OBJE_GCC3D)
	$(GCC) $(CFLAGS_GCC)  -o $@ $(OBJE_GCC3D) $(LDFLAGS) 

%_gcc3d.o : %.cpp
	$(GCC) $(CFLAGS_GCC)  -o $@ -c $<

	
	
###### For OpenMP calculation compiled with icpx ######
ICC = icpx
OBJE_ICC   = $(patsubst %.cpp,%_icc.o,$(SRC))
OBJE_ICC3D = $(patsubst %.cpp,%_icc3d.o,$(SRC))
CFLAGS_ICC = -O3 -fiopenmp

$(TARGET_ICC): $(OBJE_ICC)
	$(ICC) $(CFLAGS_ICC) $(DFLAGS) -o $@ $(OBJE_ICC) $(LDFLAGS) 

%_icc.o : %.cpp
	$(ICC) $(CFLAGS_ICC) $(DFLAGS) -o $@ -c $<

$(TARGET_ICC3D): $(OBJE_ICC3D)
	$(ICC) $(CFLAGS_ICC)  -o $@ $(OBJE_ICC3D) $(LDFLAGS) 

%_icc3d.o : %.cpp
	$(ICC) $(ICC_CFLAGS)  -o $@ -c $<


	
###### For OpenACC calculation compiled with nvc++(pgc++) ######
PGCC = pgc++
OBJE_ACC   = $(patsubst %.cpp,%_acc.o,$(SRC))
OBJE_ACC3D = $(patsubst %.cpp,%_acc3d.o,$(SRC))
CFLAGS_ACC = -O3 -acc -Minfo=accel

$(TARGET_ACC): $(OBJE_ACC)
	$(PGCC) $(CFLAGS_ACC) $(DFLAGS) -o $@ $(OBJE_ACC) $(LDFLAGS) 

%_acc.o : %.cpp
	$(PGCC) $(CFLAGS_ACC) $(DFLAGS) -o $@ -c $<

$(TARGET_ACC3D): $(OBJE_ACC3D)
	$(PGCC) $(CFLAGS_ACC)  -o $@ $(OBJE_ACC3D) $(LDFLAGS) 

%_acc3d.o : %.cpp
	$(PGCC) $(CFLAGS_ACC)  -o $@ -c $<


###### For OpenACC+CUDA calculation compiled with nvc++ ######
PGCC = pgc++
PGCC = nvc++
OBJE_CUDA    =  $(patsubst %.cpp,%_cuda.o,$(SRC))
OBJE_CUDA3D  =  $(patsubst %.cpp,%_cuda3d.o,$(SRC))
CFLAGS_CUDA  = -O3 -acc -Minfo=accel -ta=tesla,cc70,cc80 -Mcuda 
#CFLAGS_CUDA = -O3 -acc -Minfo=accel -gpu=ccall -Mcuda
LDFLAGS_CUDA = -lm -lcublas -lcusparse

$(TARGET_CUDA): $(OBJE_CUDA)
	$(PGCC) $(CFLAGS_CUDA) $(DFLAGS) -o $@ $(OBJE_CUDA) $(LDFLAGS_CUDA) 

%_cuda.o : %.cpp
	$(PGCC) $(CFLAGS_CUDA) $(DFLAGS) -o $@ -c $<
	
$(TARGET_CUDA3D): $(OBJE_CUDA3D)
	$(PGCC) $(CFLAGS_CUDA)  -o $@ $(OBJE_CUDA3D) $(LDFLAGS_CUDA) 

%_cuda3d.o : %.cpp
	$(PGCC) $(CFLAGS_CUDA)  -o $@ -c $<
	



clean:
	/bin/rm -f *.o *~ $(TARGET)  $(TARGET).exe 


main.o			:log.h
errorfunc.o		:errorfunc.h
log.o			:errorfunc.h log.h
