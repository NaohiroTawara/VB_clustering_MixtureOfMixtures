# Makefile

CC = g++
CCFLAGS = -O2 #-fopenmp
#DEBUG = -pg
OPENCVINC = `pkg-config --cflags opencv`
OPENCVLIB = `pkg-config --libs opencv`

SCDIR = $(HOME)/work/spkrCl/libs/tags/1.0.2
SCINC = -I$(SCDIR)
SCLIB = -L$(SCDIR) -lspc

TARGET = spkr_cl_VB 

all: $(TARGET)

## VB
spkrLib:
	make -C $(SCDIR)

spkr_cl_VB: main.cc vbM3trainer.o vbgmmtrainer.o vbgmm.o 
	$(CC) -o $@ $^ $(CCFLAGS) $(OPENCVINC) $(OPENCVLIB) $(SCINC) $(SCLIB) 

vbM3trainer.o: vbM3trainer.cc
	$(CC) -c $^ $(CCFLAGS) $(OPENCVINC)  $(SCINC)
vbgmmtrainer.o: vbgmmtrainer.cc
	$(CC) -c $^ $(CCFLAGS) $(OPENCVINC) $(SCINC)

vbgmm.o: vbgmm.cc
	$(CC) -c $^ $(CCFLAGS) $(OPENCVINC) $(SCINC)

clean:
	rm -rf *.o

