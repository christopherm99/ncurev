CC=clang
CFLAGS=-Iopen-gpu-kernel-modules/kernel-open/common/inc

all: sniffer saxpy

sniffer.o: sniffer.c
	$(CC) -c $(CFLAGS) $^ -o $@

sniffer: sniffer.o
	$(CC) -o $@ $^

saxpy: saxpy.cu
	nvcc -o $@ $^ -arch=sm_86 -lcuda -g

clean:
	rm -f sniffer sniffer.o saxpy

.PHONY: all clean
