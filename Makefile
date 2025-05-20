CC=clang
CFLAGS=-Iopen-gpu-kernel-modules/src/nvidia/arch/nvalloc/unix/include \
       -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc

all: sniffer saxpy

pprint.h: pprint.py stub.c
	./$<

sniffer.o: sniffer.c pprint.h
	$(CC) -c $(CFLAGS) $< -o $@

sniffer: sniffer.o
	$(CC) -o $@ $^

saxpy: saxpy.cu
	nvcc -o $@ $^ -arch=sm_86 -lcuda -g

clean:
	rm -f sniffer sniffer.o saxpy pprint.h

.PHONY: all clean
