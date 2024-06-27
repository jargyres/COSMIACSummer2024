# Makefile for Linux etc.

.PHONY: all clean
all: main

SHELL=/bin/bash
CC=g++
CFLAGS+=-O3 -Wall -I../bladeRF/host/libraries/libbladeRF/include
LDFLAGS=-lm -lpthread -L../bladeRF/host/build/output -lbladeRF

main: main.o
	${CC} $^ ${LDFLAGS} -o $@

clean:
	rm -f *.o main
