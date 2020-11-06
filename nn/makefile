GCC=gcc

CFLAGS=-g

OFLAGS=-O3 -finline-functions -funroll-loops -march=native -mtune=native -flto

LFLAGS=-lm

all: nn

nn: nn.c
	$(GCC) $(CFLAGS) $(OFLAGS) $< -o $@ $(LFLAGS)

clean:
	rm -Rf *~ nn


