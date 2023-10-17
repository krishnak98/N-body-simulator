CC=gcc
CFLAGS=-I.
DEPS = timer.h
OBJ = timer.o nbody.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

nbody: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	\rm -f *.o nbody *~ *#

nbody_omp: 
	gcc-12 -c -fopenmp -o nbody_cpu_parallel.o nbody_cpu_parallel.c -I.
	gcc-12 -fopenmp -o nbody_cpu_parallel timer.o nbody_cpu_parallel.o -I.