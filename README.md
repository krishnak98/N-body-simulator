To compile: 
1. Serial : make
2. Parallel: make nbody_omp
3. GPU: load the cuda module, then run make inside the GPU folder

To run: 
Serial: ./nbody 1000 20 : 1000 bodies and 20 iterations
Parallel: ./nbody_omp_parallel 1000 20 8: 1000 bodies and 20 iterations with 8 threads
GPU: ./nbody_gpu 1000 20: 1000 bodies and 20 iterations

I tested correctness using comparing output files (in dat form) for all 3 cases. The initial conditions I used is the same one i used in the advanced programming assignment. 

I then converted output files to a csv format, so i could use the gif created code provided in the slack channel.
(All animations are generated using 1000 bodies and 1000 iterations. )

The speedup graph shows that the best speedup we get is around 4, even when we use 8 or 16 threads. This is due to the cost of synchronization required between the threads. Increasing the number of theads more than the number of cores doesn't make any sense. As it would result in slower execution due to context switching overhead, and some threads will have to wait while others are executing on a core.

For 100,000 bodies, for 20 iterations, the serial version takes 1914.70s 
Hence per iteration, it takes: 95.7s 

for 100000 bodies and 20 iterations using openMP, 
2 threads - 1057.3s
4 threads - 603.5s
8 threads - 452.13s 
16 threads - 465.86s

for 100000 bodies and 20 iterations on the GPU, 
time taken is 1.13s
This is a lot faster than both the serial and parallel versions on the CPU.
Hence an approach using the GPU is preferred to the CPU based ones.

Link for gifs : https://drive.google.com/drive/folders/122gT5IwWmOIzOJKRqGm2uvjmjqghrfoU?usp=sharing

For plotting scripts, just run them using python. You can change the file from particles_omp.csv to nbody.csv as required