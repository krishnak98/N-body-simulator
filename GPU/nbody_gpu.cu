#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include "timer.h"
#include <cuda.h>
#include <assert.h>


#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a,b) (((a) < (b))?(a):(b))
#define SOFTENING 1e-9f

typedef struct {
  float x, y, z;        /* particle positions */
  float vx, vy, vz;     /* particle momenta */
} Particle;


/* randomly initialize particle positions and momenta */
__host__ void ran_init(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


/* calculate all interparticle forces and update instantaneous velocities */
__host__ void calc_force(Particle *p, float dt, int n, int threads) {
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      /* calculate net particle force on i'th particle */
      if (j != i) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
    }
    /* update instantaneous velocity based on force and timestep */
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}
__global__ void calc_force(Particle* p, int nParticles, float dt) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if ( idx >= nParticles ) return;
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    for(int j = 0; j < nParticles; ++j) {
    	float dx = p[j].x - p[idx].x;
	float dy = p[j].y - p[idx].y;
      	float dz = p[j].z - p[idx].z;
      	float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      	float invDist = 1.0f / sqrtf(distSqr);
      	float invDist3 = invDist * invDist * invDist;
      	Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;	
    }
    p[idx].vx += dt * Fx;
    p[idx].vy += dt * Fy;
    p[idx].vz += dt * Fz;  
}

__global__ void update_posn(Particle* p,  int nParticles, float dt) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= nParticles) return;
    p[i].x += p[i].vx*dt;
    p[i].y += p[i].vy*dt;
    p[i].z += p[i].vz*dt; 
}


int main(const int argc, const char** argv) {
  FILE *datafile    = NULL;      /* output file for particle positions */
  int   nParticles  = 300;      /* number of particles */
  int nIters = 200;
  if (argc > 1)
    nParticles      = atoi(argv[1]);

  if (argc > 2) 
    nIters         = atoi(argv[2]);
   int threads_per_block = 256; 
   float t ;
   cudaEvent_t start, stop; 
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   int n_blocks = MIN(nParticles/threads_per_block + (nParticles % threads_per_block != 0) , MAX_BLOCKS_PER_DIM);

  const float dt    = 0.01f; /* time step   */

  float *buf_h        =  (float*)malloc(nParticles*sizeof(Particle));
  Particle  *p_h          = (Particle *) buf_h;
  
  ran_init(buf_h, 6*nParticles); /* Init pos and vel data */

  Particle *p_d;
  float *buf_d;
  cudaMalloc((void**) &buf_d, nParticles * sizeof(Particle));
  p_d = (Particle *) buf_d;
  datafile          = fopen("particles_gpu.csv","w");

  float final_t = 0;

  for (int iter = 1; iter <= nIters; iter++) {
    printf("iteration:%d\n", iter);
    for (int i = 0;i < nParticles; ++i)
      fprintf(datafile, "%d, %f, %f, %f \n",iter, p_h[i].x, p_h[i].y, p_h[i].z);
    cudaEventRecord(start, 0);
    cudaMemcpy(buf_d, buf_h, nParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    calc_force<<<n_blocks, threads_per_block>>>(p_d, nParticles, dt);
    update_posn<<<n_blocks, threads_per_block>>>(p_d, nParticles, dt);
    assert((cudaMemcpy(buf_h, buf_d, nParticles * sizeof(Particle), cudaMemcpyDeviceToHost)) == cudaSuccess);
    cudaEventRecord(stop , 0);
    cudaEventElapsedTime( &t, start, stop);
    final_t += t;
  }
  fclose(datafile);
  printf("avgTime: %f   totTime: %f \n", final_t/ (1000.0 * (nIters - 1)), final_t / 1000.0);
}

