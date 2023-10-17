#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <omp.h>

#define SOFTENING 1e-9f

typedef struct {
  float x, y, z;        /* particle positions */
  float vx, vy, vz;     /* particle momenta */
} Particle;


/* randomly initialize particle positions and momenta */
void ran_init(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


/* calculate all interparticle forces and update instantaneous velocities */
void calc_force(Particle *p, float dt, int n, int threads) {
#pragma omp parallel shared(p, n) num_threads(threads)
#pragma omp for
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


int main(const int argc, const char** argv) {
  FILE *datafile    = NULL;      /* output file for particle positions */
  int nBodies  = 300;      /* number of particles */
  int threads = 1;
  const float dt    = 0.01f; /* time step   */
  int nIters  = 200;   /* number of steps in simulation */
  if (argc > 1) nBodies = atoi(argv[1]);
  if (argc > 2) nIters = atoi(argv[2]);
  if (argc > 3) threads = atoi(argv[3]);
  
  // omp_set_num_threads(threads);
  // setting it inside calc_force to avoid confusion


 

  float *buf        =  malloc(nBodies*sizeof(Particle));
  Particle  *p          = (Particle *) buf;
  
  ran_init(buf, 6*nBodies); /* Init pos and vel data */

  double totalTime  = 0.0;

  datafile          = fopen("particles_omp.csv","w");
//   fprintf(datafile,"%d %d %d\n", nParticles, nIters, 0);

  /* ------------------------------*/
  /*     MAIN LOOP                 */
  /* ------------------------------*/
  for (int iter = 1; iter <= nIters; iter++) {
    printf("iteration:%d\n", iter);
    
    for (int i = 0;i < nBodies; ++i)
      fprintf(datafile, "%d, %f, %f, %f \n",iter, p[i].x, p[i].y, p[i].z);
    StartTimer();

    calc_force(p, dt, nBodies, threads);           /* compute interparticle forces and update vel */

    for (int i = 0 ; i < nBodies; i++) {  /* compute new position */
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) {                          /* First iter is warm up */
      totalTime += tElapsed;
    }
  }

  fclose(datafile);
  double avgTime = totalTime / (double)(nIters-1);

  printf("avgTime: %f   totTime: %f \n", avgTime, totalTime);
  // free(buf);
}
