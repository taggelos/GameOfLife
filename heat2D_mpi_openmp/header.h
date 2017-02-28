#ifndef __MPI_HEAT__
#define __MPI_HEAT__

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GENERATION 600000
#define REDUCE 10  /* reduce every REDUCE */
#define CONVERGENCE NPROB*NPROB*0.000001*0.000001 /* Break generation if succeed CONVERGENCE */
#define NPROB 425 /* dimension of problem grid */
#define MAXTHREADS 150 /* maximum number of threads */
#define ERROR 0

#define BEGIN 5 /* message tag */
#define TAG 6 /* message tag */
#define DIAGS 7 /* message tag */
#define DONE 8 /* message tag */

#define MASTER 0 /* taskid of first process */

// Neighbors
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3
#define DIAGLEFT 0
#define DIAGRIGHT 1
#define SEND 0
#define RECV 1


struct Parms
{
	float cx;
	float cy;
};

typedef struct
{
	float** A;
	MPI_Request* request;
} Finalize;

//extern struct Parms parms;

inline void prtdat(int nx, float** u, char* fnam);
inline void inidat(int nx, float** u);
inline void Independent_Update(float**, float**, int);
inline void Dependent_Update(float**, float**, int, float**);
inline Finalize* worker(int* nbrs, MPI_Comm cartcomm, int subsize, int taskid);
inline void master(int* nbrs, MPI_Comm cartcomm, int numworkers, int n, int subsize);
inline float** SeqAllocate(int);
inline void SeqFree(float**);
inline void finalize(Finalize* fin);
inline float diffa(float** A, float** B, int size_of_matrix);

#endif
