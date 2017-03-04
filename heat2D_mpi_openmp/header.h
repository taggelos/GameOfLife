#ifndef __MPI_HEAT__
#define __MPI_HEAT__

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GENERATION 1
#define REDUCE 10  /* reduce every REDUCE */
#define CONVERGENCE NPROB*NPROB*0.000001*0.000001 /* Break generation if succeed CONVERGENCE */
#define NPROB 425 /* dimension of problem grid */
//#define NPROB 36 /* dimension of problem grid */
#define MAXTHREADS 150 /* maximum number of threads */
#define ERROR 0

#define BEGIN 5 /* message tag */
#define TAG 10 /* message tag */
#define UDIAGS 7 /* message tag */
#define DDIAGS 8 /* message tag */
#define DONE 9 /* message tag */

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
	int cx;
	int cy;
};

typedef struct
{
	int** A;
	MPI_Request* request;
} Finalize;

//extern struct Parms parms;

inline void prtdat(int nx, int** u, char* fnam);
inline void inidat(int nx, int** u);
inline void Independent_Update(int**, int**, int);
inline void Dependent_Update(int**, int**, int, int**);
void UpdateDiag(int** A, int** B, int size, int** DiagRecvTable , int** Row);
inline Finalize* worker(int* nbrs, MPI_Comm cartcomm, int subsize, int taskid, int n);
inline void master(int* nbrs, MPI_Comm cartcomm, int numworkers, int n, int subsize);
inline int** SeqAllocate(int);
inline void SeqFree(int**);
inline void finalize(Finalize* fin);
inline int diffa(int** A, int** B, int size_of_matrix);

#endif
