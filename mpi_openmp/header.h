#ifndef __MPI_GAME_OF_LIFE__
#define __MPI_GAME_OF_LIFE__

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GENERATION 1000

//Dimension of problem grid 
#define NPROB 3600

//Maximum number of threads
#define MAXTHREADS 150  
#define ERROR 0

//Taskid of the main process
#define MASTER 0 

//Neighbors and Diagonals
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3
#define DIAGLEFT 0
#define DIAGRIGHT 1
#define SEND 0
#define RECV 1

//Message tags 
#define BEGIN 5 
#define TAG 6 
#define DIAGS 7 
#define DONE 8 

//Reduce every value of REDUCE
#define REDUCE 10  

//Break generation if succeed CONVERGENCE
#define CONVERGENCE NPROB*NPROB*0.000001*0.000001


typedef struct
{
	char** A;
	MPI_Request* request;
} Finalize;

inline void prtdat(int nx, char** u, char* fnam);
inline void inidat(int nx, char** u);
inline void Independent_Update(char**, char**, int);
inline void Dependent_Update(char**, char**, int, char**);
void UpdateDiag(char** A, char** B, int size, char** DiagRecvTable , char** Row);
inline Finalize* worker(int* neighbors, MPI_Comm cartcomm, int subsize, int taskid, int n);
inline void master(int* neighbors, MPI_Comm cartcomm, int numworkers, int n, int subsize);
inline char** SeqAllocate(int);
inline void SeqFree(char**);
inline void finalize(Finalize* fin);
inline int diffa(char** A, char** B, int size_of_matrix);

#endif
