
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define GENERATION 600000
#define NPROB 17 /* dimension of problem grid */
#define BLOCK_SIZE 1*32 /* size of threads should be multiple of warp's size(32) */

struct Parms
{
	float cx;
	float cy;
};

extern struct Parms parms;

void inidat(int , float** );
void prtdat(int , float** , char* );
float** SeqAllocate(int);
void SeqFree(float**);
float** cudaSeqAllocate(int );
void cudaSeqFree(float**);
