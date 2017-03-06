
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define GENERATION 600000
#define NPROB 17 /* dimension of problem grid */
#define BLOCK_SIZE 1*32 /* size of threads should be multiple of warp's size(32) */


void inidat(int , bool** );
void prtdat(int , bool** , char* );
bool** SeqAllocate(int);
void SeqFree(bool**);
bool** cudaSeqAllocate(int );
void cudaSeqFree(bool**);
