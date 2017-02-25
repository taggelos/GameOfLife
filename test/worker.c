#include "header.h"

inline Finalize* worker(int* nbrs, MPI_Comm cartcomm, int subsize, int taskid)
{
	// Allocate A array with sequally
	float** A = SeqAllocate(subsize);

	// Receive from master
	MPI_Request* request = malloc(sizeof(MPI_Request));
	MPI_Irecv(&A[0][0], subsize*subsize, MPI_FLOAT, MASTER, BEGIN, cartcomm, request);

	// Create DataTypes
	MPI_Datatype Send[4];

	int starts[2];
	int subsizes[2];
	int bigsizes[2] = {subsize,subsize};

	starts[0] = subsize-1; starts[1] = 0;
	subsizes[0] = 1; subsizes[1] = subsize;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &Send[DOWN]);

	starts[0] = 0; starts[1] = 0;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &Send[UP]);

	subsizes[0] = subsize; subsizes[1] = 1;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &Send[LEFT]);

	starts[0] = 0; starts[1] = subsize-1;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &Send[RIGHT]);

	// Create Receive Buffers
	int i, z;
	float* Rec[4];
	float** handler;
	float** B = SeqAllocate(subsize);
	MPI_Request ReqSend[2][4];
	MPI_Request ReqRecv[4];
	for (i = 0; i < 4; i++)
	{
		MPI_Type_commit(&Send[i]);
		Rec[i] = calloc(subsize, sizeof(float));
		MPI_Send_init(&A[0][0], 1, Send[i], nbrs[i], TAG, cartcomm, &ReqSend[0][i]);
		MPI_Send_init(&B[0][0], 1, Send[i], nbrs[i], TAG, cartcomm, &ReqSend[1][i]);
		MPI_Recv_init(Rec[i], subsize, MPI_FLOAT, nbrs[i], TAG, cartcomm, &ReqRecv[i]);
	}

	MPI_Wait(request, MPI_STATUS_IGNORE);

	// Calculate its subarray
	float sum;
	for(i = 0; i<GENERATION; i++)
	{
		if (taskid==MASTER && i%(GENERATION/100)==0) printf("%3.2f%%\n", 100.0*i/GENERATION);
		z = i%2;

		// Send and receive neighbors
		MPI_Startall(4, ReqSend[z]);
		MPI_Startall(4, ReqRecv);

		// Calculate inside data
		Independent_Update(A, B, subsize);

		// Wait to receive neighbors
		MPI_Waitall(4, ReqRecv, MPI_STATUSES_IGNORE);

		// Calculate neighbors
		Dependent_Update(A, B, subsize, Rec);

		// Wait to send neighbors
		MPI_Waitall(4, ReqSend[z], MPI_STATUSES_IGNORE);

		// Reduce
#ifdef __CON__
		if (i%REDUCE == 0)
		{
			sum = diffa(A, B, subsize);
			// compute global residual
			MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, cartcomm);
			// solution good enough ?
			if (sum < CONVERGENCE)
			{
				printf("%d %f\n", i, sum);
				break;
			}
		}
#endif

		// set curent result to the new one
		handler = A;
		A = B;
		B = handler;
	}
	//printf("%d\n", i);

	// Send to master
	MPI_Isend(&A[0][0], subsize*subsize, MPI_FLOAT, MASTER, DONE, cartcomm, request);

	for (i = 0; i < 4; i++)
	{
		MPI_Request_free(&ReqRecv[i]);
		MPI_Request_free(&ReqSend[0][i]);
		MPI_Request_free(&ReqSend[1][i]);
		MPI_Type_free(&Send[i]);
		free(Rec[i]);
	}

	// Free arrays
	SeqFree(B);
	Finalize* fin = malloc(sizeof(Finalize));
	fin->request = request;
	fin->A = A;
	return fin;
}
