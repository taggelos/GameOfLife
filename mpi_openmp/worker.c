#include "header.h"

inline Finalize* worker(int* neighbors, MPI_Comm cartcomm, int subsize, int taskid, int n)
{
	// Allocate A array with sequally

	int** A = SeqAllocate(subsize);

	// Receive from master

	MPI_Request* request = malloc(sizeof(MPI_Request));
	MPI_Irecv(&A[0][0], subsize*subsize, MPI_INT, MASTER, BEGIN, cartcomm, request);

	// Create DataTypes for avoidance of multiple copies

	MPI_Datatype Send[4];

	int start[2];
	int subsizes[2];
	int gridsizes[2] = {subsize,subsize};

	// Set start and end for each subarray

	start[0] = 0; start[1] = 0;
	subsizes[0] = 1; subsizes[1] = subsize;
	MPI_Type_create_subarray(2, gridsizes, subsizes, start, MPI_ORDER_C, MPI_INT, &Send[UP]);

	start[0] = subsize-1; 
	MPI_Type_create_subarray(2, gridsizes, subsizes, start, MPI_ORDER_C, MPI_INT, &Send[DOWN]);

	subsizes[0] = subsize; subsizes[1] = 1;
	start[0] = 0; start[1] = subsize-1;
	MPI_Type_create_subarray(2, gridsizes, subsizes, start, MPI_ORDER_C, MPI_INT, &Send[RIGHT]);

	start[1] = 0;
	MPI_Type_create_subarray(2, gridsizes, subsizes, start, MPI_ORDER_C, MPI_INT, &Send[LEFT]);

	// Create Receive and Send Buffers 

	int i, z;
	int* Recv[4];
	int DiagSendTable[2][2];
	int **DiagRecvTable;
	int** handler;
	int** B = SeqAllocate(subsize);
	
	DiagRecvTable = malloc(2*sizeof(int*));
	DiagRecvTable[0] = malloc(2*sizeof(int));
	DiagRecvTable[1] = malloc(2*sizeof(int));
	
	for ( i=0;i<2;i++)
		for (z=0;z<2;z++)
			DiagRecvTable[i][z] = -1;

	MPI_Request ReqSend[2][4];
	MPI_Request ReqRecv[4];
	MPI_Request DiagReq[2][2];
	for (i = 0; i < 4; i++)
	{
		MPI_Type_commit(&Send[i]);
		Recv[i] = calloc(subsize, sizeof(int));
		MPI_Send_init(A[0], 1, Send[i], neighbors[i], TAG, cartcomm, &ReqSend[0][i]);
		MPI_Send_init(B[0], 1, Send[i], neighbors[i], TAG, cartcomm, &ReqSend[1][i]);
		MPI_Recv_init(Recv[i], subsize, MPI_INT, neighbors[i], TAG, cartcomm, &ReqRecv[i]);
	}
	
	MPI_Send_init(DiagSendTable[UP], 2, MPI_INT, neighbors[UP], DIAGS, cartcomm, &DiagReq[SEND][UP]);
	MPI_Recv_init(DiagRecvTable[UP], 2, MPI_INT, neighbors[UP], DIAGS, cartcomm, &DiagReq[RECV][UP]);

	MPI_Send_init(DiagSendTable[DOWN], 2, MPI_INT, neighbors[DOWN], DIAGS, cartcomm, &DiagReq[SEND][DOWN]);
	MPI_Recv_init(DiagRecvTable[DOWN], 2, MPI_INT, neighbors[DOWN], DIAGS, cartcomm, &DiagReq[RECV][DOWN]);

	MPI_Wait(request, MPI_STATUS_IGNORE);
	
	
	// Calculate each subarray

	int sum;
	for(i = 0; i<GENERATION; i++)
	{	
		
		z = i%2;
		
		// Send and receive neighbors

		MPI_Startall(4, ReqRecv);
		MPI_Startall(4, ReqSend[z]);
		MPI_Startall(2, DiagReq[RECV]);
		
		// Calculate inside data

		Independent_Update(A, B, subsize);

		// Wait to receive Left and Right

		MPI_Waitall(2, &ReqRecv[2], MPI_STATUSES_IGNORE);

		int coords[2] ;
		MPI_Cart_coords( cartcomm,taskid,2,coords); 
		

		DiagSendTable[UP][DIAGRIGHT] = Recv[RIGHT][0];
		DiagSendTable[UP][DIAGLEFT] = Recv[LEFT][0];
		DiagSendTable[DOWN][DIAGRIGHT] = Recv[RIGHT][subsize-1];
		DiagSendTable[DOWN][DIAGLEFT] = Recv[LEFT][subsize-1];

		// Send Diagonals

		MPI_Startall(2, DiagReq[SEND]);

		// Wait to receive Up and Down

		MPI_Waitall(2, ReqRecv, MPI_STATUSES_IGNORE); 
		
		// Calculate neighbors

		Dependent_Update(A, B, subsize, Recv);
	
		// Wait to receive diagonals

		MPI_Waitall(2, DiagReq[RECV], MPI_STATUSES_IGNORE); // diag
		
		// Update with right values the diagonals

		UpdateDiag(A ,B, subsize , DiagRecvTable , Recv);
		
		// Reduce

#ifdef __CON__
		if (i%REDUCE == 0)
		{
			sum = diffa(A, B, subsize);

			// Compute global residual

			MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_INT, MPI_SUM, cartcomm);

			if (sum < CONVERGENCE)
			{
				printf("%d %d\n", i, sum);
				break;
			}
		}
#endif

		// Set curent result to the new one

		handler = A;
		A = B;
		B = handler;

		// Wait to send neighbors

		MPI_Waitall(4, ReqSend[z], MPI_STATUSES_IGNORE);

		// Wait to send diagonals

		MPI_Waitall(2, DiagReq[SEND], MPI_STATUSES_IGNORE);
	}

	// Send to master

	MPI_Isend(&A[0][0], subsize*subsize, MPI_INT, MASTER, DONE, cartcomm, request);

	for (i = 0; i < 4; i++)
	{
		MPI_Request_free(&ReqRecv[i]);
		MPI_Request_free(&ReqSend[0][i]);
		MPI_Request_free(&ReqSend[1][i]);
		MPI_Type_free(&Send[i]);
		free(Recv[i]);
	}

	//Diag free
	
	MPI_Request_free(&DiagReq[SEND][UP]);
	
	MPI_Request_free(&DiagReq[SEND][DOWN]);
	
	MPI_Request_free(&DiagReq[RECV][UP]);
	
	MPI_Request_free(&DiagReq[RECV][DOWN]);

	// Free arrays

	SeqFree(B);

	Finalize* fin = malloc(sizeof(Finalize));
	fin->request = request;
	fin->A = A;

	return fin;
}
