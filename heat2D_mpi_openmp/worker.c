#include "header.h"

inline Finalize* worker(int* nbrs, MPI_Comm cartcomm, int subsize, int taskid, int n)
{
	// Allocate A array with sequally
	int** A = SeqAllocate(subsize);

	// Receive from master
	MPI_Request* request = malloc(sizeof(MPI_Request));
	MPI_Irecv(&A[0][0], subsize*subsize, MPI_INT, MASTER, BEGIN, cartcomm, request);

	// Create DataTypes
	MPI_Datatype Send[4];

	int starts[2];
	int subsizes[2];
	int bigsizes[2] = {subsize,subsize};

	starts[0] = 0; starts[1] = 0;
	subsizes[0] = 1; subsizes[1] = subsize;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &Send[UP]);

	starts[0] = subsize-1; starts[1] = 0;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &Send[DOWN]);

	subsizes[0] = subsize; subsizes[1] = 1;
	starts[0] = 0; starts[1] = subsize-1;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &Send[RIGHT]);

	starts[0] = 0; starts[1] = 0;
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &Send[LEFT]);

	// Create Receive Buffers
	int i, z;
	int* Rec[4];
	int DiagSendTable[2][2];
	int **DiagRecvTable;
	int** handler;
	int** B = SeqAllocate(subsize);
	
	DiagRecvTable = malloc(2*sizeof(int*));
	DiagRecvTable[0] = malloc(2*sizeof(int));
	DiagRecvTable[1] = malloc(2*sizeof(int));
	int antegeia,antegeia2;
		for (antegeia=0;antegeia<2;antegeia++)
			for (antegeia2=0;antegeia2<2;antegeia2++)
				DiagRecvTable[antegeia][antegeia2] = -1;

	MPI_Request ReqSend[2][4];
	MPI_Request ReqRecv[4];
	MPI_Request DiagReq[2][2];
	for (i = 0; i < 4; i++)
	{
		MPI_Type_commit(&Send[i]);
		Rec[i] = calloc(subsize, sizeof(int));
		MPI_Send_init(A[0], 1, Send[i], nbrs[i], TAG, cartcomm, &ReqSend[0][i]);
		MPI_Send_init(B[0], 1, Send[i], nbrs[i], TAG, cartcomm, &ReqSend[1][i]);
		MPI_Recv_init(Rec[i], subsize, MPI_INT, nbrs[i], TAG, cartcomm, &ReqRecv[i]);
	}
	
	MPI_Send_init(DiagSendTable[UP], 2, MPI_INT, nbrs[UP], UDIAGS, cartcomm, &DiagReq[SEND][UP]);
	MPI_Recv_init(DiagRecvTable[UP], 2, MPI_INT, nbrs[UP], UDIAGS, cartcomm, &DiagReq[RECV][UP]);

	MPI_Send_init(DiagSendTable[DOWN], 2, MPI_INT, nbrs[DOWN], UDIAGS, cartcomm, &DiagReq[SEND][DOWN]);
	MPI_Recv_init(DiagRecvTable[DOWN], 2, MPI_INT, nbrs[DOWN], UDIAGS, cartcomm, &DiagReq[RECV][DOWN]);

	MPI_Wait(request, MPI_STATUS_IGNORE);
	
	puts("LOOOOOOOOOP...");
	// Calculate its subarray
	int sum;
	for(i = 0; i<GENERATION; i++)
	{	
		//puts("???...");
		//if (taskid==MASTER && i%(GENERATION/100)==0) printf("%3.2f%%\n", 100.0*i/GENERATION);
		z = i%2;
		puts("???...");
		// Send and receive neighbors
		MPI_Startall(4, ReqRecv);
		MPI_Startall(4, ReqSend[z]);
		MPI_Startall(2, DiagReq[RECV]);
		puts("sadsdasdas...");
		// Calculate inside data
		Independent_Update(A, B, subsize);

		// wait to receive Left and Right
		MPI_Waitall(2, &ReqRecv[2], MPI_STATUSES_IGNORE);

		int coords[2] ;
		MPI_Cart_coords( cartcomm,taskid,2,coords); 
		

		DiagSendTable[UP][DIAGRIGHT] = Rec[RIGHT][0];
		DiagSendTable[UP][DIAGLEFT] = Rec[LEFT][0];
		DiagSendTable[DOWN][DIAGRIGHT] = Rec[RIGHT][subsize-1];
		DiagSendTable[DOWN][DIAGLEFT] = Rec[LEFT][subsize-1];




		
		puts("UR\tUL\tDR\tDL");
		printf("%d\t%d\t%d\t%d\n", DiagSendTable[UP][DIAGRIGHT],DiagSendTable[UP][DIAGLEFT],DiagSendTable[DOWN][DIAGRIGHT],DiagSendTable[DOWN][DIAGLEFT]);

		//Send Diagonals 
		MPI_Startall(2, DiagReq[SEND]);

		// wait to receive Up and Down
		MPI_Waitall(2, ReqRecv, MPI_STATUSES_IGNORE); 
		// Calculate neighbors
		Dependent_Update(A, B, subsize, Rec);


		if (taskid == MASTER) {
			int zz;
			puts("RIGHT");
			for (zz = 0; zz < subsize; ++zz)
				printf("%d\n", Rec[RIGHT][zz]);
			puts("LEFT");
			for (zz = 0; zz < subsize; ++zz)
				printf("%d\n", Rec[LEFT][zz]);
			//int zz;
			puts("UP");
			for (zz = 0; zz < subsize; ++zz)
				printf("%d\n", Rec[UP][zz]);
			puts("DOWN");
			for (zz = 0; zz < subsize; ++zz)
				printf("%d\n", Rec[DOWN][zz]);
		}
				
		// Wait to receive diagonals
		MPI_Waitall(2, DiagReq[RECV], MPI_STATUSES_IGNORE); // diag
		
		puts("UR\tUL\tDR\tDL");
		printf("%d\t%d\t%d\t%d\n", DiagRecvTable[UP][DIAGRIGHT],DiagRecvTable[UP][DIAGLEFT],DiagRecvTable[DOWN][DIAGRIGHT],DiagRecvTable[DOWN][DIAGLEFT]);

		UpdateDiag(A ,B, subsize , DiagRecvTable , Rec);
		
		// Reduce
#ifdef __CON__
		if (i%REDUCE == 0)
		{
			sum = diffa(A, B, subsize);
			// compute global residual
			MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_INT, MPI_SUM, cartcomm);
			// solution good enough ?
			if (sum < CONVERGENCE)
			{
				printf("%d %d\n", i, sum);
				break;
			}
		}
#endif

		// set curent result to the new one
		handler = A;
		A = B;
		B = handler;

		// Wait to send neighbors
		MPI_Waitall(4, ReqSend[z], MPI_STATUSES_IGNORE);

		// Wait to send diagonals
		MPI_Waitall(2, DiagReq[SEND], MPI_STATUSES_IGNORE);
	}
	//printf("%d\n", i);

	// Send to master
	MPI_Isend(&A[0][0], subsize*subsize, MPI_INT, MASTER, DONE, cartcomm, request);
	puts("DORAAAAAAAAAAAAAAAAAAAAAAAAAA...");
	for (i = 0; i < 4; i++)
	{
		MPI_Request_free(&ReqRecv[i]);
		MPI_Request_free(&ReqSend[0][i]);
		MPI_Request_free(&ReqSend[1][i]);
		MPI_Type_free(&Send[i]);
		free(Rec[i]);
	}

	//diag free
	puts("KEFTESSSS...");
	MPI_Request_free(&DiagReq[SEND][UP]);
	puts("KEDEFTESSSS...");
	MPI_Request_free(&DiagReq[SEND][DOWN]);
	puts("ketela...");
	MPI_Request_free(&DiagReq[RECV][UP]);
	puts("ketela...");
	MPI_Request_free(&DiagReq[RECV][DOWN]);

	puts("ketela...");
	// Free arrays
	SeqFree(B);

	Finalize* fin = malloc(sizeof(Finalize));
	fin->request = request;
	fin->A = A;
	return fin;
}
