#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
using namespace std;

int muahaha(int sum, int A) {
	return (sum == 3) || (A == 1 && sum == 2);
}

int main(int argc , char* argv[])
{
	ifstream myReadFile;
	myReadFile.open(argv[2]);
	int size = atoi(argv[1]);
	int **A=NULL;
	A=(int**)malloc(size*sizeof(int*));
	for(int i=0;i<size;i++)
	{
		A[i]=(int*)malloc(size*sizeof(int));
	}
	int curr,ii=0,jj=0;
	if (myReadFile.is_open()) 
	{
		while (!myReadFile.eof()) 
		{
			myReadFile >> curr ;
			A[ii][jj++] = curr;
			if(jj==size)
			{
				jj=0;
				ii++;
				if(ii==size)
				{
					myReadFile.close();
					break;
				}
			}

			if(!myReadFile.good())
				break;
		}
	}

	int down,up,left,right;

	/*for(int i = 0; i < size ; i++)
	{
		for(int j = 0; j < size ; j++)
		{
			if(j==size-1)
			cout << A[i][j];
			else
			cout << A[i][j] << " ";
		}
		cout << endl;
	}

	cout << endl;cout << endl;*/
	for(int i = 0; i < size ; i++)
	{
		for(int j = 0; j < size ; j++)
		{
			int sum=0;
			down = (i != size-1) ? i+1 : 0;
			up = (i != 0) ? i-1 : size-1;
			left = (j != 0) ? j-1 : size-1;
			right = (j != size-1) ? j+1 : 0;

			sum = A[down][j] + A[up][j] + A[i][left] + A[i][right] + A[down][left] + A[up][left] + A[down][right] + A[up][right] ;
			if(j==size-1)
			cout << muahaha(sum,A[i][j]);
			else
			cout << muahaha(sum,A[i][j]) << " ";
			
		}

		cout << endl;
	}

	for(int i=0;i<size;i++)
	{
		free(A[i]);
	}
	free(A);
	return 0;
}