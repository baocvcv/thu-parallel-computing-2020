#include <mpi.h>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
/*
 * Modify the "multiply, run" to implement your parallel algorihtm.
 * Compile:
 *      this is a c++ style code
 */
using namespace std;

void serial(int n, double **matrix, double *vector, double **result);
void gen(int n, double ***matrix, double **vector);
void print(int n, double **matrix, double *vector);
void free(int n, double **matrix, double *vector) ;
void run(int n);
double euclidean_dist(int n, double *vec1, double *vec2);

int main(int argc, char* argv[]) { 
	if (argc < 2) {
		cout << "Usage: " << argv[0] << " n" << endl;
		return -1;
	}
	int n = atoi(argv[1]);
	run(n);
}

void serial(int n, double **matrix, double *vector, double **result){
	/*
	 * It is a serial algorithm to 
	 * get the true value of matrix-vector multiplication
	 * please calculate the difference between true value and the value you obtain
	 *
	 */
	(*result) = new double[n];
	for(int i = 0; i < n; i++) {
		(*result)[i] = 0.0;
	}

	for (int i = 0; i < n; i++) {
		double temp = 0.0;
		for (int j = 0; j < n; j++) {
			temp += matrix[i][j] * vector[j];
		}
		(*result)[i] = temp;
	}
}

void gen(int n, double ***matrix, double **vector) {
     /*
      *  generate random matrix and vector,
      *  In order to debug conveniently, you can define a const matrix and vector
      *  but I will check your answer based on random matrix and vector
      */
	(*matrix) = new double*[n];
	srand((unsigned)time(0));
	for(int i = 0; i < n; i++) {
		(*matrix)[i] = new double[n];
		for(int j = 0; j < n; j++) {
			(*matrix)[i][j] = -1 + rand() / (double)RAND_MAX * 2;
		}
	}
	(*vector) = new double[n];
	for(int i = 0; i < n; i++) {
		(*vector)[i] = -1 + rand() * 1.0 / (double)RAND_MAX * 2;
	}
}

void print(int n, double **matrix, double *vector) {
	for(int i = 0; i < n; i++) {
		cout << vector[i] << endl;
	}
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

void free(int n, double **matrix, double *vector) {
	delete[] vector;
	for(int i = 0; i < n; i++)
		delete[] matrix[i];
	delete[] matrix;
}

void run(int n) {
     /*
      * Description: 
      * data partition, communication, calculation based on MPI programming in this function.
      * 
      * 1. call gen() on one process to generate the random matrix and vecotr.
      * 2. distribute the data to other processes.
      * 3. Implement matrix-vector mutiplication
      * 4. calculate the diffenence between product vector and the value of serial(), I'll check this answer.
      */ 

	//TODO : need to take time measurements

	MPI_Comm comm;
	int my_rank, comm_sz;
	double **matrix;
	double *vector;
	double *result_serial, *result_para;
	double *local_col;

	MPI_Init(NULL, NULL);
	comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &my_rank);
	MPI_Comm_size(comm, &comm_sz);

	// allocate mem
	result_serial = new double[n];
	result_para = new double[n];
	local_col = new double[n];

    // 1 gen
	if (my_rank == 0) {
		gen(n, &matrix, &vector);
		// serial time start
		serial(n, matrix, vector, &result_serial);
		// serial time finish
	}
	// distribution_time_start
	if (my_rank == 0) {
		// distribute data

	} else {
		// receive data

	}
	// distribution_time_finish

	// mutiplication
	// parallel time start

	// parallel time finish

	// check correctness
	double diff = euclidean_dist(n, result_serial, result_para);
	cout << "The euclidean distance between serial result "
		 << "and parallel result is: " << diff << endl;
}

double euclidean_dist(int n, double *vec1, double *vec2) {
	double result = .0;
	for (int i = 0; i < n; i++)
		result += (vec1[i]-vec2[i]) * (vec1[i]-vec2[i]);
	return result;
}