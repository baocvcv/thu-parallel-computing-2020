#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
/*
 * Modify the "multiply, run" to implement your parallel algorihtm.
 * Compile:
 *      this is a c++ style code
 */
using namespace std;

void serial(int n, double *matrix, double *vector, double **result);
void gen(int n, double **matrix, double **vector);
void print(int n, int m, double *matrix, double *vector,
	const char *msg, int mode);
void run(int n);
double distribute_data(int n, int local_n, double *matrix, double *vector,
    double **local_mat, double **local_vec,
	int comm_sz, int my_rank, MPI_Comm comm);
double euclidean_dist(int n, double *vec1, double *vec2);
void mat_vect_mult(int n, int m, double *mat, double *vect, double *res);
double time_toggle(MPI_Comm comm);

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Usage: " << argv[0] << " n" << endl;
		return -1;
	}
	int n = atoi(argv[1]);
	run(n);
}

void serial(int n, double *matrix, double *vector, double **result){
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
			temp += matrix[i*n + j] * vector[j];
		}
		(*result)[i] = temp;
	}
}

void gen(int n, double **matrix, double **vector) {
     /*
      *  generate random matrix and vector,
      *  In order to debug conveniently, you can define a const matrix and vector
      *  but I will check your answer based on random matrix and vector
      */
	(*matrix) = new double[n*n];
	srand((unsigned)time(0));
	for(int i = 0; i < n * n; i++) {
#ifdef DEBUG
		(*matrix)[i] = i;
#else
		(*matrix)[i] = -1 + rand() / (double)RAND_MAX * 2;
#endif
	}
	(*vector) = new double[n];
	for(int i = 0; i < n; i++) {
#ifdef DEBUG
		(*vector)[i] = 1;
#else
		(*vector)[i] = -1 + rand() * 1.0 / (double)RAND_MAX * 2;
#endif
	}
}

/* if mode == 0, print matrix only; mode == 1, print vect; mode == 2, both */
void print(int n, int m, double *matrix, double *vector,
		const char *msg, int mode) {
	cout << msg << endl;
	if (mode == 1 || mode == 2) {
		cout << "vect: ";
		for(int i = 0; i < m; i++) {
			cout << vector[i] << ' ';
		}
		cout << endl;
	}
	if (mode == 0 || mode == 2) {
		cout << "matrix:" << endl;
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				cout << matrix[i*n + j] << " ";
			}
			cout << endl;
		}
	}
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


	MPI_Comm comm;
	int my_rank, comm_sz;
	MPI_Init(NULL, NULL);
	comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &my_rank);
	MPI_Comm_size(comm, &comm_sz);
	int local_n = n / comm_sz;

    /* generate data and perform serial calculation */
	double *matrix, *vector, *result_serial;
	double serial_time_elapsed;
	if (my_rank == 0) {
		gen(n, &matrix, &vector);
		result_serial = new double[n];
		// serial time start
		serial_time_elapsed = MPI_Wtime();
		serial(n, matrix, vector, &result_serial);
		serial_time_elapsed = MPI_Wtime() - serial_time_elapsed;
		// serial time finish
#ifdef DEBUG
		print(n, n, matrix, vector, "[Input data]", 2);
		print(n, n, NULL, result_serial, "[Serial result]", 1);
#endif
		cout << "** Serial calculation used ";
		cout << serial_time_elapsed << "s" << endl;
	}

	double *local_mat, *local_vec;
	double *local_res = new double[n];
	double para_dist_time = distribute_data(n, local_n, matrix,
        vector, &local_mat, &local_vec, comm_sz, my_rank, comm);

	/* mat vect mutiplication */
	double *result_para = new double[n];
	double para_calc_time;
	// parallel time start
	time_toggle(comm);
	mat_vect_mult(n, local_n, local_mat, local_vec, local_res);
	para_calc_time = time_toggle(comm);
	// parallel time finish

	/* collect results */
	// distribution_time_continue
	time_toggle(comm);
	MPI_Reduce(local_res, result_para, n, MPI_DOUBLE, MPI_SUM, 0, comm);
	para_dist_time += time_toggle(comm);
	// distribution_time_finish

	/* check correctness */
	if (my_rank == 0) {
#ifdef DEBUG
		print(n, 1, NULL, result_para, "[Parallel result]", 1);
#endif
		cout << "** Parallel calculation used ";
		cout << para_dist_time+para_calc_time << "s" << endl;
		cout << "** with distribution time = " << para_dist_time << "s" << endl;
		cout << "** and  calculation  time = " << para_calc_time << "s" << endl;

		double diff = euclidean_dist(n, result_serial, result_para);
		cout << "The euclidean distance between serial result "
			<< "and parallel result is: " << diff << endl;
	}

	/* clean up */
	if (my_rank == 0) {
		delete[] result_para;
		delete[] result_serial;
	}
	delete[] local_mat;
	delete[] local_vec;
	delete[] local_res;
	MPI_Finalize();
}

/* Distributes data */
double distribute_data(int n, int local_n, double *matrix, double *vector,
        double **local_mat, double **local_vec,
		int comm_sz, int my_rank, MPI_Comm comm) {

	/* build data type */
	MPI_Datatype vect_type, mat_block_col;
	MPI_Type_vector(n, local_n, n, MPI_DOUBLE, &vect_type);
	MPI_Type_create_resized(vect_type, 0, local_n * sizeof(double),
		&mat_block_col);
	MPI_Type_commit(&mat_block_col);

	/* distribute data */
	int *sendcts_m = new int[comm_sz];
	int *sendcts_v = new int[comm_sz];
	int *displs_m = new int[comm_sz];
	int *displs_v = new int[comm_sz];
	for (int i = 0; i < comm_sz; i++) {
		sendcts_m[i] = 1;
		displs_m[i] = i;
		sendcts_v[i] = local_n;
		displs_v[i] = i * local_n;
	}

	/* allocate mem */
	*local_mat = new double[local_n * n];
	*local_vec = new double[local_n];
	double para_dist_time;
	// distribution_time_start
	time_toggle(comm);
	if (my_rank == 0) {
		MPI_Scatterv(matrix, sendcts_m, displs_m, mat_block_col, *local_mat,
			n*local_n, MPI_DOUBLE, 0, comm);
		// MPI_Scatter(matrix, 1, mat_block_col, local_mat, n*local_n, MPI_DOUBLE, 0, comm);
		MPI_Scatterv(vector, sendcts_v, displs_v, MPI_DOUBLE, *local_vec,
			local_n, MPI_DOUBLE, 0, comm);
	} else {
		MPI_Scatterv(matrix, sendcts_m, displs_m, mat_block_col, *local_mat,
			n*local_n, MPI_DOUBLE, 0, comm);
		// MPI_Scatter(matrix, 1, mat_block_col, local_mat, n*local_n, MPI_DOUBLE, 0, comm);
		MPI_Scatterv(vector, sendcts_v, displs_v, MPI_DOUBLE, *local_vec,
			local_n, MPI_DOUBLE, 0, comm);
	}
	para_dist_time = time_toggle(comm);

	// distribution_time_finish
	MPI_Type_free(&vect_type);
	MPI_Type_free(&mat_block_col);
	delete[] sendcts_v;
	delete[] displs_v;
	delete[] sendcts_m;
	delete[] displs_m;

    return para_dist_time;
}

double euclidean_dist(int n, double *vec1, double *vec2) {
	double result = .0;
	for (int i = 0; i < n; i++)
		result += (vec1[i]-vec2[i]) * (vec1[i]-vec2[i]);
	return result;
}

void mat_vect_mult(int n, int m, double *mat, double *vect, double *res) {
	for (int i = 0; i < n; i++) {
		res[i] = 0;
		for (int j = 0; j < m; j++)
			res[i] += mat[i*m + j] * vect[j];
	}
}

/*
When first called, record the current time and return 0;
when called again, return the max difference in time.
*/
double time_toggle(MPI_Comm comm) {
	static bool status = true;
	static double local_start;
	double elapsed;
	if (status) {
		MPI_Barrier(comm);
		local_start = MPI_Wtime();
		elapsed = .0;
	} else {
		double local_elapsed = MPI_Wtime() - local_start;
		MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	}
	status = !status;
	return elapsed;
}
