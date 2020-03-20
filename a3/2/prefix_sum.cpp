#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>

using namespace std;

void generate_randnum(int **a, int n, int rank);
void print_vect(int local_a[], int local_n, int n,
    const char title[], int rank, MPI_Comm comm);

int main(void) {
    int my_rank, comm_sz;
    int *local_a, *prefix_sum;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    generate_randnum(&local_a, 10, my_rank);
    print_vect(local_a, 10, 10 * comm_sz, "Before:", my_rank, comm);

    prefix_sum = new int[10];
    prefix_sum[0] = local_a[0];
    // calculate local prefix sum
    for (int i = 1; i < 10; i++) prefix_sum[i] = prefix_sum[i-1] + local_a[i];
    int delta = prefix_sum[9];
    MPI_Scan(MPI_IN_PLACE, &delta, 1, MPI_INT, MPI_SUM, comm);
    delta -= prefix_sum[9];
    // update local prefix sum using delta
    for (int i = 0; i < 10; i++) prefix_sum[i] += delta;
    print_vect(prefix_sum, 10, 10 * comm_sz, "After:", my_rank, comm);

    delete[] local_a;
    delete[] prefix_sum;

    MPI_Finalize();
    return 0;
}

void generate_randnum(int **a, int n, int rank) {
    *a = new int[n];
    srand(time(0) * rank);
    for (int i = 0; i < n; i++)
        (*a)[i] = rand() % 50;
}

void print_vect(
        int local_a[],
        int local_n,
        int n,
        const char title[],
        int rank,
        MPI_Comm comm) {

    int *b = NULL;
    if (rank == 0) {
        b = new int[n];
        MPI_Gather(local_a, local_n, MPI_INT, b, local_n, MPI_INT, 0, comm);
        cout << title << endl;
        for (int i = 0; i < n; i++)
            cout << setw(6) << b[i] << (i%10==9 ? '\n' : ' ');
        cout << endl;
        free(b);
    } else {
        MPI_Gather(local_a, local_n, MPI_INT, b, local_n, MPI_INT, 0, comm);
    }
}