#include <mpi.h>
#include <iostream>
#include <complex>
#include <vector>
using namespace std;

int julia(complex<double> z0, complex<double> c, int K, double T) {
    complex<double> z = z0;
    for (int i = 0; i < K; ++i) {
        z = z * z + c;
        if (abs(z) > T) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N, M, K;
    double c_real, c_img;
    int rank, size;
    double T = 2.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        cin >> N >> M >> K;
        cin >> c_real >> c_img;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c_real, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c_img, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    complex<double> c(c_real, c_img);
    int rows_per_proc = N/size;
    int remainder = N%size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    vector<int> local_result((end_row - start_row) * M);
    double real_start = -1.5, real_end = 1.5;
    double img_start = -1.5, img_end = 1.5;
    double real_step = (real_end - real_start) / (M - 1);
    double img_step = (img_end - img_start) / (N - 1);
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < M; ++j) {
            complex<double> z0(real_start + j * real_step, img_start + i * img_step);
            local_result[(i - start_row) * M + j] = julia(z0, c, K, T);
        }
    }

    // gathering at root process
    vector<int> global_result;
    if (rank == 0) {
        global_result.resize(N * M);
    }
    int *recvcounts = nullptr;
    int *displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];

        for (int i = 0; i < size; ++i) {
            int start = i * rows_per_proc + min(i, remainder);
            int end = start + rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = (end - start) * M;
            displs[i] = start * M;
        }
    }

    MPI_Gatherv(local_result.data(), (end_row - start_row) * M, MPI_INT, global_result.data(), recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                cout << global_result[i * M + j] << " ";
            }
            cout << endl;
        }
    }
    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
    }
    MPI_Finalize();
}
