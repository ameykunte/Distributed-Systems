#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

void scatterMatrix(vector<vector<double>>& local_matrix, int N, int rank, int size);
void gatherMatrix(vector<vector<double>>& local_matrix, vector<vector<double>>& matrix, int N, int rank, int size);
void gaussianElimination(vector<vector<double>>& local_matrix, int N, int rank, int size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    vector<vector<double>> matrix;
    //Input
    if (rank == 0) {
        cin >> N;
        matrix.resize(N, vector<double>(N));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cin >> matrix[i][j];
            }
        }
    }

    // Broadcast the matrix size to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate rows per process and the start and end indices
    int rows_per_process = N / size;
    int extra_rows = N % size;

    int local_rows = rows_per_process + (rank < extra_rows ? 1 : 0);
    int start_row = rank * rows_per_process + min(rank, extra_rows);
    int end_row = start_row + local_rows;

    // Resize the local matrix
    vector<vector<double>> local_matrix(local_rows, vector<double>(2 * N));

    // Scatter the rows of the matrix to all processes
    scatterMatrix(local_matrix, N, rank, size);

    // Perform Gaussian elimination on the local matrix
    gaussianElimination(local_matrix, N, rank, size);

    // Gather the rows of the inverse matrix from all processes
    gatherMatrix(local_matrix, matrix, N, rank, size);

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << fixed << setprecision(2) << matrix[i][j] << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

void scatterMatrix(vector<vector<double>>& local_matrix, int N, int rank, int size) {
    vector<int> sendcounts(size);
    vector<int> displs(size);

    int rows_per_proc = N / size;
    int remainder = N % size;

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N * 2;
        displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
    }

    vector<double> sendbuf;
    if (rank == 0) {
        sendbuf.resize(N * N * 2);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                sendbuf[i * N * 2 + j] = local_matrix[i][j];
                sendbuf[i * N * 2 + N + j] = (i == j) ? 1.0 : 0.0;  // Augment with identity matrix
            }
        }
    }

    vector<double> recvbuf(sendcounts[rank]);
    MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, recvbuf.data(), recvbuf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_rows = recvbuf.size() / (2 * N);
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            local_matrix[i][j] = recvbuf[i * 2 * N + j];
        }
    }
}

void gatherMatrix(vector<vector<double>>& local_matrix, vector<vector<double>>& matrix, int N, int rank, int size) {
    vector<int> recvcounts(size);
    vector<int> displs(size);

    int rows_per_proc = N / size;
    int remainder = N % size;

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
        displs[i] = (i == 0 ? 0 : displs[i - 1] + recvcounts[i - 1]);
    }

    vector<double> sendbuf(local_matrix.size() * N);
    vector<double> recvbuf(N * N);

    for (int i = 0; i < local_matrix.size(); ++i) {
        for (int j = 0; j < N; ++j) {
            sendbuf[i * N + j] = local_matrix[i][N + j];  // Extract inverse matrix part
        }
    }

    MPI_Gatherv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                matrix[i][j] = recvbuf[i * N + j];
            }
        }
    }
}

void gaussianElimination(vector<vector<double>>& local_matrix, int N, int rank, int size) {
    int local_rows = local_matrix.size();
    vector<double> pivot_row(2 * N);

    for (int k = 0; k < N; ++k) {
        int owner = k / (N / size + (k % size < N % size ? 1 : 0));  // Determine which process owns the current row

        if (rank == owner) {
            int local_row = k - (owner * (N / size) + min(owner, N % size));
            double pivot = local_matrix[local_row][k];
            for (int j = 0; j < 2 * N; ++j) {
                local_matrix[local_row][j] /= pivot;
            }
            pivot_row = local_matrix[local_row];
        }

        MPI_Bcast(pivot_row.data(), 2 * N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows; ++i) {
            if (local_matrix[i][k] != pivot_row[k]) {
                double factor = local_matrix[i][k];
                for (int j = 0; j < 2 * N; ++j) {
                    local_matrix[i][j] -= factor * pivot_row[j];
                }
            }
        }
    }
}
