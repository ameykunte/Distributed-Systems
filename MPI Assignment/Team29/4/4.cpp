#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

void scatterMatrix(const vector<vector<double>>& matrix, vector<vector<double>>& local_matrix, int N, int rank, int size);
void gatherMatrix(vector<vector<double>>& local_matrix, vector<vector<double>>& matrix, int N, int rank, int size);
void gaussianElimination(vector<vector<double>>& local_matrix, int N, int rank, int size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    vector<vector<double>> matrix;
    vector<vector<double>> local_matrix;

    // Get input 
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

    // Resize the local matrix
    local_matrix.resize(N / size, vector<double>(N * 2));

    // Scatter the rows of the matrix to all processes
    scatterMatrix(matrix, local_matrix, N, rank, size);

    // Perform Gaussian elimination on the local matrix
    gaussianElimination(local_matrix, N, rank, size);

    // Gather the rows of the inverse matrix from all processes
    gatherMatrix(local_matrix, matrix, N, rank, size);

    if (rank == 0) {
        // Output the inverse matrix
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

void scatterMatrix(const vector<vector<double>>& matrix, vector<vector<double>>& local_matrix, int N, int rank, int size) {
    vector<double> send_buffer(N * N * 2);
    vector<double> recv_buffer((N * N * 2) / size);

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                send_buffer[i * N * 2 + j] = matrix[i][j];
                send_buffer[i * N * 2 + N + j] = (i == j) ? 1.0 : 0.0;  // Augment with identity matrix
            }
        }
    }

    MPI_Scatter(send_buffer.data(), (N * N * 2) / size, MPI_DOUBLE, recv_buffer.data(), (N * N * 2) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N / size; ++i) {
        for (int j = 0; j < N * 2; ++j) {
            local_matrix[i][j] = recv_buffer[i * N * 2 + j];
        }
    }
}

void gatherMatrix(vector<vector<double>>& local_matrix, vector<vector<double>>& matrix, int N, int rank, int size) {
    vector<double> send_buffer((N * N * 2) / size);
    vector<double> recv_buffer(N * N * 2);

    for (int i = 0; i < N / size; ++i) {
        for (int j = 0; j < N * 2; ++j) {
            send_buffer[i * N * 2 + j] = local_matrix[i][j];
        }
    }

    MPI_Gather(send_buffer.data(), (N * N * 2) / size, MPI_DOUBLE, recv_buffer.data(), (N * N * 2) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                matrix[i][j] = recv_buffer[i * N * 2 + N + j];  // Extract inverse matrix from augmented matrix
            }
        }
    }
}

void gaussianElimination(vector<vector<double>>& local_matrix, int N, int rank, int size) {
    for (int k = 0; k < N; ++k) {
        int owner = k / (N / size);  // Determine which process owns the current row

        if (rank == owner) {
            int local_row = k % (N / size);

            // Normalize the pivot row
            double pivot = local_matrix[local_row][k];
            for (int j = 0; j < N * 2; ++j) {
                local_matrix[local_row][j] /= pivot;
            }

            // Broadcast the normalized pivot row to all processes
            MPI_Bcast(local_matrix[local_row].data(), N * 2, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        } else {
            vector<double> pivot_row(N * 2);
            MPI_Bcast(pivot_row.data(), N * 2, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            if (rank == owner) {
                local_matrix[k % (N / size)] = pivot_row;
            } else {
                // Update the other rows
                for (int i = 0; i < N / size; ++i) {
                    if (k % size != i) {
                        double factor = local_matrix[i][k];
                        for (int j = 0; j < N * 2; ++j) {
                            local_matrix[i][j] -= factor * pivot_row[j];
                        }
                    }
                }
            }
        }
    }
}