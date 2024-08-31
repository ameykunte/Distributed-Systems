#include <mpi.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;

void scatterDimensions(vector<int>& dimensions, int& N, int rank, int size);
int parallelMatrixChainMultiplication(const vector<int>& dimensions, int N, int rank, int size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    vector<int> dimensions;
    
    // Input
    if (rank == 0) {
        cin >> N;
        dimensions.resize(N + 1);
        for (int i = 0; i <= N; ++i) {
            cin >> dimensions[i];
        }
    }

    // Broadcast N to all processes and scatter the dimensions array
    scatterDimensions(dimensions, N, rank, size);

    // Compute the min scalar multiplications in parallel
    int minMultiplications = parallelMatrixChainMultiplication(dimensions, N, rank, size);

    // Output the result
    if (rank == 0) {
        cout << minMultiplications << endl;
    }

    MPI_Finalize();
    return 0;
}

void scatterDimensions(vector<int>& dimensions, int& N, int rank, int size) {
    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the dimensions vector for non-root processes
    if (rank != 0) {
        dimensions.resize(N + 1);
    }

    // Broadcast the dimensions array to all processes
    MPI_Bcast(dimensions.data(), N + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int parallelMatrixChainMultiplication(const vector<int>& dimensions, int N, int rank, int size) {
    // Allocate memory for the cost matrix
    vector<vector<int>> cost(N, vector<int>(N, 0));

    // Fill the cost matrix in parallel
    for (int length = 2; length <= N; ++length) {
        for (int i = rank; i < N - length + 1; i += size) {
            int j = i + length - 1;
            cost[i][j] = numeric_limits<int>::max();

            for (int k = i; k < j; ++k) {
                int q = cost[i][k] + cost[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                if (q < cost[i][j]) {
                    cost[i][j] = q;
                }
            }
        }

        // Sync processes to ensure all rows are completed
        MPI_Barrier(MPI_COMM_WORLD);

        // Share the updated cost matrix with all processes
        for (int i = 0; i < N - length + 1; ++i) {
            MPI_Bcast(&cost[i][i + length - 1], 1, MPI_INT, i % size, MPI_COMM_WORLD);
        }
    }

    // Return the final result from the root process
    return cost[0][N - 1];
}