#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
using namespace std;

struct Point {
    double x, y;
    bool operator<(const Point& other) const {
        if (x == other.x) return y < other.y;
        return x < other.x;
    }
};

double euclidean_distance(const Point& a, const Point& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N, M, K;
    vector<Point> P, Q;
    if (rank == 0) {
        cin >> N >> M >> K;
        P.resize(N);
        Q.resize(M);
        for (int i = 0; i < N; i++) {
            cin >> P[i].x >> P[i].y;
        }
        for (int j = 0; j < M; ++j) {
            cin >> Q[j].x >> Q[j].y;
        }
    }

    // broadcast all these to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        P.resize(N);
    }
    MPI_Bcast(P.data(), N * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int local_M = M / size;
    int remainder = M % size;
    if (rank < remainder) {
        local_M += 1;
    }

    vector<Point> local_Q(local_M);

    int* sendcounts = new int[size];
    int* displs = new int[size];
    displs[0] = 0;

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (M / size) * 2;
        if (i < remainder) {
            sendcounts[i] += 2;
        }
        if (i > 0) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }

    MPI_Scatterv(Q.data(), sendcounts, displs, MPI_DOUBLE, local_Q.data(), local_M * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    vector<Point> local_results;

    for (int j = 0; j < local_M; ++j) {
        priority_queue<pair<double, Point>> max_heap;
        for (int i = 0; i < N; ++i) {
            double dist = euclidean_distance(P[i], local_Q[j]);
            max_heap.push({dist, P[i]});
            if (max_heap.size() > K) {
                max_heap.pop();
            }
        }
        vector<Point> closest_points;
        while (!max_heap.empty()) {
            closest_points.push_back(max_heap.top().second);
            max_heap.pop();
        }
        reverse(closest_points.begin(), closest_points.end());
        local_results.insert(local_results.end(), closest_points.begin(), closest_points.end());
    }

    int local_size = local_results.size() * 2;
    vector<int> recvcounts(size);
    vector<int> displs_recv(size);

    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs_recv[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs_recv[i] = displs_recv[i -1] + recvcounts[i-1];
        }
    }

    vector<double> global_results;
    if (rank == 0) {
        global_results.resize(displs_recv[size-1] + recvcounts[size - 1]);
    }

    MPI_Gatherv(local_results.data(), local_size, MPI_DOUBLE, global_results.data(), recvcounts.data(), displs_recv.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // printing output in order of queries
    if (rank == 0) {
        for (size_t i = 0; i < global_results.size(); i += 2) {
            cout << global_results[i] << " " << global_results[i + 1] << endl;
        }
    }
    delete[] sendcounts;
    delete[] displs;
    MPI_Finalize();
    return 0;
}
