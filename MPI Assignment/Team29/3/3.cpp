#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n;
    vector<double> a;
    if (rank == 0){
        cin >> n;
        a.resize(n);
        for (int i = 0; i < n; i++){
            cin >> a[i];
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = n/size;
    int remainder = n%size;

    if (rank < remainder){
        local_size += 1;
    }

    vector<double> local_array(local_size);
    vector<int> sendcounts(size);
    vector<int> displacements(size);

    for (int i = 0; i < size; i++){
        if (i < remainder) {
            sendcounts[i] = n /size + 1;
        } else {
            sendcounts[i] = n/size;
        }

        if (i == 0) {
            displacements[i] = 0;
        } else {
            displacements[i] = displacements[i-1] + sendcounts[i-1];
        }
    }

    MPI_Scatterv(rank == 0 ? a.data() : nullptr, sendcounts.data(), displacements.data(), MPI_DOUBLE, local_array.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> local_prefix_sum(local_size);
    local_prefix_sum[0] = local_array[0];
    for (int i = 1; i < local_size; ++i) {
        local_prefix_sum[i] = local_prefix_sum[i-1] + local_array[i];
    }

    double last_sum = local_prefix_sum[local_size -1];
    vector<double> last_elements(size);

    MPI_Gather(&last_sum, 1, MPI_DOUBLE, rank == 0 ? last_elements.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> offsets(size, 0.0);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            offsets[i] = offsets[i-1] + last_elements[i-1];
        }
    }

    MPI_Bcast(offsets.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_size; i++) {
        local_prefix_sum[i] += offsets[rank];
    }

    vector<double> global_prefix_sum;
    if (rank == 0) {
        global_prefix_sum.resize(n);
    }

    MPI_Gatherv(local_prefix_sum.data(), local_size, MPI_DOUBLE,rank == 0 ? global_prefix_sum.data() : nullptr,sendcounts.data(), displacements.data(), MPI_DOUBLE,0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            cout << global_prefix_sum[i] << " ";
        }
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}