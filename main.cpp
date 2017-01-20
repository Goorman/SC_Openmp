#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <math.h>
#include "DPS.h"

using std::string;
using std::cout;
using std::endl;
using std::exception;

int main(int argc, const char ** argv) {
    if (argc != 3){
        cout << "Expected 2 parameters." << endl;
        cout << "Example: ./p nodes_number dirname" << endl;
        cout << "dirname == 'none' omits the printing." << endl;
        return 1;
    }

    int gridsize = atoi(argv[1]);
    string out_dir = argv[2];
    char** repacked_cla = const_cast<char**> (argv);

    MPI_Init(&argc, &repacked_cla);

    try {
        ProcessorMPIParameters pmp(MPI_COMM_WORLD);

        struct timeval tp;
	    gettimeofday(&tp, NULL); 
        long int start_time = tp.tv_sec * 1000 + tp.tv_usec / 1000;
	
	    DPS solver(gridsize);
        solver.ComputeApproximation(pmp);

	    gettimeofday(&tp, NULL);
	    long int end_time = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        double execution_time = end_time - start_time;

        if (out_dir != "none") {
            solver.PrintP(out_dir);
        } 

	if (pmp.rank == 0){
            std::cout << "Execution time: " << execution_time << "ms "<< std::endl;
            std::cout << "Iterations: " << solver.getIterationNumber() << std::endl;
            std::cout << "Approximation net dimensions: " << gridsize << "x" << gridsize << std::endl;
            std::cout << "Number of processors: " << solver.getProcessorNumber() << std::endl;
            std::cout << "Total error: " << solver.getFinalError() << std::endl;
            std::cout << "Average error per node: " << solver.getFinalError() / (gridsize * gridsize) << std::endl;
        }
    }
    catch (exception& e) {
        cout << "ERROR" << endl;
	cout << e.what() << endl;
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();

    return 0;
}

