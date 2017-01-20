#ifndef DPS_h
#define DPS_h

#include <stdio.h>
#include <mpi.h>
#include <string>
#include <math.h>

using std::string;

class DPS_exception: public std::exception
{
private:
    string error;
public:
    DPS_exception(const string& error_): 
        error (error_) {}

    virtual ~DPS_exception() throw() {}

    virtual const char* what() const throw()
    {
        return error.c_str();
    }
};

struct ProcessorMPIParameters {
    int rank;
    int size;
    MPI_Comm comm;

        public:
    ProcessorMPIParameters(MPI_Comm comm_ = MPI_COMM_WORLD){
        comm = comm_;
        MPI_Comm_rank (comm, &rank); 
        MPI_Comm_size (comm, &size);
    }
};

struct ProcessorCellsInfo {
    int x_proc_num;
    int y_proc_num;

    int x_cells_num;
    int x_cell_pos;
    int y_cells_num;
    int y_cell_pos;

    // These parameters are True if processor's cell rectangle touches the border.
    bool top;
    bool bottom;
    bool left;
    bool right;

    ProcessorCellsInfo ();
    ProcessorCellsInfo (const int rank,
                         const int grid_size_x, const int grid_size_y, 
                         const int x_proc_num_, const int y_proc_num_);
};

// DPS - Dirichlet Problem Solver
class DPS {
public:
    DPS (const int gridsize_);
    ~DPS ();
    void ComputeApproximation(const ProcessorMPIParameters& pmp);

    void PrintP(const string& out_dir) const;

    double getFinalError() const;
    int getProcessorNumber() const;
    int getIterationNumber() const;

private:
    ProcessorMPIParameters pmp;
    ProcessorCellsInfo pcinfo;

    double* p;

    int gridsize;
    double eps;
    double q;
    int x0;
    int xn;
    int y0;
    int yn;
    int iterations_counter;
    double final_error;

    MPI_Comm PrepareMPIComm (const ProcessorMPIParameters& pmp, 
                             const int x_proc_num, 
                             const int y_proc_num) const;

    double F(const double x, const double y) const;
    double phi(const double x, const double y) const;
    double x_(const int i) const;   // i from 0 to gridsize
    double y_(const int i) const;   // i from 0 to gridsize
    double hx_(const int i) const;  // i from 0 to gridsize - 1
    double hy_(const int i) const;  // i from 0 to gridsize - 1
    double hhx_(const int i) const; // i from 1 to gridsize - 1
    double hhy_(const int i) const; // i from 1 to gridsize - 1

    double compute_error() const;
    double compute_maxnorm(const double* const f1, const double* const f2) const;
    bool stop_condition(const double* const f1, const double* const f2) const;
    void compute_grid(const ProcessorMPIParameters& pmp, 
                      int& x_proc_num, int& y_proc_num);


    void compute_loa(double* const d_f, const double* const f);
    double compute_sprod(const double* const f1, const double* const f2) const;
    void compute_r(double* const r, const double* const d_p) const;
    void compute_g(double* const g, const double* const r, const double alpha) const;
    void compute_p(const double tau, const double* const g, const double* const p_prev);

    void initialize_border_with_zero(double* const f);
    void initialize_with_border_function(double* const f);

    double* send_message_lr;
    double* send_message_rl;
    double* send_message_td;
    double* send_message_bu;
    double* recv_message_lr;
    double* recv_message_rl;
    double* recv_message_td;
    double* recv_message_bu;
    MPI_Request* recv_loa_reqs;
    MPI_Request* send_loa_reqs;

    enum MPI_MessageTypes {
        LoaLeftRight,
        LoaRightLeft,
        LoaTopDown,
        LoaBottomUp,
        DumpSync
    };
};

#endif /* DPS_h */
