#include "DPS.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

ProcessorCellsInfo::ProcessorCellsInfo ():
x_proc_num (0),
y_proc_num (0),
x_cells_num (0),
x_cell_pos (0),
y_cells_num (0),
y_cell_pos (0),
top (false),
bottom (false),
left (false),
right (false)
{}

ProcessorCellsInfo::ProcessorCellsInfo (const int rank, 
										  const int grid_size_x, const int grid_size_y, 
										  const int x_proc_num_, const int y_proc_num_){
    x_proc_num = x_proc_num_;
    y_proc_num = y_proc_num_;

    int x_cells_per_proc = (grid_size_x + 1) / x_proc_num;
    int x_extra_cells_num = (grid_size_x + 1) % x_proc_num;
    int x_normal_tasks_num = x_proc_num - x_extra_cells_num;

    if (rank % x_proc_num < x_normal_tasks_num) {
        x_cells_num = x_cells_per_proc;
        x_cell_pos = rank % x_proc_num * x_cells_per_proc;
    } else {
        x_cells_num = x_cells_per_proc + 1;
        x_cell_pos = rank % x_proc_num * x_cells_per_proc + (rank % x_proc_num - x_normal_tasks_num);
    }

    int y_cells_per_proc = (grid_size_y + 1) / y_proc_num;
    int y_extra_cells_num = (grid_size_y + 1) % y_proc_num;
    int y_normal_tasks_num = y_proc_num - y_extra_cells_num;

    if (rank / x_proc_num < y_normal_tasks_num) {
        y_cells_num = y_cells_per_proc;
        y_cell_pos = rank / x_proc_num * y_cells_per_proc;
    } else {
        y_cells_num = y_cells_per_proc + 1;
        y_cell_pos = rank / x_proc_num * y_cells_per_proc + (rank / x_proc_num - y_normal_tasks_num);
    }

    top = (rank < x_proc_num);
    bottom = (rank >= x_proc_num * (y_proc_num - 1));
    left = (rank % x_proc_num == 0);
    right = (rank % x_proc_num == x_proc_num - 1);
}

// --------------------------------------------------------------------------------

DPS::DPS(const int gridsize_): 
gridsize(gridsize_),
eps(0.0001),
q(1.5),
x0(0),
xn(2),
y0(0),
yn(2),
iterations_counter(0),
final_error(0),

p (NULL),

send_message_lr (NULL),
send_message_rl (NULL),
send_message_td (NULL),
send_message_bu (NULL),
recv_message_lr (NULL),
recv_message_rl (NULL),
recv_message_td (NULL),
recv_message_bu (NULL),
recv_loa_reqs (NULL),
send_loa_reqs (NULL)
{
	send_loa_reqs = new MPI_Request [4];
    recv_loa_reqs = new MPI_Request [4];
}

DPS::~DPS(){
    if (p != NULL){
        delete [] p;
    }

    if (send_message_lr != NULL){
        delete [] send_message_lr;
    }
    if (send_message_rl != NULL){
        delete [] send_message_rl;
    }
    if (send_message_td != NULL){
        delete [] send_message_td;
    }
    if (send_message_bu != NULL){
        delete [] send_message_bu;
    }
    if (recv_message_lr != NULL){
        delete [] recv_message_lr;
    }
    if (recv_message_rl != NULL){
        delete [] recv_message_rl;
    }
    if (recv_message_td != NULL){
        delete [] recv_message_td;
    }
    if (recv_message_bu != NULL){
        delete [] recv_message_bu;
    }
    if (recv_loa_reqs != NULL){
        delete [] recv_loa_reqs;
    }
    if (send_loa_reqs != NULL){
        delete [] send_loa_reqs;
    }

    if (pmp.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&pmp.comm);
    }
}

void DPS::compute_grid(const ProcessorMPIParameters& pmp_, int& x_proc_num, int& y_proc_num){
    x_proc_num = int(ceil(sqrt(pmp_.size)));
    while(pmp_.size % x_proc_num != 0) {
        x_proc_num--;
    }
    y_proc_num = pmp_.size / x_proc_num;
}

double DPS::getFinalError() const {
    return final_error;
}

int DPS::getProcessorNumber() const {
    return pmp.size;
}

int DPS::getIterationNumber() const {
    return iterations_counter;
}

MPI_Comm DPS::PrepareMPIComm(const ProcessorMPIParameters& pmp, 
							 const int x_proc_num, const int y_proc_num) const{
    MPI_Comm rank_comm;
    if (pmp.rank < x_proc_num * y_proc_num){
        MPI_Comm_split(MPI_COMM_WORLD, 1, pmp.rank, &rank_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, pmp.rank, &rank_comm);
    }

    return rank_comm;
}

void DPS::ComputeApproximation(const ProcessorMPIParameters& pmp_) {
    int x_proc_num = 0;
    int y_proc_num = 0;
    compute_grid(pmp_, x_proc_num, y_proc_num);

    MPI_Comm algComm = PrepareMPIComm(pmp_, x_proc_num, y_proc_num);
    if (algComm == MPI_COMM_NULL)
        return;

    if (pmp.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&pmp.comm);
    }
    pmp = ProcessorMPIParameters(algComm);
    pcinfo = ProcessorCellsInfo(pmp.rank, gridsize, gridsize, x_proc_num, y_proc_num);

    if (p != NULL) delete [] p;

    p = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* p_prev = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* g = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* r = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* dp = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* dg = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* dr = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];

    double sprod_dg_and_g = 1;
    double sprod_dr_and_g = 1;
    double sprod_r_and_g = 1;
    double alpha = 0;
    double tau = 0;

    initialize_border_with_zero(g);
    initialize_border_with_zero(r);
    initialize_border_with_zero(dp);
    initialize_border_with_zero(dg);
    initialize_border_with_zero(dr);
    initialize_with_border_function(p);
    initialize_with_border_function(p_prev);

    iterations_counter = 0;
	// ALGORITHM ITERATION 1
    {
    	compute_loa(dp, p_prev);
    	compute_r(r, dp);
    	std::swap(g, r);
    	compute_loa(dg, g);
    	sprod_r_and_g = compute_sprod(g, g);
    	sprod_dg_and_g = compute_sprod(dg, g);
    	if (sprod_dg_and_g != 0){
            tau = sprod_r_and_g / sprod_dg_and_g;
        } else {
            throw DPS_exception( "Division by 0 in tau computation, iteration 1.");
        }
        compute_p (tau, g, p_prev);
        std::swap(p, p_prev);
        iterations_counter++;
    }
	// ALGORITHM ITERATION 2+
    while(true){
        compute_loa(dp, p_prev);
        compute_r (r, dp);
        compute_loa(dr, r);
        sprod_dr_and_g = compute_sprod(dr, g);
        alpha = sprod_dr_and_g / sprod_dg_and_g;
        compute_g (g, r, alpha);
        compute_loa(dg, g);
        sprod_r_and_g = compute_sprod(r, g);
        sprod_dg_and_g = compute_sprod(dg, g);
        if (sprod_dg_and_g != 0){
            tau = sprod_r_and_g / sprod_dg_and_g;
        } else {
            throw DPS_exception( "Division by 0 in tau computation.");
        }
        compute_p (tau, g, p_prev);

        if (stop_condition (p, p_prev))
            break;

        std::swap(p, p_prev);
        iterations_counter++;
    }

    final_error = compute_error();

    delete [] dp;
    delete [] dg;
    delete [] dr;
    delete [] g;
    delete [] r;
}

double DPS::F(const double x, const double y) const {
    return (x * x + y * y) / ((1 + x * y) * (1 + x * y));
}

double DPS::phi(const double x, const double y) const {
    return log(1 + x * y);
}

double DPS::x_(const int i) const {
	double c = (pow(1 + i * 1.0 / gridsize, q) - 1) / (pow(2, q) - 1);
	return xn * c + x0 * (1 - c);
}

double DPS::y_(const int i) const {
	double c = (pow(1 + i * 1.0 / gridsize, q) - 1) / (pow(2, q) - 1);
	return yn * c + y0 * (1 - c);
}

double DPS::hx_(const int i) const {
    return x_(i+1) - x_(i);
}

double DPS::hy_(const int i) const {
    return y_(i+1) - y_(i);
}

double DPS::hhx_(const int i) const {
    return 0.5*(hx_(i) + hx_(i-1));
}

double DPS::hhy_(const int i) const {
    return 0.5*(hy_(i) + hy_(i-1));
}

double DPS::compute_error() const {
    double local_error = 0;

    #pragma omp parallel for schedule(static) reduction(+:local_error)
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            local_error += fabs(phi(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + j)) - p[j * pcinfo.x_cells_num + i]);
        }
    }

    double global_error = 0;

    int ret = MPI_Allreduce(
        &local_error,      
        &global_error,     
        1,                          
        MPI_DOUBLE,                
        MPI_SUM,                   
        pmp.comm
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error computing error.");

    return global_error;
}

double DPS::compute_maxnorm(const double* const f1, const double* const f2) const {
	double norm = 0;
    double priv_norm = 0;
    #pragma omp parallel firstprivate (priv_norm)
    {
        #pragma omp for schedule (static)
        for (int i = 0; i < pcinfo.x_cells_num * pcinfo.y_cells_num; i++){
            priv_norm = fmax(priv_norm, fabs(f1[i] - f2[i]));
        }

        #pragma omp critical
        {
            norm = fmax(priv_norm, norm);
        }
    }

    double global_norm = 0;

    int ret = MPI_Allreduce(
        &norm,                     
        &global_norm,               
        1,                         
        MPI_DOUBLE,               
        MPI_MAX,                
        pmp.comm            
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error computing function norm difference.");
    return global_norm;
}

bool DPS::stop_condition(const double* const f1, const double* const f2) const {
	double global_norm = compute_maxnorm(f1, f2); 
    return global_norm < eps;
}

void DPS::compute_loa(double* const df, const double* const f){
    int i = 0;
    int j = 0;
    int ret = MPI_SUCCESS;

    #pragma omp parallel for schedule(static) private(i)
    for (j = 1; j < pcinfo.y_cells_num - 1; j++){
        for (i = 1; i < pcinfo.x_cells_num - 1; i++){
            int index_i = pcinfo.x_cell_pos + i;
            int index_j = pcinfo.y_cell_pos + j;
            df[j * pcinfo.x_cells_num + i] = (
                    (f[j * pcinfo.x_cells_num + i    ] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                    (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i    ]) / hx_(index_i)
                ) / hhx_(index_i) + (
                    (f[ j      * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                    (f[(j + 1) * pcinfo.x_cells_num + i] - f[ j      * pcinfo.x_cells_num + i]) / hy_(index_j)
                ) / hhy_(index_j);
        }
    }

    if (send_message_lr == NULL)
        send_message_lr = new double [pcinfo.y_cells_num];
    if (send_message_rl == NULL)
        send_message_rl = new double [pcinfo.y_cells_num];
    if (send_message_td == NULL)
        send_message_td = new double [pcinfo.x_cells_num];
    if (send_message_bu == NULL)
        send_message_bu = new double [pcinfo.x_cells_num];

    if (recv_message_lr == NULL)
        recv_message_lr = new double [pcinfo.y_cells_num];
    if (recv_message_rl == NULL)
        recv_message_rl = new double [pcinfo.y_cells_num];
    if (recv_message_td == NULL)
        recv_message_td = new double [pcinfo.x_cells_num];
    if (recv_message_bu == NULL)
        recv_message_bu = new double [pcinfo.x_cells_num];

    #pragma omp parallel
    #pragma omp sections private(i, j)
    {
        #pragma omp section
        for (j = 0; j < pcinfo.y_cells_num; j++){
            send_message_lr[j] = f[ (j + 1) * pcinfo.x_cells_num - 1];
        }
        #pragma omp section
        for (j = 0; j < pcinfo.y_cells_num; j++){
            send_message_rl[j] = f[ j * pcinfo.x_cells_num + 0];
        }
        #pragma omp section
        for (i = 0; i < pcinfo.x_cells_num; i++){
            send_message_td[i] = f[ (pcinfo.y_cells_num - 1) * pcinfo.x_cells_num + i];
        }
        #pragma omp section
        for (i = 0; i < pcinfo.x_cells_num; i++){
            send_message_bu[i] = f[i];
        }
    }

    int send_amount = 0;
    int recv_amount = 0;

    #pragma omp sections private (ret)
    {
        #pragma omp section
        {
            if (not pcinfo.right){

                ret = MPI_Isend(
                    send_message_lr,                   
                    pcinfo.y_cells_num,                
                    MPI_DOUBLE,                              
                    pmp.rank + 1,            
                    DPS::LoaLeftRight,                
                    pmp.comm,                       
                    &(send_loa_reqs[send_amount])          
                );
                send_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message from left to right.");
            }
            if (not pcinfo.left){

                ret = MPI_Isend(
                    send_message_rl,                           
                    pcinfo.y_cells_num,                 
                    MPI_DOUBLE,                                
                    pmp.rank - 1,                      
                    DPS::LoaRightLeft,                     
                    pmp.comm,                       
                    &(send_loa_reqs[send_amount])            
                );
                send_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message from right to left.");
            }
            if (not pcinfo.bottom){

                ret = MPI_Isend(
                    send_message_td,                       
                    pcinfo.x_cells_num,                     
                    MPI_DOUBLE,                               
                    pmp.rank + pcinfo.x_proc_num,   
                    DPS::LoaTopDown,                       
                    pmp.comm,                          
                    &(send_loa_reqs[send_amount])          
                );
                send_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message top -> down.");
            }
            if (not pcinfo.top){

                ret = MPI_Isend(
                    send_message_bu,                        
                    pcinfo.x_cells_num,                     
                    MPI_DOUBLE,                             
                    pmp.rank - pcinfo.x_proc_num,  
                    DPS::LoaBottomUp,                      
                    pmp.comm,                           
                    &(send_loa_reqs[send_amount])          
                );
                send_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message bottom -> up.");
            }
        }

        #pragma omp section
        {
            if (not pcinfo.left){

                ret = MPI_Irecv(
                    recv_message_lr,                            
                    pcinfo.y_cells_num,                    
                    MPI_DOUBLE,                            
                    pmp.rank - 1,                 
                    DPS::LoaLeftRight,                    
                    pmp.comm,                         
                    &(recv_loa_reqs[recv_amount])            
                );
                recv_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message from left to right.");
            }
            if (not pcinfo.right){

                ret = MPI_Irecv(
                    recv_message_rl,                            
                    pcinfo.y_cells_num,                  
                    MPI_DOUBLE,                              
                    pmp.rank + 1,                  
                    DPS::LoaRightLeft,                     
                    pmp.comm,                        
                    &(recv_loa_reqs[recv_amount])           
                );
                recv_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message from right to left.");
            }
            if (not pcinfo.top){

                ret = MPI_Irecv(
                    recv_message_td,                         
                    pcinfo.x_cells_num,               
                    MPI_DOUBLE,                                
                    pmp.rank - pcinfo.x_proc_num,   
                    DPS::LoaTopDown,                      
                    pmp.comm,                          
                    &(recv_loa_reqs[recv_amount])        
                );
                recv_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message top -> down.");
            }
            if (not pcinfo.bottom){

                ret = MPI_Irecv(
                    recv_message_bu,                          
                    pcinfo.x_cells_num,                 
                    MPI_DOUBLE,                           
                    pmp.rank + pcinfo.x_proc_num, 
                    DPS::LoaBottomUp,                     
                    pmp.comm,                          
                    &(recv_loa_reqs[recv_amount])           
                );
                recv_amount++;

                if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message bottom -> up.");
            }
        } 
    }

    ret = MPI_Waitall(
        recv_amount,
        recv_loa_reqs,
        MPI_STATUS_IGNORE 
    );

    if (ret != MPI_SUCCESS) throw DPS_exception("Error waiting for recv's in compute_loa.");

    if (pcinfo.x_cells_num > 1 and pcinfo.y_cells_num > 1)
    {
        #pragma omp parallel private(i, j)
        {
            if (not pcinfo.left) {
                i = 0;
                #pragma omp for schedule (static) nowait
                for (j = 1; j < pcinfo.y_cells_num - 1; j++){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i  ] - recv_message_lr[j]) / hx_(index_i - 1) -
                            (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j      * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                            (f[(j + 1) * pcinfo.x_cells_num + i] - f[ j      * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
            if (not pcinfo.right) {
                i = pcinfo.x_cells_num - 1;
                #pragma omp for schedule (static) nowait
                for (j = 1; j < pcinfo.y_cells_num - 1; j++){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                            (recv_message_rl[j]            - f[j * pcinfo.x_cells_num + i    ]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j      * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                            (f[(j + 1) * pcinfo.x_cells_num + i] - f[ j      * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
            if (not pcinfo.top) {
                j = 0;
                #pragma omp for schedule (static) nowait
                for (i = 1; i < pcinfo.x_cells_num - 1; i++){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i    ] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                            (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i    ]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j      * pcinfo.x_cells_num + i] - recv_message_td[i]) / hy_(index_j - 1) -
                            (f[(j + 1) * pcinfo.x_cells_num + i] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
            if (not pcinfo.bottom) {
                j = pcinfo.y_cells_num - 1;
                #pragma omp for schedule (static) nowait
                for (i = 1; i < pcinfo.x_cells_num - 1; i++){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i    ] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                            (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i    ]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[j * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                            (recv_message_bu[i] - f[ j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
        }

        #pragma omp parallel
        #pragma omp sections private(i, j)
        {
            #pragma omp section
            {
                j = 0;
                i = 0;
                if (not pcinfo.top and not pcinfo.left) {
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i    ] - recv_message_lr[0]) / hx_(index_i - 1) -
                            (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j      * pcinfo.x_cells_num + i] - recv_message_td [0]) / hy_(index_j - 1) -
                            (f[(j + 1) * pcinfo.x_cells_num + i] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }

            #pragma omp section
            {
                j = 0;
                i = pcinfo.x_cells_num -1;
                if (not pcinfo.top and not pcinfo.right){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i  ] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                            (recv_message_rl[0]              - f[j * pcinfo.x_cells_num + i    ]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j      * pcinfo.x_cells_num + i] - recv_message_td [pcinfo.x_cells_num - 1]) / hy_(index_j - 1) -
                            (f[(j + 1) * pcinfo.x_cells_num + i] - f[ j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }

            #pragma omp section
            {
                j = pcinfo.y_cells_num -1;
                i = 0;
                if (not pcinfo.bottom and not pcinfo.left){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i    ] - recv_message_lr[pcinfo.y_cells_num - 1]) / hx_(index_i - 1) -
                            (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                            (recv_message_bu [0] - f[ j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }

            #pragma omp section
            {
                j = pcinfo.y_cells_num - 1;
                i = pcinfo.x_cells_num - 1;
                if (not pcinfo.bottom and not pcinfo.right){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                            (recv_message_rl [pcinfo.y_cells_num - 1] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[j * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                            (recv_message_bu [pcinfo.x_cells_num - 1] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
        }

    } else if (pcinfo.x_cells_num > 1 and pcinfo.y_cells_num == 1){
        j = 0;
        if (not pcinfo.top and not pcinfo.bottom) {
            #pragma omp parallel
            #pragma omp for schedule (static)
            for (i = 1; i < pcinfo.x_cells_num - 1; i++){
                int index_i = pcinfo.x_cell_pos + i;
                int index_j = pcinfo.y_cell_pos + j;
                df[j * pcinfo.x_cells_num + i] = (
                        (f[j * pcinfo.x_cells_num + i    ] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                        (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i    ]) / hx_(index_i)
                    ) / hhx_(index_i) + (
                        (f[j * pcinfo.x_cells_num + i] - recv_message_td[i]) / hy_(index_j - 1) -
                        (recv_message_bu[i] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                    ) / hhy_(index_j);
            }
        }

        #pragma omp parallel
        #pragma omp sections private(i, j)
        {
            #pragma omp section
            {
                j = 0;
                i = 0;
                if (not pcinfo.top and not pcinfo.bottom and not pcinfo.left) {
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i    ] - recv_message_lr [0]) / hx_(index_i - 1) -
                            (f[j * pcinfo.x_cells_num + i + 1] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[j * pcinfo.x_cells_num + i] - recv_message_td [0]) / hy_(index_j - 1) -
                            (recv_message_bu[0] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }

            #pragma omp section
            {
                j = 0;
                i = pcinfo.x_cells_num - 1;
                if (not pcinfo.top and not pcinfo.bottom and not pcinfo.right){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i] - f[j * pcinfo.x_cells_num + i - 1]) / hx_(index_i - 1) -
                            (recv_message_rl[0] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[j * pcinfo.x_cells_num + i] - recv_message_td [pcinfo.x_cells_num - 1]) / hy_(index_j - 1) -
                            (recv_message_bu [pcinfo.x_cells_num - 1] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
        }

    } else if (pcinfo.x_cells_num == 1 and pcinfo.y_cells_num > 1){
        i = 0;
        if (not pcinfo.left and not pcinfo.right) {
            #pragma omp parallel for schedule(static)  
            for (j = 1; j < pcinfo.y_cells_num - 1; j++){
                int index_i = pcinfo.x_cell_pos + i;
                int index_j = pcinfo.y_cell_pos + j;
                df[j * pcinfo.x_cells_num + i] = (
                        (f[j * pcinfo.x_cells_num + i] - recv_message_lr[j]) / hx_(index_i - 1) -
                        (recv_message_rl[j] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                    ) / hhx_(index_i) + (
                        (f[ j      * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                        (f[(j + 1) * pcinfo.x_cells_num + i] - f[ j      * pcinfo.x_cells_num + i]) / hy_(index_j)
                    ) / hhy_(index_j);
            }
        }

        #pragma omp parallel
        #pragma omp sections private(i, j)
        {
            #pragma omp section
            {
                j = 0;
                i = 0;
                if (not pcinfo.left and not pcinfo.right and not pcinfo.top) {
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i] - recv_message_lr [0]) / hx_(index_i - 1) -
                            (recv_message_rl[0] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[ j      * pcinfo.x_cells_num + i] - recv_message_td [0]) / hy_(index_j - 1) -
                            (f[(j + 1) * pcinfo.x_cells_num + i] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }

            #pragma omp section
            {
                j = pcinfo.y_cells_num - 1;
                i = 0;
                if (not pcinfo.left and not pcinfo.right and not pcinfo.bottom){
                    int index_i = pcinfo.x_cell_pos + i;
                    int index_j = pcinfo.y_cell_pos + j;
                    df[j * pcinfo.x_cells_num + i] = (
                            (f[j * pcinfo.x_cells_num + i  ] - recv_message_lr[j]) / hx_(index_i - 1) -
                            (recv_message_rl[j] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                        ) / hhx_(index_i) + (
                            (f[j * pcinfo.x_cells_num + i] - f[(j - 1) * pcinfo.x_cells_num + i]) / hy_(index_j - 1) -
                            (recv_message_bu [0] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                        ) / hhy_(index_j);
                }
            }
        }

    } else if (pcinfo.x_cells_num == 1 and pcinfo.y_cells_num == 1){
        i = 0;
        j = 0;
        if (not pcinfo.left and not pcinfo.right and not pcinfo.top and not pcinfo.bottom){
            int index_i = pcinfo.x_cell_pos + i;
            int index_j = pcinfo.y_cell_pos + j;
            df[j * pcinfo.x_cells_num + i] = (
                    (f[j * pcinfo.x_cells_num + i] - recv_message_lr[j]) / hx_(index_i - 1) -
                    (recv_message_rl[j] - f[j * pcinfo.x_cells_num + i]) / hx_(index_i)
                ) / hhx_(index_i)+ (
                    (f[j * pcinfo.x_cells_num + i] - recv_message_td[0]) / hy_(index_j - 1) -
                    (recv_message_bu [0] - f[j * pcinfo.x_cells_num + i]) / hy_(index_j)
                ) / hhy_(index_j);
        }
    }

    ret = MPI_Waitall(
        send_amount, 
        send_loa_reqs,
        MPI_STATUS_IGNORE
    );

    if (ret != MPI_SUCCESS) throw DPS_exception("Error waiting for sends after last compute_loa.");
}

void DPS::compute_r(double* const r, const double* const dp) const {
    #pragma omp parallel for schedule(static) 
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            r[j * pcinfo.x_cells_num + i] =
                dp[j * pcinfo.x_cells_num + i] -
                F(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + j));
        }
    }
}

void DPS::compute_g(double* const g, const double* const r, const double alpha) const {
    #pragma omp parallel for schedule(static) 
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            g[j * pcinfo.x_cells_num + i] = r[j * pcinfo.x_cells_num + i] - alpha * g[j * pcinfo.x_cells_num + i];
        }
    }
}

void DPS::compute_p(const double tau, const double* const g, const double* const p_prev) {
    #pragma omp parallel for schedule(static) 
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            p[j * pcinfo.x_cells_num + i] = p_prev[j * pcinfo.x_cells_num + i] - tau * g[j * pcinfo.x_cells_num + i];
        }
    }
}

double DPS::compute_sprod(const double* const f1, const double* const f2) const {
    double local_scalar_product = 0;

    #pragma omp parallel for schedule(static) reduction(+:local_scalar_product)
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            double hhx = hhx_(pcinfo.x_cell_pos + i);
            double hhy = hhx_(pcinfo.y_cell_pos + j);
            local_scalar_product += hhx * hhy * f1[j * pcinfo.x_cells_num + i] * f2[j * pcinfo.x_cells_num + i];
        }
    }

    double global_scalar_product = 0;

    int ret = MPI_Allreduce(
        &local_scalar_product,      
        &global_scalar_product,     
        1,                          
        MPI_DOUBLE,                
        MPI_SUM,                   
        pmp.comm
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error reducing scalar_product.");

    return global_scalar_product;
}

void DPS::initialize_border_with_zero(double* const f){
    if (pcinfo.left){
        #pragma omp parallel for schedule(static) 
        for (int j = 0; j < pcinfo.y_cells_num; j++){
            f[j * pcinfo.x_cells_num + 0] = 0;
        }
    }
    if (pcinfo.right){
        #pragma omp parallel for schedule(static) 
        for (int j = 0; j < pcinfo.y_cells_num; j++){
            f[j * pcinfo.x_cells_num + (pcinfo.x_cells_num - 1)] = 0;
        }
    }
    if (pcinfo.top){
        #pragma omp parallel for schedule(static) 
        for (int i = 0; i < pcinfo.x_cells_num; i++){
            f[0 * pcinfo.x_cells_num + i] = 0;
        }
    }
    if (pcinfo.bottom){
        #pragma omp parallel for schedule(static) 
        for (int i = 0; i < pcinfo.x_cells_num; i++){
            f[(pcinfo.y_cells_num - 1) * pcinfo.x_cells_num + i] = 0;
        }
    }
}

void DPS::initialize_with_border_function(double* const f){
    #pragma omp parallel
    {
        if (pcinfo.left){
            #pragma omp for schedule (static)
            for (int j = 0; j < pcinfo.y_cells_num; j++){
                double value = phi(x_(pcinfo.x_cell_pos + 0), y_(pcinfo.y_cell_pos + j));
                f[j * pcinfo.x_cells_num + 0] = value;
            }
        }
        if (pcinfo.right){
            #pragma omp for schedule (static)
            for (int j = 0; j < pcinfo.y_cells_num; j++){
                double value = phi(x_(pcinfo.x_cell_pos + (pcinfo.x_cells_num - 1)), y_(pcinfo.y_cell_pos + j));
                f[j * pcinfo.x_cells_num + (pcinfo.x_cells_num - 1)] = value;
            }
        }
        if (pcinfo.top){
            #pragma omp for schedule (static)
            for (int i = 0; i < pcinfo.x_cells_num; i++){
                double value = phi(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + 0));
                f[0 * pcinfo.x_cells_num + i] = value;
            }
        }
        if (pcinfo.bottom){
            #pragma omp for schedule (static)
            for (int i = 0; i < pcinfo.x_cells_num; i++){
                double value = phi(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + (pcinfo.y_cells_num - 1)));
                f[(pcinfo.y_cells_num - 1) * pcinfo.x_cells_num + i] = value;
            }
        }

        #pragma omp for schedule (static)
        for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
            memset(&(f[j * pcinfo.x_cells_num + static_cast<int>(pcinfo.left)]), 0,
                (pcinfo.x_cells_num - static_cast<int>(pcinfo.right) - static_cast<int>(pcinfo.left)) * sizeof(*f));
        }
    }
}

void DPS::PrintP(const string& dir_name) const {
    std::stringstream ss;
    ss << "./" << dir_name << "/fa" << pmp.rank << ".txt";
    std::fstream fout(ss.str().c_str(), std::fstream::out);

    for (int j = 0; j < pcinfo.y_cells_num; ++j) {
        for (int i = 0; i < pcinfo.x_cells_num; ++i) {
            fout << x_(pcinfo.x_cell_pos + i) << " " << y_(pcinfo.y_cell_pos + j) 
                 << " " << p[j * pcinfo.x_cells_num + i] << std::endl;
        }
    }

    fout.close();
}

