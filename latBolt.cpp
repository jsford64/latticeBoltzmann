#include <fmt/format.h>
#include <fmt/ostream.h>
#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>


int main(int argc, char* argv[]) {
    using namespace Eigen;
    
    // Problem setup
    constexpr int     H = 1;   // Length of X grid, Y grid
    constexpr double U0 = 1;   // Physical characteristic velocity (lid)
    constexpr double Re = 300; // Reynold's number

    // Lattice parameters
    constexpr int      dx = 1;
    constexpr int      dy = 1;
    constexpr int      dt = 1;
    constexpr double   cs = 1/std::sqrt(3.0);
    constexpr double rhoo = 5.00;
    constexpr double   Ma = 0.1;
    constexpr double  tau = 0.75;

    // Solver convergence tolerance
    constexpr double tolerance = 1e-8;

    // Constants
    constexpr double uo = Ma*cs;                // Lattice characteristic velocity
    constexpr double ur = U0/uo;                // Reference velocity
    constexpr double omega = 1.0/tau;           // Collision frequency
    constexpr double alpha = cs*cs*(tau-0.5);   // Lattice kinematic viscosity
    constexpr double N = Re*alpha/uo;           // Lattice Re matching physical Re
    constexpr int nx = 2*std::floor(N/2.0);     // Count of nodes in X direction
    constexpr int ny = nx;                      // Count of nodes in Y direction
    VectorXd x = VectorXd::LinSpaced(nx, 0, H); // X nodes
    VectorXd y = VectorXd::LinSpaced(ny, 0, H); // Y nodes

    fmt::print("{} x {}\n", nx, ny);

    // DQ29 model parameters
    VectorXd w(9);  // Weight in equilibrium distribution function
    w << 4/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/36.0, 1/36.0, 1/36.0, 1/36.0;
    VectorXd cx(9); // Discrete velocity X component
    cx << 0, 1, 0, -1, 0, 1, -1, -1, 1;
    VectorXd cy(9); // Discrete velocity Y component
    cy << 0, 0, 1, 0, -1, 1, 1, -1, -1;

    // Initialization
    Tensor<double,3>   f(9,nx,ny);
    Tensor<double,3> feq(9,nx,ny);

    // NOTE(Jordan): I have to use loops to set the Tensors to zero
    // because there is a bug in the Eigen setZero function for tensors.
    for(int i=0; i<nx; ++i) {
        for(int j=0; j<ny; ++j) {
            for(int k=0; k<9; ++k) {
                f(k,i,j) = 0.0;
                feq(k,i,j) = 0.0;
            }
        }
    }

    MatrixXd rho(nx, ny); rho.fill(rhoo);
    MatrixXd  u(nx, ny);     u.setZero();
    MatrixXd  v(nx, ny);     v.setZero();
    MatrixXd ut(nx, ny);    ut.setZero();
    MatrixXd vt(nx, ny);    vt.setZero();

    double error = std::numeric_limits<double>::infinity();
    int iterations = 0;

    for(int i=0; i<u.rows(); ++i) {
        u(i, u.cols()-1) = uo;
        v(i, u.cols()-1) = 0.0;
    }

    // Solve the governing equations
    while( error > tolerance /*&& iterations < 1000*/ ) {
        // Collision step
        for(int i=0; i<nx; ++i) {
            for(int j=0; j<ny; ++j) {
                const double t1 = u(i,j)*u(i,j)+v(i,j)*v(i,j);
                for(int k=0; k<9; ++k) {
                    const double t2 = u(i,j)*cx[k]+v(i,j)*cy[k];
                    feq(k,i,j) = rho(i,j)*w[k]*(1.0+3.0*t2+4.5*t2*t2-1.5*t1); // Note(Jordan): Could use horner's method here.
                    f(k,i,j) = omega*feq(k,i,j) + (1.0 - omega) * f(k,i,j);
                }
            }
        }

        // Streaming step
        for(int j=0; j<ny; ++j) {
            // Right to left
            for(int i=nx-1; i>0; --i) {
                f(1,i,j) = f(1,i-1,j);
            }
            // Left to right
            for(int i=0; i<nx-1; ++i) {
                f(3,i,j) = f(3,i+1,j);
            }
        } 

        // Top to bottom
        for(int j=ny-1; j>0; --j) {
            for(int i=0; i<nx; ++i) {
                f(2,i,j) = f(2,i,j-1);
            }
            for(int i=nx-1; i>0; --i) {
                f(5,i,j) = f(5,i-1,j-1);
            }
            for(int i=0; i<nx-1; ++i) {
                f(6,i,j) = f(6,i+1,j-1);
            }
        }

        // Bottom to top
        for(int j=0; j<ny-1; ++j) {
            for(int i=0; i<nx; ++i) {
                f(4,i,j) = f(4,i,j+1);
            }
            for(int i=0; i<nx-1; ++i) {
                f(7,i,j) = f(7,i+1,j+1);
            }
            for(int i=nx-1; i>0; --i) {
                f(8,i,j) = f(8,i-1,j+1);
            }
        }

        // East/West boundary conditions
        for(int j=0; j<ny; ++j) {
            // Bounce back on west boundary
            f(1,0,j) = f(3,0,j);
            f(5,0,j) = f(7,0,j);
            f(8,0,j) = f(6,0,j);

            // Bounce back on east boundary
            f(3,nx-1,j) = f(1,nx-1,j);
            f(7,nx-1,j) = f(5,nx-1,j);
            f(6,nx-1,j) = f(8,nx-1,j);
        }

        // Bounce back on south boundary
        for(int i=0; i<nx; ++i) {
            f(2,i,0) = f(4,i,0);
            f(5,i,0) = f(7,i,0);
            f(6,i,0) = f(8,i,0);
        }

        // Moving lid, north boundary
        for(int i=1; i<nx-1; ++i) {
            double rhon = 1*(f(0,i,ny-1) + f(1,i,ny-1) + f(3,i,ny-1)) + 
                          2*(f(2,i,ny-1) + f(6,i,ny-1) + f(5,i,ny-1));
            f(4,i,ny-1) = f(2,i,ny-1);
            f(8,i,ny-1) = f(6,i,ny-1) + rhon*uo/6.0;
            f(7,i,ny-1) = f(5,i,ny-1) - rhon*uo/6.0;
        }


        // Update rho
        // NOTE(Jordan): Could maybe replace with: "rho = f.sum(0);"?
        for(int j=0; j<ny; ++j) {
            for(int i=0; i<nx; ++i) {
                rho(i,j) = 0.0;
                for(int k=0; k<9; ++k) {
                    rho(i,j) += f(k,i,j);
                }
            }
        }
        for(int i=0; i<nx; ++i) {
            rho(i,ny-1) = 1*( f(0,i,ny-1) + f(1,i,ny-1) + f(3,i,ny-1) ) +
                          2*( f(2,i,ny-1) + f(6,i,ny-1) + f(5,i,ny-1) );
        }

        // Update u, v
        for(int i=1; i<nx; ++i) {
            for(int j=1; j<ny-1; ++j) {
                double usum = 0.0;
                double vsum = 0.0;
                for(int k=0; k<9; ++k) {
                    usum += f(k,i,j)*cx[k];
                    vsum += f(k,i,j)*cy[k];
                }
                u(i,j) = usum / rho(i,j);
                v(i,j) = vsum / rho(i,j);
            }
        }

        // Error monitoring
        error = ((u-ut).norm() + (v-vt).norm()) / (nx*ny);
        ut = u;
        vt = v;
        iterations++;
        if( iterations % 100 == 0 ) {
            fmt::print("[{}]: {:0.8f} / {}\n", iterations, error, tolerance);
        }
    }

    // Convert from lattice units to physical units.
    u = ur*u;
    v = ur*v;

    // Save the results to .csv files for plotting in python.
    std::ofstream ufile("u.csv");
    std::ofstream vfile("v.csv");
    std::ofstream xfile("x.csv");
    std::ofstream yfile("y.csv");
    fmt::print(ufile, "{}\n", u);
    fmt::print(vfile, "{}\n", v);
    fmt::print(xfile, "{}\n", x);
    fmt::print(yfile, "{}\n", y);

    return 0;
}
