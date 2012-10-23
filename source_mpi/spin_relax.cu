#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>

#include <thrust/device_vector.h>

#include "spin_stepper.hpp"
#include "fourier_analyzer.hpp"
#include "spin_initializer.hpp"

typedef double value_type;
typedef thrust::device_vector< value_type > device_type;

typedef std::vector< value_type > host_type;

double time_diff_in_ms( timeval &t1 , timeval &t2 )
{ return (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0 + 0.5; }

void spin_relax( const int N , const double J , const double q , 
                 const int steps, const double dt , 
                 const double beta0 , const double nu )
{
    const double delta_sqr = 1 - 2*J*J;
    std::clog << "System size: " << N << ", sigma: " << delta_sqr << std::endl;


    /* define initial conditions */
    host_type s_x_host( N+2 , 0.0 );
    host_type s_y_host( N+2 , 0.0 );
    host_type s_z_host( N+2 , 0.0 );

    host_type h_x_host( N , 1.0 );
    host_type h_y_host( N , 1.0 );
    host_type h_z_host( N , 1.0 );

    /* some test intial conditions */
    /*
      s_x_host[ N/2+1 ] = sqrt(0.5);
      s_y_host[ N/2+1 ] = sqrt(0.5);
      s_x_host[ N/2+2 ] = 1.0;
    */

    spin_initializer< value_type > init( q , nu );

    init.init_gaussian_random( delta_sqr , h_x_host , h_y_host , h_z_host );
    //init.init_normalized_random( h_x_host , h_y_host , h_z_host );
    init.init_normalized_random( s_x_host , s_y_host , s_z_host );

    // fix edges to zero:
    s_x_host[0] = 0.0; s_x_host[N+1] = 0.0;
    s_y_host[0] = 0.0; s_y_host[N+1] = 0.0;
    s_z_host[0] = 0.0; s_z_host[N+1] = 0.0;

    init.relax( h_x_host , h_y_host , h_z_host , 
                s_x_host , s_y_host , s_z_host ,
                beta0 );
    
    //std::clog << "initialization finished" << std::endl;
    
    // initialize device vectors
    //std::clog << "copy to device..." << std::endl;
    
    // vectors s_*_host have length N+2 to account for boundary conditions
    device_type s_x( s_x_host.begin() , s_x_host.end() );
    device_type s_y( s_y_host.begin() , s_y_host.end() );
    device_type s_z( s_z_host.begin() , s_z_host.end() );
    
    device_type h_x( h_x_host.begin() , h_x_host.end() );
    device_type h_y( h_y_host.begin() , h_y_host.end() );
    device_type h_z( h_z_host.begin() , h_z_host.end() );
    
    device_type energies( N );
    
    spin_stepper< device_type , value_type > stepper( N , J , h_x , h_y , h_z );
    fourier_analyzer< device_type , value_type > fourier( N , q );
    
    stepper.energies( s_x , s_y , s_z , energies );
    
    char filename[255];
    sprintf( filename , "../data_study_N%d/fourier_J%.5f_q%.5f.dat" , N , J , q/N );
    std::ofstream res_file( filename );
    if( !res_file.good() )
    {
        std::cout << "unable to open file: " << filename << std::endl;
        exit(-1);
    }

    //std::clog << "Starting time evolution..." << std::endl;

    timeval elapsed_time_start , elapsed_time_end;
    gettimeofday(&elapsed_time_start , NULL);
	
    int n;
    
    for( n=0 ; n<steps ; ++n )
    {
        
        
        if( (n%10) == 0 )
        {
            res_file << n*dt << '\t';
            stepper.energies( s_x , s_y , s_z , energies );
            res_file << thrust::reduce( energies.begin() , energies.end() ) << '\t';
            const double fq = fourier.analyze( energies )/N;
            res_file << fq << std::endl;
            if( fq < 1.0 ) 
                break;
        }
        
        stepper.do_step( s_x , s_y , s_z , dt );
    }

    gettimeofday(&elapsed_time_end , NULL);
    double elapsed_time = 0.001 * time_diff_in_ms( elapsed_time_start , elapsed_time_end );

    std::cout << "Finished " << n+1 << " steps for N=" << N << ", J=" << J << ", q=" << q/N;
    std::cout << " in " << elapsed_time << " seconds" << std::endl;
}
