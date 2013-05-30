#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>

#include <sys/time.h>

#include <thrust/device_vector.h>

#include "spin_stepper.hpp"
#include "spin_stepper_cuda.hpp"
#include "fourier_analyzer.hpp"
#include "spin_initializer.hpp"

typedef double value_type;
typedef thrust::device_vector< value_type > device_type;

typedef std::vector< value_type > host_type;

int N = 1000000;
static const int steps = 5001;
static const value_type dt = 0.1;
value_type q = 2.0*M_PI * N/40;
static const value_type beta0 = 0.0001;
static const value_type nu = 0.5;

double time_diff_in_ms( timeval &t1 , timeval &t2 )
{ return (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0 + 0.5; }

int main( int argc , char** argv )
{

    if( argc>1 )
    {
        N = atoi( argv[1] );
        q = 2.0*M_PI * N/40;
    }

    std::clog << "System size: " << N << ", q: " << q << std::endl;


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

    init.init_normalized_random( h_x_host , h_y_host , h_z_host );
    init.init_normalized_random( s_x_host , s_y_host , s_z_host );

    // fix edges to zero:
    s_x_host[0] = 0.0; s_x_host[N+1] = 0.0;
    s_y_host[0] = 0.0; s_y_host[N+1] = 0.0;
    s_z_host[0] = 0.0; s_z_host[N+1] = 0.0;

    init.relax( h_x_host , h_y_host , h_z_host , 
                s_x_host , s_y_host , s_z_host ,
                beta0 );
    
    std::clog << "initialization finished" << std::endl;
    
    // initialize device vectors
    std::clog << "copy to device..." << std::endl;
    
    // vectors s_*_host have length N+2 to account for boundary conditions
    device_type s_x1( s_x_host.begin() , s_x_host.end() );
    device_type s_y1( s_y_host.begin() , s_y_host.end() );
    device_type s_z1( s_z_host.begin() , s_z_host.end() );
    device_type s_x2( s_x_host.begin() , s_x_host.end() );
    device_type s_y2( s_y_host.begin() , s_y_host.end() );
    device_type s_z2( s_z_host.begin() , s_z_host.end() );
    
    device_type h_x( h_x_host.begin() , h_x_host.end() );
    device_type h_y( h_y_host.begin() , h_y_host.end() );
    device_type h_z( h_z_host.begin() , h_z_host.end() );
    
    device_type energies( N );
    
    spin_stepper< device_type , value_type > stepper( N , h_x , h_y , h_z );
    spin_stepper_cuda< device_type , value_type > stepper_cuda( N , h_x , h_y , h_z );
    fourier_analyzer< device_type , value_type > fourier( N , q );
    
    stepper.energies( s_x1 , s_y1 , s_z1 , energies );
    
    std::ofstream en_file( "en0.dat" );
    thrust::copy( energies.begin(), energies.end(), 
                  std::ostream_iterator<value_type>( en_file , "\n"));
    en_file.close();

    char filename[255];
    sprintf( filename , "result1_N%d.dat" , N );
    std::ofstream res_file1( filename );
    sprintf( filename , "result2_N%d.dat" , N );
    std::ofstream res_file2( filename );

    std::clog << "Starting time evolution..." << std::endl;

    timeval elapsed_time_start , elapsed_time_end;
    gettimeofday(&elapsed_time_start , NULL);
    
    for( int n=0 ; n<steps ; ++n )
    {
        stepper.do_step( s_x1 , s_y1 , s_z1 , dt );
        stepper_cuda.do_step( s_x2 , s_y2 , s_z2 , dt );
        
        if( (n%10) == 0 )
        {
            res_file1 << n*dt << '\t';
            stepper.energies( s_x1 , s_y1 , s_z1 , energies );
            res_file1 << thrust::reduce( energies.begin() , energies.end() ) << '\t';
            res_file1 << fourier.analyze( energies )/N << std::endl;

            res_file2 << n*dt << '\t';
            stepper_cuda.energies( s_x2 , s_y2 , s_z2 , energies );
            res_file2 << thrust::reduce( energies.begin() , energies.end() ) << '\t';
            res_file2 << fourier.analyze( energies )/N << std::endl;
        }
        
    }

    gettimeofday(&elapsed_time_end , NULL);
    double elapsed_time = 0.001 * time_diff_in_ms( elapsed_time_start , elapsed_time_end );

    std::clog << "Finished " << steps << " steps for N=" << N << " in " << elapsed_time << " seconds" << std::endl;
    
    en_file.open( "enT.dat" );
    thrust::copy( energies.begin(), energies.end(), 
                  std::ostream_iterator<value_type>( en_file , "\n"));
}
