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

int N = 10;
static const int steps = 11;
static const double dt = 0.1;

double time_diff_in_ms( timeval &t1 , timeval &t2 )
{ return (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0 + 0.5; }

int main( int argc , char** argv )
{

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

    spin_initializer< value_type > init( 0.0 , 0.0 );

    init.init_normalized_random( h_x_host , h_y_host , h_z_host );
    init.init_normalized_random( s_x_host , s_y_host , s_z_host );


    // fix edges to zero:
    s_x_host[0] = 0.0; s_x_host[N+1] = 0.0;
    s_y_host[0] = 0.0; s_y_host[N+1] = 0.0;
    s_z_host[0] = 0.0; s_z_host[N+1] = 0.0;


    char filename[255];
    sprintf( filename , "h_x_N%d.dat" , N );
    std::ofstream h_file( filename );
    thrust::copy( h_x_host.begin(), h_x_host.end(), 
                  std::ostream_iterator<value_type>( h_file , "\n"));
    h_file.close();
    sprintf( filename , "h_y_N%d.dat" , N );
    h_file.open( filename );
    thrust::copy( h_y_host.begin(), h_y_host.end(), 
                  std::ostream_iterator<value_type>( h_file , "\n"));
    h_file.close();
    sprintf( filename , "h_z_N%d.dat" , N );
    h_file.open( filename );
    thrust::copy( h_z_host.begin(), h_z_host.end(), 
                  std::ostream_iterator<value_type>( h_file , "\n"));
    h_file.close();

    sprintf( filename , "s_x_N%d.dat" , N );
    std::ofstream s_file( filename );
    thrust::copy( s_x_host.begin(), s_x_host.end(), 
                  std::ostream_iterator<value_type>( s_file , "\n"));
    s_file.close();
    sprintf( filename , "s_y_N%d.dat" , N );
    s_file.open( filename );
    thrust::copy( s_y_host.begin(), s_y_host.end(), 
                  std::ostream_iterator<value_type>( s_file , "\n"));
    s_file.close();
    sprintf( filename , "s_z_N%d.dat" , N );
    s_file.open( filename );
    thrust::copy( s_z_host.begin(), s_z_host.end(), 
                  std::ostream_iterator<value_type>( s_file , "\n"));
    s_file.close();


    // initialize device vectors
    std::clog << "copy to device..." << std::endl;
    
    // vectors s_*_host have length N+2 to account for boundary conditions
    device_type s_x( s_x_host.begin() , s_x_host.end() );
    device_type s_y( s_y_host.begin() , s_y_host.end() );
    device_type s_z( s_z_host.begin() , s_z_host.end() );
    
    device_type h_x( h_x_host.begin() , h_x_host.end() );
    device_type h_y( h_y_host.begin() , h_y_host.end() );
    device_type h_z( h_z_host.begin() , h_z_host.end() );
    
    device_type energies( N );
    
    spin_stepper< device_type , value_type > stepper( N , h_x , h_y , h_z );
    
    std::clog << "Starting time evolution..." << std::endl;

    timeval elapsed_time_start , elapsed_time_end;
    gettimeofday(&elapsed_time_start , NULL);
    
    for( int n=0 ; n<steps ; ++n )
    {
        stepper.energies( s_x , s_y , s_z , energies );
    
        char filename[255];
        sprintf( filename , "en_N%d_step%d.dat" , N , n );
        std::ofstream en_file( filename );
        thrust::copy( energies.begin(), energies.end(), 
                      std::ostream_iterator<value_type>( en_file , "\n"));
        en_file.close();

        stepper.do_step( s_x , s_y , s_z , dt );
    }

    gettimeofday(&elapsed_time_end , NULL);
    double elapsed_time = 0.001 * time_diff_in_ms( elapsed_time_start , elapsed_time_end );

    std::clog << "Finished " << steps << " steps for N=" << N << " in " << elapsed_time << " seconds" << std::endl;
}
