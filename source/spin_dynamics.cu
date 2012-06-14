#include <iostream>
#include <fstream>
#include <boost/random.hpp>

#include <thrust/device_vector.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "spin_stepper.hpp"
#include "fourier_analyzer.hpp"
#include "spin_initializer.hpp"

typedef double value_type;
typedef thrust::device_vector< value_type > device_type;

typedef std::vector< value_type > host_type;

static const int N = 10000;
static const int steps = 1001;
static const double dt = 0.1;
static const double q = 25.0;
static const double nu = 0.5;

int main()
{
    boost::mt19937 rng;

    /* define initial conditions */
    host_type s_x_host( N+2 , 0.0 );
    host_type s_y_host( N+2 , 0.0 );
    host_type s_z_host( N+2 , 0.0 );

    host_type h_x_host( N , 1.0 );
    host_type h_y_host( N , 0.0 );
    host_type h_z_host( N , 0.0 );

    /* some test intial conditions */
    /*
    s_x_host[ N/2+1 ] = sqrt(0.5);
    s_y_host[ N/2+1 ] = sqrt(0.5);
    s_x_host[ N/2+2 ] = 1.0;
    */

    spin_initializer< value_type > init( q , nu , rng );

    init.init_normalized_random( h_x_host , h_y_host , h_z_host );
    init.init_normalized_random( s_x_host , s_y_host , s_z_host );

    // fix edges to zero:
    s_x_host[0] = 0.0; s_x_host[N+1] = 0.0;
    s_y_host[0] = 0.0; s_y_host[N+1] = 0.0;
    s_z_host[0] = 0.0; s_z_host[N+1] = 0.0;

    init.relax( h_x_host , h_y_host , h_z_host , 
                s_x_host , s_y_host , s_z_host );

    std::clog << "initialization finished" << std::endl;
    
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
    fourier_analyzer< device_type , value_type > fourier( N , q );

    stepper.energies( s_x , s_y , s_z , energies );

    std::ofstream en_file( "en0.dat" );
    thrust::copy( energies.begin(), energies.end(), 
                  std::ostream_iterator<value_type>( en_file , "\n"));
    en_file.close();

    for( int n=0 ; n<steps ; ++n )
    {
        stepper.do_step( s_x , s_y , s_z , dt );
        
        if( (n%10) == 0 )
        {
            std::cout << n*dt << '\t';
            stepper.energies( s_x , s_y , s_z , energies );
            std::cout << thrust::reduce( energies.begin() , energies.end() ) << '\t';
            std::cout << fourier.analyze( energies ) << std::endl;
        }
        
    }

    en_file.open( "enT.dat" );
    thrust::copy( energies.begin(), energies.end(), 
                  std::ostream_iterator<value_type>( en_file , "\n"));
}
