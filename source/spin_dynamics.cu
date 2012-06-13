#include <iostream>

#include <thrust/device_vector.h>

#include "spin_stepper.hpp"

typedef double value_type;
typedef thrust::device_vector< value_type > device_type;

static const int N = 1000;
static const int steps = 100;
static const double dt = 0.1;

int main()
{

    device_type s_x( N+2 );
    device_type s_y( N+2 );
    device_type s_z( N+2 );
    
    device_type b_x( N );
    device_type b_y( N );
    device_type b_z( N );
    device_type b_norm( N );
    
    device_type h_x( N );
    device_type h_y( N );
    device_type h_z( N );
    
    spin_stepper< device_type , value_type > stepper( N , h_x , h_y , h_z );

    for( int n=0 ; n<steps ; ++n )
    {
        stepper.do_step( s_x , s_y , s_z , 
                         b_x , b_y , b_z , 
                         b_norm , dt );
    }

}
