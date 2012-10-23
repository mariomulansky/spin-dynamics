#include <time.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>


static boost::variate_generator<boost::mt19937, boost::normal_distribution<> > 
   generator( boost::mt19937(time(0)) , boost::normal_distribution<>() );

double gen_normal(void)
{
  return generator();
}
