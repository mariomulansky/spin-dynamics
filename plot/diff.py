from pylab import *
from scipy import stats
from os import path

rc( "font" , size=20 )

N = 1000000
T = "5E4"

M = 10
K = 10
J = 0.7


avrg_file_name = "../data_study_N%d_T%s/fourier_avrg.dat" % ( N , T )

slopes = []
Js = []

if path.exists( avrg_file_name ):
    avrg_data = loadtxt( avrg_file_name )
    
    slopes = avrg_data[0,:]
    Js = avrg_data[1,:]

else:
   
    while J > 0.1:

        print "Averaging J =" , J 

        slope_avrg=0

        for k in arange( 1 , K ):
            q = 2*pi*k / (K*20)
        
            data = loadtxt( "../data_study_N%d_T%s/fourier_J%.5f_q%.5f.dat" % ( N , T , J , q ) )
            data = data[ data[:,2] > 20.0 ]
            data = data[ len(data)/2: ]
    
            slope, intercept, r_value, p_value, std_err = stats.linregress( data[:,0]*q**2 , log10(data[:,2]) )

            slope_avrg += slope

        slopes = append( slopes , -slope_avrg/K )
        Js = append( Js , J )
    
        J /= 1.1220184543

    savetxt( avrg_file_name , [ slopes , Js ] )

plot( log10(Js) , log10(slopes) , 'ob' )
plot( log10(Js) , log10(10*Js**8) , '--k' , lw=2 )

xlabel("J")
ylabel("D")

xticks( arange( -0.9,-0.0,0.2) )

show()
