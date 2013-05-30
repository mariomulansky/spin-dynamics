from pylab import *
from scipy import stats

rc( "font" , size=20 )

ax=axes([0.13,0.13,0.8,0.78])

N = 1000000
T = "5E4"

dt = 0.1

m = 1
J = 0.7/1.1220184543**m

qs = 2*pi*arange(1,10)/(10.0*20)

slope_avrg=0

for q in qs:
    data = loadtxt( "../data_study_N%d_T%s/fourier_J%.5f_q%.5f.dat" % ( N , T , J , q ) )
    #data = data[ data[:,2] > 10.0 ]
    #data = data[ len(data)/2: ]

    ax.plot( data[:,0]*q**2 , log10(data[:,2]) , '-' , label="%.4f" % q )
    
    slope, intercept, r_value, p_value, std_err = stats.linregress( data[:,0]*q**2 , log10(data[:,2]) )

    slope_avrg += slope

tq = linspace( 0 , 500 , 100 )
#ax.plot( tq , slope_avrg*tq/10+intercept , '--k' , lw=2 )

ax.set_title( "J = %.3f" %J )
ax.set_xlabel( "$t\cdot q^2$" )
ax.set_ylabel( "$F_q$" )

#ax.axis( [ 0 , 15 , 10, 2E5 ] )

ax.legend( loc = "upper right" , prop={"size":12} )

#savefig( "relax_N%d_%.2f.pdf" % (N,dt) )
show()
