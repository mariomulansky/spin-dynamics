from pylab import *

rc( "font" , size=20 )

ax=axes([0.15,0.15,0.8,0.8])

N = 1000000
dt = 0.1

qs = 2*pi/arange(10,101,5)
D = 0.75

for q in qs:
    data = loadtxt( "../data_N%d_dt%.2f/fourier_q%.5f.dat" % ( N , dt , q ) )
    data = data[ data[:,2] > 10.0 ]
    if q > 0.3:
        l = "$q = %.3f$" % q
    else:
        l = ""

    ax.semilogy( data[:,0]*q**2 , data[:,2]/data[0,2] , label=l  )

tq = linspace( 0 , 10 , 100 )
#ax.plot( tq , 1E5*exp( -D*tq ) , '--k' , lw=2 , label="$\sim e^{-0.75\, t\cdot q^2}$" )

ax.set_xlabel( "$t\cdot q^2$" )
ax.set_ylabel( "$F_q$" )

#ax.axis( [ 0 , 15 , 10, 2E5 ] )

ax.legend( loc = "upper right" , prop={"size":16} )

#savefig( "relax_N%d_%.2f.pdf" % (N,dt) )
show()
