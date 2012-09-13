from pylab import *

rc( "font" , size=18 )

N = 10000000
qs = 2*pi/arange(10,101,5)

D = 0.75

for q in qs:
    data = loadtxt( "../data_N%d/fourier_q%.5f.dat" % ( N , q ) )
    data = data[ data[:,2] > 10.0 ]
    if q > 0.3:
        l = "$q = %.3f$" % q
    else:
        l = ""

    semilogy( data[:,0]*q**2 , data[:,2] , label=l  )

tq = linspace( 0 , 10 , 100 )
plot( tq , 1E5*exp( -D*tq ) , '--k' , lw=2 , label="$\sim e^{-0.75\, t\cdot q^2}$" )

xlabel( "$t\cdot q^2$" )
ylabel( "$F_q$" )

legend( loc = "upper right" , prop={"size":16} )

savefig( "relax_N%d.pdf" % N )
show()
