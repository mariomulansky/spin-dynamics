from pylab import *

for N in [ 1000 , 10000 , 100000 , 1000000 ]:
    data = loadtxt( "resultN%d.dat" % N )
    semilogy( data[:,0] , data[:,2] , "-" , lw=2.0 , label="N=%d" % N )

data = loadtxt( "resultN100000_old.dat" )
semilogy( data[:,0] , data[:,2] , "x" , label="N=100000" )


xlabel( "time" )
ylabel( "FT Amplitude" )
legend( loc="upper right" )
show()
