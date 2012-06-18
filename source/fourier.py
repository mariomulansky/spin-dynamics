from pylab import *
from numpy.fft import rfft

data0 = loadtxt( "en0.dat" )
data1 = loadtxt( "enT.dat" )

ft0 = rfft( data0 )
ft1 = rfft( data1 )

#figure()
#plot( data )

figure()
semilogy( abs(ft0[:1000])**2 )
semilogy( abs(ft1[:1000])**2 )

show()
