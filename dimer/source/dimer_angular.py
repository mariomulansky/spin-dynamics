import numpy
import numpy as np
import pylab as pl
from scipy.integrate import odeint as sc_odeint
from sympy import *
import pickle
from IPython import embed

init_printing(use_unicode=True)

# Start with finding the equations of motion analytically usinf sympy
p1,p2,t1,t2,x1,x2,y1,y2,t = symbols('p1 p2 t1 t2 x1 x2 y1 y2 t' , real=True)

H = Matrix( [[2, 0, 1, 0], [0, -2, 2, 1], [1, 2, 0, 0], [0, 1, 0, 0]] )

# triplet state in lab basis
tr = Matrix( [[exp(-I*(p1 + p2)/2)*cos(t1/2)*cos(t2/2)], 
              [exp(I*(p2 - p1)/2)*cos(t1/2)*sin(t2/2)], 
              [exp(I*(p1 - p2)/2)*sin(t1/2)*cos(t2/2)], 
              [exp(I*(p1 + p2)/2)*sin(t1/2)*sin(t2/2)]] )
#singlet state in lab basis
si = Matrix( [[exp(-I* (p1 + p2)/2)*sin((t2 - t1)/2)], 
              [-cos((t1 - t2)/2)*exp(I*(p2 - p1)/2)], 
              [cos((t1 - t2)/2) *exp(I*(p1 - p2)/2)], 
              [exp(I*(p1 + p2)/2)*sin((t2 - t1)/2)]] )/sqrt(2)

#print tr.subs( [(t1,1.0) , (t2,0.185639) , (p1,0.113828) , (p2,0.121173) , 
#                (x1,0.197626) , (y1,0.122736) , (x2,0.176155) , (y2,0.142152)] ).evalf()

#print si.subs( [(t1,1.0) , (t2,0.185639) , (p1,0.113828) , (p2,0.121173) , 
#                (x1,0.197626) , (y1,0.122736) , (x2,0.176155) , (y2,0.142152)] ).evalf()

#this should be fulfilled
#print simplify(si.adjoint() * si) # 1
#print simplify(tr.adjoint() * tr) # 1
#print simplify(si.adjoint() * tr) # 0
#print simplify(tr.adjoint() * si) # 0

psi_c = (x1+I*y1)*tr + (x2+I*y2)*si

#print psi_c.shape

base = Matrix( [t1,t2,p1,p2,x1,y1,x2,y2] )

# factor 2 ?
psi_r = psi_c.applyfunc(re)
psi_i = psi_c.applyfunc(im)

psi_r_func = lambdify( (t1,t2,p1,p2,x1,y1,x2,y2) , psi_r.tolist() , modules="numpy" )

#embed()

psi = Matrix( [ psi_r , psi_i ] )

jacobian_filename = "jacobian.exp"

try:
    print "Loading the Jacobian from" , jacobian_filename
    jac_file = open( jacobian_filename , "rb" )
    M = pickle.load( jac_file )
except IOError:
    print "IOError"
    print "Calculating the Jacobian..."
    M = simplify(psi.jacobian(base))
    output = open( jacobian_filename , "wb" )
    pickle.dump( M , output )
    output.close()

# the python expresion for the jacobian
jac = lambdify( (t1,t2,p1,p2,x1,y1,x2,y2) , M )

print "Calculating H \psi..."

H_psi = Matrix( [ H * psi_i , -H * psi_r ] )

# the python expression for the rhs
rhs = lambdify( (t1,t2,p1,p2,x1,y1,x2,y2) , H_psi )


"""
print jac( 1.0 , 0.185639 , 0.113828 , 0.121173 , 
           0.197626 , 0.122736 , 0.176155 , 0.142152 )

print rhs( 1.0 , 0.185639 , 0.113828 , 0.121173 , 
           0.197626 , 0.122736 , 0.176155 , 0.142152 )
"""

def dimer( x , t ):
    #print "Call to dimer:" , x
    J = np.array(jac( x[0] , x[1] , x[2] , x[3] , x[4] , x[5] , x[6] , x[7] ))
    H_psi = np.array(rhs( x[0] , x[1] , x[2] , x[3] , x[4] , x[5] , x[6] , x[7] ))
    #print J
    #print H_psi
    #print np.linalg.solve( J , H_psi )
    return np.linalg.solve( J , H_psi ).reshape(8)

x = np.array( [ 1.0 , 0.185639 , 0.113828 , 0.121173 , 
                0.197626 , 0.122736 , 0.176155 , 0.142152 ] )

print dimer( x , 0.0 )

x0 = np.array( [ 1.0, 1.13785, 1.15652, 1.30168, 
                 1.36847, 1.16183, 1.96979, 1.50797 ] )

M = 51

times = np.linspace( 0.0 , 0.6 , M )

traj = sc_odeint( dimer , x0 , times )

#pl.plot( times , traj[:,4]**2+traj[:,5]**2+traj[:,6]**2+traj[:,7]**2 , '--k' )

wf = np.array( psi_r_func( traj[:,0] , traj[:,1] , traj[:,2] , traj[:,3] , 
                           traj[:,4] , traj[:,5] , traj[:,6] , traj[:,7] ) )

wf = wf.reshape( (4,M) )

print wf.shape

for i in xrange(4):
    pl.plot( times , wf[i] )

pl.show()
