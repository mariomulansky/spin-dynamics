from math import *
import numpy as np
import string

from ConvertFormula.language import LanguageMathematica,LanguagePython
from ConvertFormula.parser_text import ParserText
from ConvertFormula.formatter import Formatter

parser = ParserText( LanguageMathematica( ) )
formatter = Formatter( LanguagePython( int2float=True ) )

expr_dict = {}

#load the jacobian from mathematica
filename = "../mathematica/jacobian.mathematica"
try:
    f = open( filename , "r" )
except IOError:
    print "File not found:" , filename
    exit


for i,line in enumerate(f):
    #print line
    line = line.lstrip( ' {' )
    line = line.rstrip( ' }\n' )
    #print line
    expressions = line.split( ',' )
    for j,expr in enumerate(expressions):
        parser.parse_text( expr )
        # TODO: allow for expression optimization here
        #parser.optimize_runtime()
        tokens = parser.result
        py_expr = formatter( tokens )
        expr_dict["jac_expr_%d_%d"%(i,j)] = py_expr

#load the Hpsi from mathematic
filename = "../mathematica/Hpsi.mathematica"
try:
    f = open( filename , "r" )
except IOError:
    print "File not found:" , filename
    exit

for i,line in enumerate(f):
    #print line
    line = line.lstrip( ' {' )
    line = line.rstrip( ' }\n' )
    #print line
    parser.parse_text( line )
    # TODO: allow for expression optimization here
    #parser.optimize_runtime()
    tokens = parser.result
    py_expr = formatter( tokens )
    expr_dict["Hpsi_expr_%d"%(i)] = py_expr


try:
    template_file = open("dimer_funcs.tmpl")
except IOError:
    print "File not found: dimer_funcs.tmpl"
    exit

src = string.Template( template_file.read() )
dest = src.substitute( expr_dict )

try:
    func_file = open("dimer_funcs.py" , 'w')
except IOError:
    print "Can't create file: dimer_funcs.py"

func_file.write( dest )
func_file.close()

import dimer_funcs

x = 0.5*np.ones(8)
print dimer_funcs.dimer_jacobi( x , 0.0 )
