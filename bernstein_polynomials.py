#!/usr/bin/env python
from math import factorial
from sympy import *
import IPython
import itertools
import re

def choose(n,k):
    return factorial(n) / (factorial(k) * factorial(n-k))

def bernstein(dim,x):
    result = 0
    for i in range(dim+1):
        result += choose(dim,i) * ((1-x)**(dim-i)) * (x**i)
    return result

def bezier(dim):
    u, v = symbols('u,v', real=True)
    b = dict()
    result = 0
    for i,j in itertools.product(range(dim),range(dim)):
        b[i,j] = symbols('b%d%d' % (i,j), real=True)
        result += b[i,j] * bernstein(dim,u) * bernstein(dim,v)
    '''
    b = symbols('b', real=True)
    #i, j = symbols('i,j')
    result = b * bernstein(dim,u) * bernstein(dim,v)
    '''
    return u,v,result

def formatter(u,v,phi):
    def strpow(p):
        tmp = str(p)
        tmp = re.sub(r'([0-9]+)', r'\1.0', tmp)
        tmp = re.sub(r'\*\*([0-9]+)\.0', r'.powi(\1)', tmp)
        tmp = re.sub(r'(b[0-9][0-9])\.0', r'\1', tmp)
        return tmp

    #print(str(phi))
    print('// BEGIN code generated by bernstein_polynomials.py')
    print('result.sigma = %s;' % strpow(phi))

    print('')

    print('result.sigma_u = %s;' % strpow(phi.diff(u)))
    print('result.sigma_v = %s;' % strpow(phi.diff(v)))

    print('')

    print('result.sigma_uu = %s;' % strpow(phi.diff(u).diff(u)))
    print('result.sigma_uv = %s;' % strpow(phi.diff(u).diff(v)))
    print('result.sigma_vv = %s;' % strpow(phi.diff(v).diff(v)))
    print('// END code generated by bernstein_polynomials.py')

#IPython.embed()
formatter(*bezier(3))