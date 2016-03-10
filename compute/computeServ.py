#!/usr/bin/python

import flask
from flask import request
from sympy import *
import json
# sympy initilization
x,y,z,t = symbols('x,y,z,t', real=True)

FCompose = False
# Hopefully tiffany will have this part done
equations = [(10*cos(t/4), 10*sin(t/4), 0),
             (cos(t*4), sin(t*4), 0),
             (0,0,t)]

try:
    import config
except:
    print("Could not find config. Using defaults")

DEBUG=True


# flask initilization
app = flask.Flask(__name__)
app.config.from_object(__name__)

# T = r' Norm
# N = r'' Norm
# B = T x N

def tdiff(r):
    return ((diff(r[0],t), diff(r[1],t), diff(r[2],t)))

def cross(a, b):
    # a2b3 - a3b2,a3b1 - a1b3, a1b2 - a2,b1
    rtrn = [None, None, None]
    rtrn[0] = (a[1]*b[2] - a[2]*b[1])
    rtrn[1] = (a[2]*b[0] - a[0]*b[2])
    rtrn[2] = (a[0]*b[1] - a[1]*b[0])
    return tuple(rtrn)

def dot(a, b):
    return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

def norm(v):
    # Setting these to variables helps solidify them as positive, ensuring sqrt simplifies correctly
    a = v[0]
    b = v[1]
    c = v[2]
    return (sqrt(a**2 + b**2 + c**2))

def frange(counter, end, jump):
  while counter < end:
    yield counter
    counter += jump

def findPoint(curve, time):
    return [float(simplify(curve[0].subs(t, time))),
            float(simplify(curve[1].subs(t, time))),
            float(simplify(curve[2].subs(t, time)))]

def compose(f1, f2):
    return [simplify(eq1 + eq2) for eq1,eq2 in zip(f1,f2)]

@app.route("/points")
def getPoints():
    #/poitns?ID=<curveNum>,order=<derivativeNum>,tstart<starttime>,tend=<end_time>,dt=<deltatime>
    curveNum = int(request.args.get('ID'))
    order    = int(request.args.get('order'))
    tstart   = float(request.args.get('tstart'))
    tend     = float(request.args.get('tend'))
    dt       = float(request.args.get('dt'))

    # # TODO: choose curve number
    # Base case
    mycurve = [0,0,0]
    for equation in equations:
        mycurve = compose(mycurve, equation)

    # curve = equation
    for i in range(order):
        mycurve = tdiff(mycurve)
        pprint(mycurve)

    points = [findPoint(mycurve, time) for time in frange(tstart, tend, dt)]
    return json.dumps(points)

@app.route("/alldata")
def getData():
    tstart   = float(request.args.get('tstart'))
    tend     = float(re0quest.args.get('tend'))
    dt       = float(request.args.get('dt'))


    # return json.dumps([solve(time) for time in range(tstart, tend, dt)])

if __name__ == "__main__":
    app.run(port=1337)
