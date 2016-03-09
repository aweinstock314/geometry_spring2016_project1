#!/usr/bin/python

import flask
from flask import request
from sympy import *
import json

DEBUG=True

# sympy initilization
x,y,z,t = symbols('x,y,z,t', real=True)

# flask initilization
app = flask.Flask(__name__)
app.config.from_object(__name__)

equations = [(cos(t), sin(t), t),
             (cos(t), sin(t), t)]

# Hopefully tiffany will have this part done
equation = (cos(t), sin(t), t)

def tdiff(r):
    return ((diff(r[0],t), diff(r[1],t), diff(r[2],t)))

def frange(counter, end, jump):
  while counter < end:
    yield counter
    counter += jump

def findPoint(curve, time):
    return [float(simplify(curve[0].subs(t, time))),
            float(simplify(curve[1].subs(t, time))),
            float(simplify(curve[2].subs(t, time)))]

@app.route("/points")
def getPoints():
    #/poitns?ID=<curveNum>,order=<derivativeNum>,tstart<starttime>,tend=<end_time>,dt=<deltatime>
    curveNum = int(request.args.get('ID'))
    order    = int(request.args.get('order'))
    tstart   = float(request.args.get('tstart'))
    tend     = float(request.args.get('tend'))
    dt       = float(request.args.get('dt'))

    # TODO: choose curve number

    curve = equation
    for i in range(order):
        curve = tdiff(curve)
    points = [findPoint(curve, time) for time in frange(tstart, tend, dt)]
    print(points)
    return json.dumps(points)

@app.route("/alldata")
def getData():
    tstart   = float(request.args.get('tstart'))
    tend     = float(request.args.get('tend'))
    dt       = float(request.args.get('dt'))


    # return json.dumps([solve(time) for time in range(tstart, tend, dt)])

if __name__ == "__main__":
    app.run(port=1337)
