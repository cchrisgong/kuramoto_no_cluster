import numpy as np
from numpy import cos, sin, pi, exp, arccos, sqrt, log, loadtxt, fabs, std
import sys

def ode_vdP(var):
	return np.array([var[1], mu * (1e0 - var[0] ** 2e0) * var[1] - var[0] - eps * ((1e0 / N * sum(var[1]))**2e0 * var[0] ) ])
	
def Euler_Maru_vdP_additiv_noise(var, h, sqrth, xi, eta):
	var = var + h * ode_vdP(var) + sqrth * sigma * np.array([xi * np.ones(N), np.zeros(N)])
	return var
	
def sRK4(var, h, sigma, eta, psi):
	#second order
	K1 = var
	K2 = var + TwoThirds * h * ode_vdP(K1) + TwoThirds * eta * sigma
	K3 = var + ThreeHalves * h * ode_vdP(K1) - OneThird * h * ode_vdP(K2) + OneHalf * eta * sigma + OneSixth * eta * sigma - TwoThirds * psi * sigma
	K4 = var + SevenSixth * h * ode_vdP(K1) - OneHalf * eta * sigma + OneHalf * eta * sigma + OneSixth * psi * sigma + OneHalf * psi * sigma
	var = var + h * (OneFourth * ode_vdP(K1) + ThreeFourth * ode_vdP(K2) - ThreeFourth * ode_vdP(K3) + ThreeFourth * ode_vdP(K4)) + eta * (- OneHalf * sigma + ThreeHalves * sigma - ThreeFourth * sigma + ThreeFourth * sigma) + psi * (ThreeHalves * sigma - ThreeHalves * sigma)
	return var
	
def integration(h, sqrth, sigma, Nint, xin, yin):
	var = np.array([xin, yin])

	for i in range(int(Nint)):
		time = i * h
		u = np.random.normal(0, 1)
		v = np.random.normal(0, 1)
		eta = u * sqrth
		psi = sqrth * (u/2e0 + v/c)
		var2 = sRK4(var, h, sigma, eta, psi)	
		
if __name__ == "__main__":
	Ncasenum = 1
	h = 1e-3

	T = 10000e0
	sqrth = sqrt(h)
	Nint = int(T/h)	
	
	mu = 1. #van der pol nonlinearity parameter
	b = 0.01  #coupling strength (repulsiveness degree) # b cannot be too big because we want to use mu to control the shape of the limit cycle, when b is large it means when the repulsive coupling is large it distorts the limit cycle too much

	sigma_list = [0.02]

	c = 2e0 * sqrt(3e0) #sRK4 parameter
	TwoThirds = 2e0 / 3e0
	OneThird = 1e0 / 3e0
	ThreeHalves = 3e0 / 2e0
	OneHalf = 1e0 / 2e0
	OneSixth = 1e0 / 6e0
	SevenSixth = 7e0 / 6e0
	ThreeFourth = 3e0 / 4e0
	OneFourth = 1e0 / 4e0		
	
	for l in range(len(sigma_list)):
		xin = np.random.uniform(-1.,1.,size = N) * 5.
		yin = np.random.uniform(-1.,1.,size = N) * 5.		
		integration(h, sqrth, sigma, Nint, xin, yin)
		
		
