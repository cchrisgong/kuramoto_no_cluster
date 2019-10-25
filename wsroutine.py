#from Vector import *
import numpy as np
from numpy import cos, sin, pi, exp, arccos, sqrt, log, loadtxt, fabs, std
import sys
from scipy.optimize import minimize
import cmath

h_min = 1e-2
########################################### integration routine ##############################
###############################  Initial Condition Conversion by minimizing Lyapunov function ####################################
def mobtrans(thetak, Phi, rho):
	return (rho * exp(1j * Phi) + exp(1j * thetak)) / ( 1e0 + rho * exp( 1j * (thetak - Phi) ) )

def invmobtrans(phi, Phi, rho):
	return (rho * exp(1j * Phi) - exp(1j * phi)) / (rho * exp(1j * (phi - Phi)) - 1e0)

#################################################################################################################################
	
def Uf(x, phiin_t):
	#x = [rho, Phi]
	#Potential H = (1./N) * Sum[ log[ (1 - 2 * rho * cos(phi_j - Phi) + rho ^2) / (1 - rho^2) ] ]
	[x1, x2] = x
	U = sum(log( ( 1e0 - 2e0 * x1 * cos(phiin_t - x2) + x1 ** 2e0) / (1.000000000001e0 - x1 ** 2e0) ))
	return U
	
def mini(fun, phiin_t):
	R, Theta = C_k(phiin_t, 1e0, N)
	#want to find the coordinate (rho, Phi) that maximize the lyapunov function
	#initial guess for (rho, Phi), as suggested by the SW paper, is taken to be (R, phi + pi) and (R, phi - pi) 
	# i.e. the opposite of the order parameter
	rhoguess = R
	Phi_guess = Theta + pi
	#res1 = minimize(fun, (rhoguess, Phi_guess), method='SLSQP', bounds=((1e-3, 1e0 - 1e-3), (0, 2e0 * pi)), args = (phiin_t, N))
	res1 = minimize(fun, (rhoguess, Phi_guess), method='BFGS', args = (phiin_t), tol=1e-13)
	return res1.x

def Ufderiv1(rho, arg):
	[phi, Phi] = arg
	rhosq = rho ** 2e0
	unten = 1e0 - 2e0 * rho * cos(phi - Phi) + rhosq
	dUdrho = sum(( (1e0 + rhosq) * cos(phi - Phi) - 2e0 * rho) /unten)
	dUdrho /= 1e0 - rhosq
	return dUdrho

def Ufderiv2(Phi, arg):
	[phi, rho] = arg
	rhosq = rho ** 2e0
	unten = 1e0 - 2e0 * rho * cos(phi - Phi) + rhosq
	dUdPhi = sum(sin(phi - Phi) / unten)
	dUdPhi = dUdPhi * rho
	return dUdPhi

# mini0 and its subroutine minimize_ss uses explicit partial derivatives of the potential function to find minimum of the potential, should be more accurate than python package
def minimize_ss(rhoguess, Phiguess, phi):
	rho = 1.1e0# rhoguess #rho_old
	Phi = Phiguess
	rho2 = rhoguess#1.1e0   #rho_new
	i = 0
	while (fabs(rho2-rho) > 1e-14 and i<3000000):
		rho = np.copy(rho2)
		rho2 = RK4warg(Ufderiv1, rho, [phi, Phi])
		Phi = RK4warg(Ufderiv2, Phi, [phi, rho])
		i+=1
		
#	print(i)
	if i == 3000000:
		print("Warning: minimization algorithm reaches maximum steps!")
	return [rho, Phi], i
   	
def mini0(phi):
	R, Theta = C_k(phi, 1e0)
	rhoguess = R
	Phiguess = Theta + pi
	res, i = minimize_ss(rhoguess, Phiguess, phi)
	return res, i

def MMS(inputlst):
	#cross ratio of four distinct points, according to Marvel Mirollo Strogatz paper section V.A.
	[phi1, phi2, phi3, phi4] = inputlst
	S13 = sin((phi1-phi3)/2.)
	S24 = sin((phi2-phi4)/2.)
	S14 = sin((phi1-phi4)/2.)
	S23 = sin((phi2-phi3)/2.)	
	return ((S13 * S24) / (S14 * S23))

def C_k(phi, k):
	sumN = sum(np.exp(1j * k * phi))
	R, Theta = cmath.polar(sumN / len(phi))
	return R, Theta
	
def MMS_nodivider(inputlst):
	#cross ratio of four distinct points, according to Marvel Mirollo Strogatz paper section V.A.
	[phi1, phi2, phi3, phi4] = inputlst
	S13 = sin((phi1-phi3)/2.)
	S24 = sin((phi2-phi4)/2.)
	S14 = sin((phi1-phi4)/2.)
	S23 = sin((phi2-phi3)/2.)	
	return [S13, S24, S14, S23]
#################################################################################################################################
def shortcuts(N, rho, Phi, thetaks):
	rhosq = rho ** 2e0
	tworho = 2e0 * rho 
	cosvA = cosv( thetaks - Phi * onesv(N))
	sinvA = sinv( thetaks - Phi * onesv(N))
	denom = 1e0 + 2e0 * rho * cosvA + rhosq
	b1 = (1e0 - rhosq) / 2e0
	b2 = (1e0 + rhosq) / tworho
	return tworho, rhosq, cosvA, sinvA, denom, b1, b2
	
###############################  Watanabe Strogatz Dynamical Equation for rho, Phi (with noise) ###############################
def Euler_Maru_WS_rho(var, args): #Euler-Maru for integrating WS Phi of common noise sys.
	[invN, dt, sqrt_dt, sigma, rho, Phi, xi_t, eta_t, psdelta, tworho, rhosq, cosvA, sinvA, denom, b1, b2] = args
	drho = b1 * invN * sum( ( 2e0 * rho + (1e0 + rhosq) * cosvA) / denom ) * cos(psdelta) + ( ((1e0 - rhosq ) ** 2e0) / 2e0 ) * invN * sum( sinvA / denom) * sin(psdelta)
	g = b1 * sigma * sqrt_dt 
	var = var + dt * drho - g * xi_t * cos(Phi) - g * eta_t * sin(Phi)
	return var

def Euler_Maru_WS_Phi(var, args): #Euler-Maru for integrating WS of common noise sys.
	[invN, dt, sqrt_dt, sigma, rho, Phi, xi_t, eta_t, psdelta, tworho, rhosq, cosvA, sinvA, denom, b1, b2] = args
	dPhi = ( (1e0 - rhosq ** 2e0) / tworho ) * invN * sum( sinvA / denom ) * cos(psdelta) - b2 * invN * sum( ( tworho + (1e0 + rhosq) * cosvA) / denom) * sin(psdelta)
	g = b2 * sigma * sqrt_dt 
	var = var + dt * dPhi + g * xi_t * sin(Phi) - g * eta_t * cos(Phi)
	return var
	
def Euler_Maru_WS_alpha(var, args): #Euler-Maru for integrating WS Phi of common noise sys.
	[invN, dt, sqrt_dt, sigma, rho, Phi, xi_t, eta_t, psdelta, tworho, rhosq, cosvA, sinvA, denom, b1, b2] = args
	dalpha = (1e0 - rhosq) * rho * invN * sum( sinvA/denom) * cos(psdelta) - rho * invN * sum( ( 2e0 * rho + (1e0 + rhosq) * cosvA) / denom) * sin(psdelta)
	g = rho * sigma * sqrt_dt 
	var = var + dt * dalpha + g * xi_t * sin(Phi) - g * eta_t * cos(Phi)
	return var

##############################################################################################################################
def ssInit(psikrand, Nint, tol, N, h):
	# steady state of synchronization
	psi = psikrand
	counter = 0
	for i in range(Nint):
		# calls Runge-Kutta integration method
		psi2 = RK4(psi, N, h)
		counter += 1
		if C_k(psi2, 1e0, N)[0] < tol:
			break
		psi = psi2
		
	if counter != Nint:
		return psi, C_k(psi, 1e0, N)[0], C_k(psi, 1e0, N)[1] #return steady state phi, R, Theta of the incoherent psi_k's
	else:
		return "Desired tolerance not reached with current integration steps!"

def ode(phi, N):
	# output vector = time derivative of input vector phi = dphi
	# Kuramoto equation is: dphi[i] = w[i] - epsilon * R * sin(Theta - phi[i])
	# order parameter/ mean field: R * exp(i * Theta) = sum(j = 1...N) exp(i * phi(j))
	dphi = - C_k(phi, 1e0, N)[0] * sinv(C_k(phi, 1e0, N)[1] - phi) #Z
	return dphi
	
def RK4warg(func, var, arg):
#	algorithm function of RungeKutta of 4th order, input the initial vector var, outputs the new vector var, h the time step. 
#		phi, k1 to k4 are all N-Vector
	k1 = h_min * func(var, arg)
	k2 = h_min * func(var + k1 / 2e0, arg)
	k3 = h_min * func(var + k2 / 2e0, arg)
	k4 = h_min * func(var + k3, arg)
	var = var + (k1 + 2e0 * k2 + 2e0 * k3 + k4) / 6e0
	return var
