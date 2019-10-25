'''
studying numerical error caused by RK4 integrating a regular Kuramoto-Sakaguchi equation by monitoring the WS constants (complex number with modulus and phase). We can use three ways to determine the constants, 1 and 2nd way at each time step we minimize the potential function to find the WS mean field, then converting the phases to constants, using (1) the subroutine optimization alg. in python and (2) using the explicit derivative method and integrate the mean field using them until finding the steady state of the mean field. 3rd way is to use the Marvel Mirollo Strogatz cross ratio of four complex numbers on a circle. 

In this python script the phase shift dependence of the numerical error is studied

'''

from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize, fsolve, brentq
import numpy as np
from numpy import cos, sin, pi, exp, arccos, log, arctan2, imag, real, log2
from numpy import loadtxt, fabs, std
import matplotlib.pyplot as plt
import cmath
from time import sleep
import os
import sys
from wsroutine import *

def wrap2(phi):
	#returns angle (vector) phi to interval [-pi,pi]
	return np.mod(phi - pi, 2e0 * pi) - pi

def wrap(phi):
	#returns angle (vector) phi to interval [0,2pi]
	return np.mod(phi, 2e0 * pi)

def C_k(phi, k):
	sumN = np.sum(np.exp(1j * k * phi), axis = 1)
	R = np.abs(sumN / N)
	Theta = np.angle(sumN / N)
	return R, Theta
	
def ode(phi, t, phaseshift):
	# output vector = time derivative of input vector phi = dphi
	# Kuramoto equation is: dphi[i] = w[i] + epsilon * R * sin(Theta - phi[i])
	# order parameter/ mean field: R * exp(i * Theta) = sum(j = 1...N) exp(i * phi(j))
#	dphi = -0.4e0*C_k(phi, 1e0)[0] * sin(C_k(phi, 1e0)[1] - phi)
#	dphi = C_k(phi, 1e0)[0] * sin(C_k(phi, 1e0)[1] - phi - delta)
	dphi = C_k(phi, 1e0)[0] * sin(C_k(phi, 1e0)[1] - phi - phaseshift)
	return dphi

############# integration routine ##############################
def RK4(phi, t, h, ps):
	#algorithm function of RungeKutta of 4th order, input the initial vector phi, outputs the new vector phi, h the time step. 
	#phi, k1 to k4 are all N-Vector	
	k1 = h * ode(phi, t, ps)
	k2 = h * ode(phi + k1 / 2e0, t + h / 2e0, ps)
	k3 = h * ode(phi + k2 / 2e0, t + h / 2e0, ps)
	k4 = h * ode(phi + k3, t + h, ps)	
	phi = phi + (k1 + 2e0 * k2 + 2e0 * k3 + k4) / 6e0		
	return phi

def constantconverters(phi, rho, Phi):
	RHS = invmobtrans(phi, Phi, rho)
	thetaks2_k = arctan2(imag(RHS), real(RHS))
	thetaks2_k = thetaks2_k - thetaks2_k[1] #get rid of the common rotation
#	alpha0 = sum(thetaks0) / N
	return thetaks2_k, np.absolute(RHS)

def phaseconverters(thetak, rho, Phi):
	RHS = mobtrans(thetak, Phi, rho)
	phi_k = arctan2(imag(RHS), real(RHS))
	return phi_k

def constantconverters_MMS(phi):
	z_list = exp(1j*phi)
	clst = np.array([])
	for i in range(int(N)):
		inputlst = [z_list[i], z_list[np.mod(i+1,int(N))], z_list[np.mod(i+2, int(N))], z_list[np.mod(i+3,int(N))]]
		c = MMS(inputlst)
		clst = np.append(clst, c)
	psik = np.arctan2(imag(clst), real(clst))
	constant_mod = np.absolute(clst)
	return psik, constant_mod
					
###############################  Initial Condition Conversion by minimizing Lyapunov function ####################################
#################################################################################################################################		
########################## Integration loops #########################################################################	
def steadystate(datfile, phi_datfile, phiin, rho0, Phi0, h, ps):
	f = open(datfile, 'w')
	f2 = open(phi_datfile, 'w')
	phi = phiin
	for phii in phi:
		f2.write(str(phii) + '\t')
	f2.write('\n')
#	psi_lst, mod_lst = constantconverters(phi, rho0, Phi0) #converting using minimization method
	psi_lst, mod_lst = constantconverters_MMS(phi) #converting using Mirollo Marvel Strogatz formula
#	f.write("time" + "\t" + "rho" + "\t" + "Phi" + "\t" + "error_phase" + "\t" + "error_modulus" + "\n")
	f.write("time" + "\t" + "error_phase" + "\t" + "error_modulus" + "\n")	
#	f.write(str(0) + "\t" + str(rho0) + "\t" + str(Phi0) + "\t" + str(0) + "\t" + str(0) + "\n")
	f.write(str(0) + "\t" + str(0) + "\t" + str(0) + "\n")	
	for i in range(Nint):
		t = (i+1) * h
		phi2 = RK4(phi, t, h, ps)

		if i % int(1e0/h) == 0:
#			#########################
#			[rho3, Phi3], imax = mini0(phi2) #minimization using explicit derivative 
#			if imax == 3000000:
#				break
#			[rho3, Phi3] = mini(Uf, phi2) #minimization using subroutine

			for phii in phi2:
				f2.write(str(phii) + '\t')
			f2.write('\n')

			psiks, mods = constantconverters_MMS(phi2)
#			psiks, mods = constantconverters(phi2, rho3, Phi3)
			Elist_ph = sin(np.absolute(psi_lst - psiks))
			Elist_mod = np.absolute(mod_lst - mods)
			print(Elist_ph)
			maxE_phase = max(Elist_ph)
			maxE_mod = max(Elist_mod)
			print (i*h, maxE_phase, maxE_mod)
			f.write(str(t) + "\t" + str(maxE_phase) +"\t" + str(maxE_mod) + "\n")
#			f.write(str(t) + "\t" + str(rho3) + "\t" + str(Phi3) + "\t" + str(maxE_phase) +"\t" + str(maxE_mod) + "\n")

		phi = phi2
		
	f.close()
	f2.close()	
	
def plot(phi, filename):
	# plotting the phase on a unit circle
	fig = plt.figure(2)
	fig.clf()

	ax = fig.add_subplot(111, aspect='equal', xlim=(-1.1,1.1), ylim=(-1.1,1.1))
	for j in range(int(N)):
		plt.scatter(cos(phi)[j], sin(phi)[j], s = 100, c = 'red')
#		plt.grid()
	circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
	ax.add_patch(circ)
	plt.axis('off')
	plt.show(block=False)
	sleep(1)
#	fig.savefig("Zosc_"+str(filename) +".png", dpi=500, bbox_inches='tight')
#	fig.savefig("Ztheta_"+str(filename) +".png", dpi=500, bbox_inches='tight')
	plt.close()
	
if __name__ == "__main__":	
################################# Random initial conditions #############################
	
	hlist = [1e-2, 1e-1, 1e0]
	alphalist = [0.25,0.3,0.35,0.4]
	
	color_list = ["blue", "green", "red", "purple", "orange", "brown", "black", "yellow", "magenta", "cyan"]
	
#	ax1.set_ylabel("phi",fontsize=18)
	for casenum in range(1):
		filepath3 = "phiin" + str(casenum) + ".dat"
#		FILE = open(filepath3,'w')
#		for i in range(len(phiin)-1):
#			FILE.write(str(phiin[i]) + '\t ')
#		FILE.write(str(phiin[-1]))
#		FILE.close()

		phiin = loadtxt(filepath3, unpack=True)
		N = 10e0	
#		phiin = 2e0 * pi * randomv(10)
		
		datfoldr = 'phaseshift_dependence_dat'
		try:
			os.makedirs(datfoldr)
		except OSError:
			pass

		os.chdir("./" + datfoldr)

		fig = plt.figure(figsize=(10,7))
		ax1 = fig.add_subplot(111)
		ax1.set_xlabel("Time",fontsize=18)
		
		for j in range(len(hlist)):
		
			h = hlist[j]
			tBegin = 0e0 # integration time range
			tEnd = 2000e0
			Nint = int((tEnd - tBegin) / h) # number of integration steps				
			for k in range(len(alphalist)):
				psalpha = alphalist[k]
				phaseshift = psalpha * 2e0 *pi
				
				#######################
	#			datname = "errorphasemod_RK4_nonoise_derivative"
				datname = "errorphasemod_RK4_nonoise_MMS_phaseshift"

				datfile = datname + "_h=" + str(h) + "_alpha=" + str(psalpha) + ".dat"
				phifile = datname + "_h=" + str(h) + "_alpha=" + str(psalpha) + "_phi.dat"

				#######################
				[rho0, Phi0],imax0 = mini0(phiin) #first convert initial condition using derivative minimization 
	#			[rho0, Phi0] = mini(Uf, phiin) #first convert initial condition using subroutine

#				steadystate(datfile, phifile, phiin, rho0, Phi0, h, phaseshift)
	#			
#				tlst, rho, Phi, Elst_phase, Elst_mod = loadtxt(datfile, unpack = 1, skiprows = 1)
#				rho = -rho
				tlst, Elst_phase, Elst_mod = loadtxt(datfile, unpack = 1, skiprows = 1)
	#			tlst, Elst = loadtxt(datfile, unpack = 1, skiprows = 1)

				phis = loadtxt(phifile, unpack = 0, skiprows = 0)
#				phis = np.transpose(phis)
				ax1.plot(tlst, C_k(phis, 4e0)[0], c = color_list[j], linestyle = '-', linewidth = 2, label = "h=" + str(h) + " $\\alpha$=" +str(psalpha), markeredgecolor = 'none', alpha = 0.1*k)
				
				
	datname = "R4_RK4_nonoise_MMS_phaseshift"
	ax1.set_ylabel("R4",fontsize=18)
#	datname = "errormod_RK4_nonoise_MMS_phaseshift"
#	ax1.set_ylabel("Error_mod",fontsize=18)	

	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))	

#	ax1.set_yscale('log')
	ax1.set_xscale('log')
#	ax1.set_xlim(0, 100)
	fig.savefig(datname + ".png", dpi=500)
	plt.close()
