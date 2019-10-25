'''
studying numerical error caused by RK4 integrating a regular Kuramoto-Sakaguchi equation by monitoring the WS constants (complex number with modulus and phase). We can use three ways to determine the constants, 1 and 2nd way at each time step we minimize the potential function to find the WS mean field, then converting the phases to constants, using (1) the subroutine optimization alg. in python and (2) using the explicit derivative method and integrate the mean field using them until finding the steady state of the mean field. 3rd way is to use the Marvel Mirollo Strogatz cross ratio of four complex numbers on a circle. 

In this python script the phase shift dependence of the numerical error is studied

'''
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize, fsolve, brentq
import numpy as np
from numpy import cos, sin, pi, exp, arccos, log, arctan2, imag, real, log2, mean
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
	sumN = np.sum(np.exp(1j * k * phi))
	R = np.abs(sumN / N)
	Theta = np.angle(sumN / N)
	return R, Theta

def C_k2(phi, k):
#	print(phi[0])
	sumN = np.sum(np.exp(1j * k * phi), axis = 1)
	R = np.abs(sumN / N)
	Theta = np.angle(sumN / N)
	return R, Theta
		
def ode(phi, t, phaseshift):
	# output vector = time derivative of input vector phi = dphi
	# Kuramoto equation is: dphi[i] = w[i] + epsilon * R * sin(Theta - phi[i])
	# order parameter/ mean field: R * exp(i * Theta) = sum(j = 1...N) exp(i * phi(j))
	dphi = C_k(phi, 1e0)[0] * sin(C_k(phi, 1e0)[1] - phi - phaseshift)
	return dphi

############# integration routine ##############################
def Euler(phi, t, h, ps):
	phi = phi + h * ode(phi, t, ps)
	return phi
	
def heun(phi, t, h, ps):
	k1 = phi + h * ode(phi, t, ps)
	phi2 = phi + (h/2e0) * (ode(phi, t, ps) + ode(k1, t + h, ps))
	return phi2
	
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
	clst = np.array([])
	for i in range(int(N-3)):
		inputlst = [phi[i], phi[i+1], phi[i+2], phi[i+3]]
		c = MMS(inputlst)
		clst = np.append(clst, c)
	return clst

def constantconverters_MMS2(phi):
	clst = np.array([])
	for i in range(int(N-3)):
		inputlst = [phi[i], phi[i+1], phi[i+2], phi[i+3]]
		c = MMS_nodivider(inputlst)
		clst = np.append(clst, c)
	return clst					
###############################  Initial Condition Conversion by minimizing Lyapunov function ####################################
#################################################################################################################################		
########################## Integration loops #########################################################################	
def integration(datfile, phi_datfile, phiin, h, ps, algtag):
	f = open(datfile, 'w')
	phi = phiin
	for i in range(Nint):
		t = (i+1) * h
		if algtag == 0:
			phi2 = Euler(phi, t, h, ps)
					
		if algtag == 1:
			phi2 = heun(phi, t, h, ps)
		
		if algtag == 2:
			phi2 = RK4(phi, t, h, ps)

		if i % int(2e0/h) == 0:
			f.write(str(t)+ '\t' + str(C_k(phi, 1e0)[0]) + '\t' + str(C_k(phi, 2e0)[0])+ '\t' + str(C_k(phi, 3e0)[0]) +'\n')

		phi = phi2
		
	f.close()
	
def plotkura(phiplot, time):
	# plotting the phase on a unit circle
	fig = plt.figure(2)
#	fig.clf()
	ax = fig.add_subplot(111, aspect='equal', xlim=(-1.1,1.1), ylim=(-1.1,1.1))
	circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
	ax.add_patch(circ)
	for j in range(len(phiplot)):
		plt.scatter(cos(phiplot)[j], sin(phiplot)[j])
#		ax.annotate(str(j), xy=(cos(phi)[j], sin(phi)[j]), textcoords='offset points')
#	plt.grid()
#	plt.show(block=False)
#	sleep(1)
	fig.savefig(os.path.join(os.getcwd(), datfoldr, str(time)+"kura.png"), dpi=500)
	plt.close()

def plotRn(datplotfilelist_total, phiplotfilelist_total):
	fig = plt.figure(figsize=(28,7))
	gs = gridspec.GridSpec(1, 3)
	gs.update(wspace=0.2, hspace=0.2)
	ax1 = plt.subplot(gs[:, :1])
	ax2 = plt.subplot(gs[:, 1])
	ax3 = plt.subplot(gs[:, 2])
		
	ax1.set_xlabel("Time",fontsize=28)
	ax2.set_xlabel("Time",fontsize=28)
	ax3.set_xlabel("Time",fontsize=28)
	ax1.set_xlim(0, 1100)
	ax2.set_xlim(0, 1100)
	ax3.set_xlim(0, 1100)
	ax1.set_ylim(0, 1)
	ax2.set_ylim(0, 1)
	ax3.set_ylim(0, 1)

	plot_list = []

	for j in range(len(hlist)):
		h = hlist[j]
		datfilelist = datplotfilelist_total[j]
		phifilelist = phiplotfilelist_total[j]
		R_lst = []
		R2_lst = []
		R3_lst = []	
		for casenum in range(NumofCases):
			tlst, R, R2, R3 = loadtxt(datfilelist[0], unpack = 1, skiprows = 1)

			R_lst.append(R)
			R2_lst.append(R2)
			R3_lst.append(R3)
#		print(mean(R2_lst, axis=0)[0],mean(R2_lst, axis=0)[-1])
		ax1.plot(tlst, mean(R_lst, axis=0), c = color_list[j], linestyle = '-', linewidth = 2, markeredgecolor = 'none')
		ax2.plot(tlst, mean(R2_lst, axis=0), c = color_list[j], linestyle = '-', linewidth = 2, markeredgecolor = 'none')
		ax3.plot(tlst, mean(R3_lst, axis=0), c = color_list[j], linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
	
	ticklabels = ax1.get_yticklabels()
	for label in ticklabels:
		label.set_fontsize(20)
	ticklabels = ax1.get_xticklabels()
	for label in ticklabels:
		label.set_fontsize(20)
	ticklabels = ax2.get_yticklabels()
	for label in ticklabels:
		label.set_fontsize(20)
	ticklabels = ax2.get_xticklabels()
	for label in ticklabels:
		label.set_fontsize(20)
	ticklabels = ax3.get_yticklabels()
	for label in ticklabels:
		label.set_fontsize(20)
	ticklabels = ax3.get_xticklabels()
	for label in ticklabels:
		label.set_fontsize(20)
							
	ax1.set_ylabel("$R$",fontsize=28)
	ax2.set_ylabel("$R_2$",fontsize=28)
	ax3.set_ylabel("$R_3$",fontsize=28)
	
	datname = "Euler_nonoise"
#	datname = "errormod_RK4_nonoise_MMS_phaseshift"
#	ax1.set_ylabel("Error_mod",fontsize=18)	

#	box = ax1.get_position()
#	ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax3.legend(loc='center left', fontsize=28, bbox_to_anchor=(1., 0.5))	

#	ax1.set_yscale('log')
#	ax1.set_xscale('log')
#	ax1.set_xlim(0, 100)
#	fig.tight_layout()
	fig.savefig(datname + "_lin_alpha=" + str(psalpha) + ".png", dpi=500)
	plt.close()
	
if __name__ == "__main__":	
################################# Random initial conditions #############################
	NumofCases = 10
#	hlist = [1e-1, 5e-2, 2e-2, 1e-2]
	hlist = np.linspace(0.1, 1e0,10)
	alphalist = [0.27]
	color_list = ["cyan", "black", "red", "purple", "orange", "magenta","brown", "green", "blue" ,"yellow"]
	
	datplotfilelist_total = [] 
	phiplotfilelist_total = []
#	ax1.set_ylabel("phi",fontsize=18)
	datfoldr = 'nonoise_R2_dependence_dat'
	try:
		os.makedirs(datfoldr)
	except OSError:
		pass
	psalpha = alphalist[0]
	phaseshift = psalpha * 2e0 *pi
	os.chdir("./" + datfoldr)
	N = 100e0

	for j in range(len(hlist)):
		h = hlist[j]
		tBegin = 0e0 # integration time range
		tEnd = 2000e0
		Nint = int((tEnd - tBegin) / h) # number of integration steps			
#		print(Nint)
		datplotfilelist = []
		phiplotfilelist = []
		
		for casenum in range(NumofCases):
			filepath3 = "phiin" + str(casenum) + ".dat"
#			phiin = 2e0 * pi * np.random.random(100)
#			FILE = open(filepath3,'w')
#			for i in range(len(phiin)-1):
#				FILE.write(str(phiin[i]) + '\t ')
#			FILE.write(str(phiin[-1]))
#			FILE.close()
			
			phiin = loadtxt(filepath3, unpack=True)
			
#			for k in range(len(alphalist)):
			
			#######################
#			datname = "errorphasemod_RK4_nonoise_derivative"
#			datname = "errorphasemod_RK4_nonoise_MMS_phaseshift"
			datname = "errorphasemod_Euler_MMS_averaged"				

			datfile = datname + "_case" + str(casenum) + "_h=" + str(h) + "_alpha=" + str(psalpha) + ".dat"
			phifile = datname + "_case" + str(casenum) + "_h=" + str(h) + "_alpha=" + str(psalpha) + "_phi.dat"

			#######################
#			[rho0, Phi0] = mini(Uf, phiin) #first convert initial condition using subroutine

			integration(datfile, phifile, phiin, h, phaseshift, 0)
#			integration(datfile, phifile, phiin, rho0, Phi0, h, phaseshift, 2)	
			
#			tlst, Elst_phase, Elst_mod = loadtxt(datfile, unpack = 1, skiprows = 1)
#			tlst, Elst = loadtxt(datfile, unpack = 1, skiprows = 1)

#			if rescaletag == 1:
#			tlst = tlst * h ** (-1.)
						
			datplotfilelist.append(datfile)
			phiplotfilelist.append(phifile)
			
		datplotfilelist_total.append(datplotfilelist)
		phiplotfilelist_total.append(phiplotfilelist)
	plotRn(datplotfilelist_total, phiplotfilelist_total)
