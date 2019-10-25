'''
studying numerical error caused by RK4 integrating a regular Kuramoto-Sakaguchi equation by monitoring the WS constants (complex number with modulus and phase). We can use three ways to determine the constants, 1 and 2nd way at each time step we minimize the potential function to find the WS mean field, then converting the phases to constants, using (1) the subroutine optimization alg. in python and (2) using the explicit derivative method and integrate the mean field using them until finding the steady state of the mean field. 3rd way is to use the Marvel Mirollo Strogatz cross ratio of four complex numbers on a circle. 

'''

from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize, fsolve, brentq
import numpy as np
from numpy import cos, sin, pi, exp, arccos, log, arctan2, imag, real, log2, amax, log10, mean
from numpy import loadtxt, fabs, std
import matplotlib.pyplot as plt
import cmath
from time import sleep
import os
import sys
from wsroutine import *

def wrap2(phi):
	#returns angle (vector) phi to interval [-pi,pi]
	return np.mod(phi-pi,2e0*pi)-pi

def wrap(phi):
	#returns angle (vector) phi to interval [0,2pi]
	return np.mod(phi, 2e0*pi)

def C_k(phi, k):
	sumN = sum(np.exp(1j * k * phi))
	R, Theta = cmath.polar(sumN / len(phi))
	return R, Theta
	
def ode(phi, t):
	# output vector = time derivative of input vector phi = dphi
	# Kuramoto equation is: dphi[i] = w[i] + epsilon * R * sin(Theta - phi[i])
	# order parameter/ mean field: R * exp(i * Theta) = sum(j = 1...N) exp(i * phi(j))
	dphi = 0.2e0 * sin(1.752e0 * t) - 0.4e0 * cos(2.33e0 * t) *(C_k(phi, 1e0)[0]) * sin(C_k(phi, 1e0)[1] - phi)	
	return dphi

def Euler(phi, t, h):
	phi = phi + h * ode(phi, t)
	return phi
	
############# integration routine ##############################
def RK4(phi, t, h):
	#algorithm function of RungeKutta of 4th order, input the initial vector phi, outputs the new vector phi, h the time step. 
	#phi, k1 to k4 are all N-Vector	
	k1 = h * ode(phi, t)
	k2 = h * ode(phi + k1 / 2e0, t + h / 2e0)
	k3 = h * ode(phi + k2 / 2e0, t + h / 2e0)
	k4 = h * ode(phi + k3, t + h)	
	phi = phi + (k1 + 2e0 * k2 + 2e0 * k3 + k4) / 6e0		
	return phi

def constantconverters(phi, rho, Phi):
	RHS = invmobtrans(phi, Phi, rho)
	thetaks2_k = arctan2(imag(RHS), real(RHS))
	thetaks2_k = thetaks2_k - thetaks2_k[1] #get rid of the common rotation
#	alpha0 = sum(thetaks0) / N
	return thetaks2_k

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
	clst = []
	for i in range(int(N-3)):
		inputlst = [phi[i], phi[i+1], phi[i+2], phi[i+3]]
		c = MMS_nodivider(inputlst)
		clst.append(c)
	return clst

def MMS_nodivider(inputlst):
	#cross ratio of four distinct points, according to Marvel Mirollo Strogatz paper section V.A.
	[phi1, phi2, phi3, phi4] = inputlst
	S13 = sin((phi1-phi3)/2e0)
	S24 = sin((phi2-phi4)/2e0)
	S14 = sin((phi1-phi4)/2e0)
	S23 = sin((phi2-phi3)/2e0)
	return [S13, S24, S14, S23]
	
###############################  Initial Condition Conversion by minimizing Lyapunov function ####################################
#################################################################################################################################		
########################## Integration loops #########################################################################	
def steadystate(datfile, phi_datfile, phiin, h, tag):
	f = open(datfile, 'w')
#	f2 = open(phi_datfile, 'w')
	phi = phiin
#	for phii in phi:
#		f2.write(str(phii) + '\t')
#	f2.write('\n')
	if tag == 0:
		psiks0 = constantconverters(phi, rho0, Phi0) #converting using minimization method
		plotpoints = np.asarray(np.arange(800) ** 1.05/h, dtype = int)
	if tag == 1 or tag ==3:
		Slist0 = constantconverters_MMS2(phi) #converting using modified Mirollo Marvel Strogatz formula
	elif tag == 2:
		clist0 = constantconverters_MMS(phi) #converting using modified Mirollo Marvel Strogatz formula
	
	f.write("time" + "\t" + "error_max" + "\n")	

	for i in range(Nint):
		t = (i+1) * h
		if tag < 3:
			phi2 = RK4(phi, t, h)
		else:
			phi2 = Euler(phi, t, h)
		
			#########################
		if tag == 0:
			if t < 1e0:
				[rho3, Phi3], imax = mini0(phi2) #minimization using explicit derivative 
				if imax == 3000000:
					break
				psiks = constantconverters(phi2, rho3, Phi3)
				Elst = abs(psiks - psiks0)
				ind = np.where(Elst > 3)
				Elst[ind[0]] = np.abs(Elst[ind[0]] - pi * 2e0)

				f.write(str(t) + "\t" + str(max(Elst)) + "\n")
			else:
				if i in plotpoints:
					[rho3, Phi3], imax = mini0(phi2) #minimization using explicit derivative 
					if imax == 3000000:
						break
					psiks = constantconverters(phi2, rho3, Phi3)
					Elst = abs(psiks - psiks0)
					ind = np.where(Elst > 3)
					Elst[ind[0]] = np.abs(Elst[ind[0]] - pi * 2e0)
					f.write(str(t) + "\t" + str(max(Elst)) + "\n")
			if i % 10000 == 0:
				print(C_k(psiks, 2e0))	
				print(C_k(phi2, 2e0))		
#			[rho3, Phi3] = mini(Uf, phi2) #minimization using subroutine

#			for phii in phi2:
#				f2.write(str(phii) + '\t')
#			f2.write('\n')

##################################################################
		if tag == 1 or tag == 3:
			Slist = constantconverters_MMS2(phi)
			Elist_nodivider = []
			for j in range(int(N-3)):
				E_nodivider = np.absolute(Slist[j][0]*Slist[j][1]*Slist0[j][2]*Slist0[j][3] - Slist[j][2]*Slist[j][3]*Slist0[j][0]*Slist0[j][1])
				Elist_nodivider.append(E_nodivider)
			maxE_nodivider = amax(Elist_nodivider)
			f.write(str(t) + "\t" + str(maxE_nodivider) + "\n")
		
		elif tag == 2:
			clist = constantconverters_MMS(phi)
			max_E = amax(abs(clist-clist0))
			f.write(str(t) + "\t" + str(max_E) + "\n")	
#			f.write(str(t) + "\t" + str(rho3) + "\t" + str(Phi3) + "\t" + str(maxE_phase) +"\t" + str(maxE_mod) + "\n")
		phi = phi2
		
	f.close()
#	f2.close()	
		
if __name__ == "__main__":
	color_list = ["blue", "green", "red", "purple", "orange", "brown", "black", "yellow", "magenta", "cyan"]
	markers = ['d', 'v','*']
	
################################# Plotting error diffusion vs. h #############################	
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	labels = ['Euler', 'RK4, WS', 'RK4, MMS']
	for j, fil in enumerate(['Euler_endpoints.dat', 'RK4_endpoints_WS.dat', 'RK4_endpoints_MMS.dat']):
		h, Ends = loadtxt(fil, delimiter=' ', unpack=True)
		
		ticklabels = ax1.get_yticklabels()
		for label in ticklabels:
			label.set_fontsize(20)
		ticklabels = ax1.get_xticklabels()
		for label in ticklabels:
			label.set_fontsize(20)
				
		x = log10(h)
		y = Ends
		print(Ends)
		z = np.polyfit(x, y, 1)
		ax1.plot([-4,(log10(h)[-1]-log10(h))[0]],z[0]*np.array([-4,(log10(h)[-1]-log10(h))[0]])+z[1], c = 'black', linestyle = '--', linewidth = 1.5, alpha = 0.3, markeredgecolor = 'none')
		ax1.scatter(log10(h), Ends, c = color_list[j], marker = markers[j], s = 200, alpha = 0.7, label = labels[j] + ', slope = ' + str(round(z[0],2)), edgecolors='none')
		
	ax1.set_xlabel("$\mathrm{log}_{10}(h)$",fontsize=22)
	legend1 = ax1.legend(loc="lower right", shadow=True, fontsize = 17)

	ax1.set_ylabel("$\mathrm{log}_{10}(\mathrm{Err})$",fontsize=22)
	ax1.set_xlim(-3.2, 0.2)
	ax1.set_ylim(-12,-1)
	fig.savefig("h_endError.jpg", dpi=500)
	plt.close()
#	sys.exit()	
	
################################# Random initial conditions #############################
	for tag in [1,3]:
		hlist = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0]
#		hlist = [5e-2, 1e-1, 2e-1, 5e-1]
#		hlist = [1e-1]
		
		
		fig = plt.figure(figsize=(10,7))
		ax1 = fig.add_subplot(111)	
		if tag == 3:
			hlist = [1e-3, 2e-3, 5e-3, 1e-2]
		
	#	ax1.set_ylabel("Phi",fontsize=18)
	#	ax1.set_ylabel("phi",fontsize=18)
	#	ax1.set_ylabel("rho",fontsize=18)
		for j in range(len(hlist)):
			Elst_l = []
			tlst_l = []
			h = hlist[j]
			for casenum in range(8):
#			casenum = int(sys.argv[1])-1	
				tBegin = 0e0 # integration time range
				tEnd = 1000e0

				Nint = int((tEnd - tBegin) / h) # number of integration steps		

				phi0datfile = "phiin" + str(casenum) + ".dat"

				N = 10e0
		#		if casenum > 5:
		#			phiin = 2e0 * np.pi * np.random.rand(int(N))
		#			f2 = open(phi0datfile,'w')
		#			for i in range(len(phiin) - 1):
		#				f2.write(str(phiin[i]) + ', ')
		#			f2.write(str(phiin[-1]))
		#			f2.close()
		#					
				phiin = loadtxt(phi0datfile, delimiter=',', unpack=True)

				#######################
				if tag == 0:
					datname = "error_RK4_nonoise_derivative"
					[rho0, Phi0],imax0 = mini0(phiin) #first convert initial condition using derivative minimization 
		####		[rho0, Phi0] = mini(Uf, phiin) #first convert initial condition using subroutine				

				elif tag == 1:
					datname = "error_RK4_nonoise_MMS_nodivider"

				elif tag ==2:
					datname = "error_RK4_nonoise_MMS"

				else:
					datname = "error_Euler_nonoise_nodivider"

				datfile = datname + "_h=" + str(h) + "_" +  str(casenum) + ".dat"
				phifile = datname + "_h=" + str(h) + "_" + str(casenum) + "_phi.dat"

				#######################

#				steadystate(datfile, phifile, phiin, h, tag)
			
				if tag ==0:
					tlst, Elst = loadtxt(datfile, unpack = 1, skiprows = 1)
				else:
					tlst, Elst = loadtxt(datfile, unpack = 1, skiprows = 2)	
	#			phis = loadtxt(phifile, unpack = 0, skiprows = 0)
	#			phis = np.transpose(phis)

				Elst_l.append(Elst)
		
			if tag != 1 and tag !=3:
				ax1.plot(tlst, np.log10(np.mean(Elst_l, axis = 0)), c = color_list[j], linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
			if tag == 1 or tag ==3:
				ax1.plot(tlst[1:], np.log10(np.mean(Elst_l, axis = 0))[1:], c = color_list[j], linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
				
		#			ax1.plot(tlst, phis[0], c = color_list[j], linestyle = '-', linewidth = 2, markeredgecolor = 'none')
		#			ax1.plot(tlst, phis[1], c = color_list[j], linestyle = '-', linewidth = 2, markeredgecolor = 'none')
		#			ax1.plot(tlst, phis[4], c = color_list[j], linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
			if j == 0:
				ax1.plot(tlst, 1e0*log10(tlst)-6.5, c = 'black', linestyle = '--', linewidth = 3.5, alpha = 0.4, markeredgecolor = 'none')
		ax1.set_xlabel("Time",fontsize=18)
		legend1 = ax1.legend(loc="upper left", shadow=True)
		ax1.set_xscale('log')

		ax1.set_xlim(min(hlist), tEnd)

	
		ticklabels = ax1.get_yticklabels()
		for label in ticklabels:
			label.set_fontsize(18)
		ticklabels = ax1.get_xticklabels()
		for label in ticklabels:
			label.set_fontsize(18)

		if tag == 0:
			ax1.set_ylabel("$\mathrm{log}_{10}(\mathrm{Err}_{\mathrm{WS}})$",fontsize=22)				
		if tag == 1:
			ax1.set_ylim(-16, -1)
			ax1.set_ylabel("$\mathrm{log}_{10}(\mathrm{Err}_{\mathrm{MMS}})$",fontsize=22)			
		elif tag == 2:
			ax1.set_ylim(-12, 1)
			ax1.set_ylabel("$\mathrm{log}_{10}(\mathrm{Err}_{\mathrm{MMS}})$",fontsize=22)		
		elif tag == 3:
			ax1.set_ylim(-9, -1)
			ax1.set_ylabel("$\mathrm{log}_{10}(\mathrm{Err}_{\mathrm{MMS}})$",fontsize=22)
			
		fig.savefig(datname + ".jpg", dpi=500)
		plt.close()

		
