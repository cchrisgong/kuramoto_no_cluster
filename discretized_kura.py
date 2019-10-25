'''
asking what does the Euler map integrates up to h^2, we simulate this modified dynamical equation and see if it clusters on roughly the same time scale

'''
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize, fsolve, brentq
import numpy as np
from numpy import cos, sin, pi, exp, arccos, log, arctan2, imag, real, log2, amax, log10, mean, conj
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

def Z_k(phi, k):
	return sum(np.exp(1j * k * phi))/len(phi)
	
def C_k(phi, k):
	sumN = sum(np.exp(1j * k * phi))
	R, Theta = cmath.polar(sumN / len(phi))
	return R, Theta
	
def odehighmodes(phi):
	# output vector = time derivative of input vector phi = dphi
	# Kuramoto equation is: dphi[i] = w[i] + epsilon * R * sin(Theta - phi[i])
	# order parameter/ mean field: R * exp(i * Theta) = sum(j = 1...N) exp(i * phi(j))
	Z_1 = Z_k(phi, 1e0)
	Z_2 = Z_k(phi, 2e0)
	ZSQ = Z_1 ** 2e0
	
	dphi = imag(Z_1 * exp(1j* (phaseshift - phi))) - (epsilon/4e0) * imag(Z_1 * exp(1j * (2e0 * phaseshift - phi)) + Z_1 * conj(Z_2) * exp(1j * phi) - ZSQ * exp(1j * 2e0 * (phaseshift - phi) ) )
	
#	Z_3 = Z_k(phi, 3e0)
#	ZabsSQ = np.abs(Z_1) ** 2e0
	
#	dphi = imag(Z_1 * exp(1j* (phaseshift - phi))) + (epsilon/4e0) * imag(Z_1 * exp(1j * (2e0 * phaseshift - phi)) + Z_1 * conj(Z_2) * exp(1j * phi) - ZSQ * exp(1j * 2e0 * (phaseshift - phi) ) ) + (epsilon**2e0/12e0) * imag(ZabsSQ * Z_2 * exp(1j * (phaseshift - 2e0 * phi)) + ZSQ * conj(Z_2) * exp(1j * phaseshift) - ZSQ * exp(1j * (phaseshift - 2e0 * phi)) - ZabsSQ * exp(-1j * phaseshift) + .5e0 * Z_1 ** 3e0 * exp(-1j * 3e0 * phaseshift) - 1.5e0 * ZSQ * Z_1 * exp(-1j * phaseshift) -  .5e0 * ZSQ * Z_1 * exp(1j * (phaseshift - phi)) + 0.5e0 * (conj(Z_1) ** 2e0) * Z_3 * exp(-1j * (phaseshift - phi)))
	
	return dphi
	
def ode(phi):
	Z_1 = Z_k(phi, 1e0)
	dphi = imag(Z_1 * exp(1j* (phaseshift - phi)))
	return dphi

def Euler(phi, h):
	phi = phi + h * odehighmodes(phi)
	return phi
	
############# integration routine ##############################

def Euler_Heun(func, var, h, sqrth, xi, eta):
	# Stratonovich scheme
	d0 = func(var)
	f0 = var + sqrth * sigma * (xi * sin(var) + eta * cos(var))
#	d1 = odekura(f0, psdelta)
#	var = var + 0.5e0 * h * (d1 + d0) + 0.5e0 * sqrth * sigma * (xi * (sin(var) + sin(f0)) + eta * (cos(f0) + cos(var)))
	var = var + h * d0 + 0.5e0 * sqrth * sigma * (xi * (sin(var) + sin(f0)) + eta * (cos(f0) + cos(var)))	
	return var
	
########################## Integration loops #########################################################################	
def integration(datfile, phiin, algtag):
	f = open(datfile, 'w')
	phi = phiin
	for i in range(Nint):
		t = (i+1) * h
		
		if algtag == 0:
			phi = Euler(phi, h)
			
		if algtag == 1:
			xi = np.random.normal(0, 1)
			eta = np.random.normal(0, 1)
			phi = Euler_Heun(ode, phi, h, sqrth, xi, eta)
#			phi = Euler_Heun(odehighmodes, phi, h, sqrth, xi, eta)			
		
		if i % int(10e0/h) == 0:
			f.write(str(t) + '\t' + str(C_k(phi, 1e0)[0]) + '\t' + str(C_k(phi, 2e0)[0]) + '\t' + str(C_k(phi, 3e0)[0]) + '\n')
#		if i % int(1e0/h) == 0:
#			print(t, C_k(phi, 1e0)[0], C_k(phi, 2e0)[0])
#		
	f.close()
		
def plotRn(datplotfilelist_total):
	fig = plt.figure(figsize=(10,7))
#	gs = gridspec.GridSpec(1, 3)
#	gs.update(wspace=0.2, hspace=0.2)
	ax1 = fig.add_subplot(111)
#	ax2 = plt.subplot(gs[:, 1])
#	ax3 = plt.subplot(gs[:, 2])
		
	ax1.set_xlabel("Time",fontsize=20)
	ax1.set_xlim(0, tEnd)
	ax1.set_ylim(0, 1)

	plot_list = []

	for j in range(len(datplotfilelist_total)):
		datfilelist = datplotfilelist_total[j]
		R2_lst = []
		for casenum in range(NumofCases):
			tlst, R, R2, R3 = loadtxt(datfilelist[0], unpack = 1, skiprows = 1)

			R2_lst.append(R2)
		if j != 2:
			ax1.plot(tlst[::10], mean(R2_lst, axis=0)[::10], c = color_list[j], linestyle = '-', linewidth = 2, alpha = 0.6, markeredgecolor = 'none', label = 'Original dynamics, h=' + str(hlist2[j]) )
		else:
			ax1.plot(tlst[::10], mean(R2_lst, axis=0)[::10], c = color_list[j], linestyle = '-', linewidth = 2, alpha = 0.6, markeredgecolor = 'none', label = 'Corrected dynamics $\epsilon$=' + str(epsilon) +"\n h = " + str(hlist[0]) )
	ticklabels = ax1.get_yticklabels()
	for label in ticklabels:
		label.set_fontsize(15)
	ticklabels = ax1.get_xticklabels()
	for label in ticklabels:
		label.set_fontsize(15)
							
	ax1.set_ylabel("$R_2$",fontsize=24)
	
#	ax1.legend(loc='upper left', fontsize=14)	
	ax1.legend(loc='lower right', bbox_to_anchor=(.95,.23))

	fig.savefig(datname + "_lin_alpha=" + str(psalpha) + ".jpg", dpi=150)
	plt.close()
		
if __name__ == "__main__":
	color_list = ["blue", "green", "red", "purple", "orange", "brown", "black", "yellow", "magenta", "cyan"]
	
	NumofCases = 7
#	hlist = [2e-2, 1e-2, 2e-3,1e-3]
	hlist = [5e-3]
	hlist2 = [2e-2, 5e-3]
	epsilon = 2e-2
	sigma = 0.1e0
	alphalist = [0.3e0]
	
	datplotfilelist_total = [] 

	psalpha = alphalist[0]
	phaseshift = psalpha * 2e0 *pi
	N = 100e0
#	j = int(sys.argv[1])-4-1
	
	datnames = ["Original_Euler_R2", "highermodes_Euler_R2"]
	for i, datname in enumerate(datnames):
		if i == 0:
			for j in range(len(hlist2)):
				print(j)
				h = hlist2[j]
				sqrth = sqrt(h)
				tBegin = 0e0 # integration time range
				tEnd = 100000e0
				Nint = int((tEnd - tBegin) / h) # number of integration steps			
			#		print(Nint)
				datplotfilelist = []
	
				for casenum in range(NumofCases):
		#			casenum = int(sys.argv[1])-1
#					filepath3 = "phiin" + str(casenum) + ".dat"
			#		phiin = 2e0 * pi * np.random.random(100)
			#		FILE = open(filepath3,'w')
			#		for i in range(len(phiin)-1):
			#			FILE.write(str(phiin[i]) + '\t ')
			#		FILE.write(str(phiin[-1]))
			#		FILE.close()
			#			
#					phiin = loadtxt(filepath3, unpack=True)
	
					#######################
				
					datfile = datname + "_run" + str(casenum) + "_epsilon" + str(epsilon) + "_h=" + str(h) + "_alpha=" + str(psalpha) + ".dat"
					phifile = datname + "_run" + str(casenum) + "_epsilon" + str(epsilon) + "_h=" + str(h) + "_alpha=" + str(psalpha) + "_phi.dat"
				
					#######################
		#			integration(datfile, phiin, 1)
	
					datplotfilelist.append(datfile)
				datplotfilelist_total.append(datplotfilelist)
		else:
			for j in range(len(hlist)):
				h = hlist[j]
				sqrth = sqrt(h)
				tBegin = 0e0 # integration time range
				tEnd = 100000e0
				Nint = int((tEnd - tBegin) / h) # number of integration steps			
			#		print(Nint)
				datplotfilelist = []
	
				for casenum in range(NumofCases):
					#######################
				
					datfile = datname + "_run" + str(casenum) + "_epsilon" + str(epsilon) + "_h=" + str(h) + "_alpha=" + str(psalpha) + ".dat"
					phifile = datname + "_run" + str(casenum) + "_epsilon" + str(epsilon) + "_h=" + str(h) + "_alpha=" + str(psalpha) + "_phi.dat"
				
					#######################
		#			integration(datfile, phiin, 1)
	
					datplotfilelist.append(datfile)
		
			datplotfilelist_total.append(datplotfilelist)
	print(datplotfilelist_total)
	plotRn(datplotfilelist_total)
