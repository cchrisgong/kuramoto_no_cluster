'''
simulating WS equations with noise (one noise term, large frequency) imitated by deterministic chaos, investigating at repulsive regime using Euler-Maruyama scheme (because RK4 would be the wrong scheme). 

This should help us testing step size dependence in stochastic intergration. We found the clustering from Gil paper is due to numerical effect from integration, the larger the step sizes, the larger the numerical errors, (Euler-Maruyama has an integration error of tau^{1/2}) which means the (erroneous) cluster state should be reached faster in fewer integration steps. Then this means if we use a better integration, e.g. RK4 that preserves the constants very well, we should be able to see that the clusters do not form for similar stochastic terms

'''

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, exp, arccos, sqrt, log, loadtxt, fabs, std, arctan2,  real, imag, mean, random, amax, amin, median
import cmath
from time import sleep
import os
import sys
from wsroutine import *
import random
from matplotlib import gridspec

def lorenz(var):
	"first equation of Lorenz system"
	x,y,z = var[0,:],var[1,:],var[2,:]
	dx = sigma_lorenz * (y-x)
	dy = x * (rho - z) - y
	dz = x * y - beta * z
	return np.array([dx, dy, dz])

def odekura_RK4(phi, arg2):
	[xi_lorenz_t, eta_lorenz_t, psdelta] = arg2
	[R, Theta] = C_k(phi, 1e0)
	dphi = omega + R * sin(Theta - phi + psdelta) + xi_lorenz_t * sin(phi) + eta_lorenz_t * cos(phi)
	return dphi

def wrap2(phi):
	#returns angle (vector) phi to interval [-pi,pi]
	return np.mod(phi-pi,2e0*pi)-pi

def wrap(phi):
	#returns angle (vector) phi to interval [0,2pi]
	return np.mod(phi, 2e0*pi)

def C_k(phi, k):
	sumN = sum(np.exp(1j * k * phi))
	R, Theta = cmath.polar(sumN / (N))
	return R, Theta

def RK4(func, var, h):
	#algorithm function of RungeKutta of 4th order, input the initial vector phi, outputs the new vector phi, h the time step. 
	#phi, k1 to k4 are all N-Vector
	k1 = h * func(var)
	k2 = h * func(var)
	k3 = h * func(var + k2 / 2e0)
	k4 = h * func(var+ k3)
	var = var + (k1 + 2e0 * k2 + 2e0 * k3 + k4) / 6e0
	return var
	
def RK4warg(func, h, var, arg):
	#algorithm function of RungeKutta of 4th order, input the initial vector var, outputs the new vector var, h the time step. 
	#phi, k1 to k4 are all N-Vector
	k1 = h * func(var, arg)
	k2 = h * func(var + k1 / 2e0, arg)
	k3 = h * func(var + k2 / 2e0, arg)
	k4 = h * func(var + k3, arg)
	var = var + (k1 + 2e0 * k2 + 2e0 * k3 + k4) / 6e0
	return var
		
def constantconverters(phi, rho, Phi):
	RHS = invmobtrans(phi, Phi, rho)
	thetaks2_k = arctan2(imag(RHS), real(RHS))
	thetaks2_k = thetaks2_k - thetaks2_k[1] #get rid of the common rotation
#	alpha0 = sum(thetaks0) / N
	return thetaks2_k, np.absolute(RHS)

def constantconverters_MMS(phi):
	clst = np.array([])
	for i in range(int(N-3)):
		inputlst = [phi[i], phi[i+1], phi[i+2], phi[i+3]]
		c = MMS(inputlst)
		clst = np.append(clst, c)
	return clst
	
def autocorr(x):
	result = np.correlate(x, x, mode='same')
	return result[int(len(result)/2):]

def generatelorenznoise(noisedatfile):
	xyz = np.random.uniform(-1.,1.,(3,M))

	f = open(noisedatfile, 'w')
	f.write('time' + '\t' + 'noise' + '\n')

	for i in range(int(100/0.01)): #pass transient 
		xyz = RK4(lorenz, xyz, h)# lorenz integration
	f.write(str(0) + '\t' + str(np.mean(xyz[1,:])) +'\n') 

	for i in range(1,int((T_long-100)/h)): #skip transient
		xyz = RK4(lorenz, xyz, h)
		f.write(str(i * h) + '\t' + str(np.mean(xyz[1,:])) +'\n') #noise is the average of lorenz signal
	f.close()
	
def integration(Nint, phiin, psdelta, datfile, T):
	phi = phiin
	c0 = constantconverters_MMS(phi) #convert phases to constants and mean field
	
	f = open(datfile, 'w')
	
	f.write('time'+ '\t' + 'R' + '\t' + 'R_2' + '\t' + 'psi'+ '\t' + 'psi_error' + '\t' + 'noise'+ '\n')
	f.write(str(0) + '\t' + str(C_k(phi, 1e0)[0]) + '\t' + str(C_k(phi, 2e0)[0]) + '\t' + str(0) + '\t' + str(0)  + '\t' + str(0) + '\t' + str(0) + '\t' + str(sigma*noise[0]) + '\t' + str(sigma*noise2[0]) + '\n')
	
	for i in range(1,int((T_long-100)/h)): #skip transient
		xi_t = sigma * noise[i] #normalization from lorentz "noise" to gaussian noise
		eta_t = sigma * noise2[i]
		phi = RK4warg(odekura_RK4, h, phi, [xi_t, eta_t, psdelta])
		if i % int(10e0/h) == 0:
			time = i * h
#			print (time, "mean", xi_t, eta_t)
#			psiks, mods = constantconverters(phi, rho3, Phi3)
			c_lst = constantconverters_MMS(phi)
			Elist = np.absolute(c0 - c_lst)
			maxE = amax(Elist)
			minE = amin(Elist)
			medE = median(Elist)
			print (i*h, maxE)
			f.write(str(time)+ '\t' + str(C_k(phi, 1e0)[0]) + '\t' + str(C_k(phi, 2e0)[0]) + '\t' + str(Elist[0])+ '\t' + str(maxE) + "\t" + str(minE) +"\t" + str(medE) + '\t' + str(xi_t) + '\t' + str(eta_t)  + '\n')
	f.close()

def plotkura(phi, time):
	# plotting the phase on a unit circle
	fig = plt.figure(2)
	fig.clf()
	ax = fig.add_subplot(111, aspect='equal', xlim=(-1.1,1.1), ylim=(-1.1,1.1))
	circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
	ax.add_patch(circ)
	for j in range(len(phi)):
		plt.scatter(cosv(phi)[j], sinv(phi)[j])
	plt.grid()
	#plt.show(block=False)'autocorr_lorenz.png'
	#sleep(1)
	fig.savefig(os.path.join(os.getcwd(), 'case'+str(casenum), str(time)+"kura.png"), dpi=500)
	plt.close()

def plotevol(datplotfile_list, figpath, rescaletag, column_tag, T):
	fig = plt.figure(figsize=(11,7))
	gs = gridspec.GridSpec(1, 2, width_ratios=[12, 1]) 
	ax1 = plt.subplot(gs[0])
	if rescaletag == 0:
		ax1.set_xlabel("Time",fontsize=18)
	if rescaletag == 1:
		ax1.set_xlabel("$T/h^{\\alpha}$ $(\\alpha = $" + str(round(a,3)) + ")", fontsize=18)
	if rescaletag == 2:
		ax1.set_xlabel("$T/T^{*}$",fontsize=18)
	ax1.set_ylabel(column_label_list[column_tag],fontsize=18)
	
	ax1.grid(True,linestyle= '-',which='major',color= '0.75')
	ax1.grid(True,linestyle= '-',which='minor',color= '0.75')
	ax1.grid(True, which='both')
	ax1.minorticks_on()	
	
	ax1.set_xlim(0, T-100)
	plot_list = []
#	if column_tag == 3 or column_tag == 4:
#		ax1.set_yscale('log')
	for casenum in range(NumofCases):
		plot_list.append([])
		datplotfile = datplotfile_list[casenum]
		for j in range(len(alpha_list)):
			alpha = alpha_list[j]
			time, R1, R2, E0, Emax, Emin, Emed, noise1, noise2 = loadtxt(datplotfile[j], unpack = 1, skiprows = 1)
			data = [time, R1, R2, E0, Emax, Emin, Emed, noise1, noise2 ]
			plot_list[casenum].append(data[column_tag])
	plot_list = list(map(list, 	zip(*plot_list)))
	for j in range(len(alpha_list)):
		alpha = alpha_list[j]
		ax1.plot(data[0], mean(plot_list[j], axis=0), c = color_list[j], marker = marker_list[j], markersize=1.5, linestyle = '-', linewidth = 2, label = "$\\alpha$=" + str(alpha), markeredgecolor = 'none')
		if column_tag == 7:
			break
	if column_tag != 7:
		legend1 = ax1.legend(loc="upper left", shadow=True, bbox_to_anchor=(1, 1))
	fig.savefig(figpath, dpi=500)
	plt.close()

def plotautocorr(noisedatfile, plotpath):
	time, noise = loadtxt(noisedatfile, unpack = 1, skiprows = 1)
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	lorauto = autocorr(noise)
	ax1.set_xlim(0, 10)
	plt.plot(time[:len(lorauto)],lorauto)
	fig.savefig(plotpath, dpi=500)
	plt.close()
	
if __name__ == "__main__":
#	Lorenz parameter
	sigma_lorenz = 10
	beta = 8e0/3e0
	rho = 28
	
	#h_list = [1e-4, 1e-5, 1e-6] # integration step size	
	h = 1e-3 #kuramoto integration step size
	
	marker_list = ["v", "o", "^", "s", "8", "D", "+", ".", "^", "."]
	color_list = ["blue", "green", "red", "purple", "orange", "brown", "black", "yellow", "magenta", "cyan"]
	column_label_list  = ["time", "R", "$R_{2}$","E0","Emax", "Emin", "Emed", "noise1", "noise2"]
	figure_label_list  = ["time", "R", "R2", "E0","Emax", "Emin", "Emed", "noise1", "noise2"]
	T_long = 40000e0

	omega = 10e0 #frequency
	sigma = 1e-2 #noise strength
	#alpha = 3e-1 #phase shift parameter
	alpha_list = [2.5e-1, 3e-1, 3.5e-1, 4e-1]
##	alpha_list = [1e-1,2e-1]
#	alpha_list = [2.6e-1, 3e-1, 3.5e-1]
	M = 12 #number of coupled Lorenz, M is arbitrary, in a sense that the more one adds, the better it is an approximation of gaussian
	NumofCases = 1 #for averaging out noise
	
	datplotfilelist = []
	for casenum in range(NumofCases):
		phi0datfile = "phiin" + str(casenum) + ".dat"
		N = 100e0
		#phiin = 2e0 * pi * randomv(N)
		#f2 = open(phi0datfile,'w')
		#for i in range(len(phiin)-1):
			#f2.write(str(phiin[i]) + ', ')
		#f2.write(str(phiin[-1]))
		#f2.close()
		phiin = loadtxt(phi0datfile, delimiter=',', unpack=True)
		
		datplotfilelist.append([])
		noisedatfile1 = 'lorenz_noise_' + str(casenum) + '_h=' + str(h)+ '_M=' + str(M) +'1.dat'
		noisedatfile2 = 'lorenz_noise_' + str(casenum) + '_h=' + str(h)+ '_M=' + str(M) +'2.dat'		
		################################
#		generatelorenznoise(noisedatfile1)
#		generatelorenznoise(noisedatfile2)
#		noise = loadtxt(noisedatfile1, usecols = (1,), unpack = 1, skiprows = 1)
#		noise2 = loadtxt(noisedatfile2, usecols = (1,), unpack = 1, skiprows = 1)	
		for k in range(len(alpha_list)):
			alpha = alpha_list[k]
			psdelta = alpha * 2e0 * pi #phase shift

			Nint = int(T_long/h)

			datfile = 'lorenz_asnoise_' + str(casenum) + '_h=' + str(h)+ '_T=' + str(T_long)+ '_alpha=' + str(alpha) + '_sigma=' + str(sigma) + '_M=' + str(M) + '_omega='+ str(omega) + '_MMS.dat'
			
			datplotfilelist[casenum].append(datfile)
			################################
#			integration(Nint, phiin, psdelta, datfile, T_long)
	for j in [1,2,3,4,5,6]:
		plotevol(datplotfilelist, 'lorenz_asnoise' + figure_label_list[j] + '_h=' + str(h) + '_T=' + str(T_long) + '_sigma=' + str(sigma) +'_M=' + str(M) +'_omega='+ str(omega) + '_alphadep_MMS_real.png', 0, j, T_long)
#	plotautocorr(noisedatfile, 'autocorr_lorenz' + '_h=' + str(h) + '_M=' + str(M) + '.png')
	
