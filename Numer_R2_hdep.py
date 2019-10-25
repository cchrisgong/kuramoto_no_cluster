'''
simulating WS equations with noise (one noise term, large frequency), investigating at repulsive regime using Euler-Maruyama scheme (because RK4 would be the wrong scheme). 

Testing step size dependence, if the clustering from Gil paper is due to numerical effect from integration, then the larger the step sizes, the larger the numerical errors, (Euler-Maruyama has an integration error of tau^{1/2}) which means the (erroneous) cluster state should be reached faster in fewer integration steps

two noise terms
'''
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, exp, arccos, sqrt, log, loadtxt, fabs, std, arctan2,  real, imag, mean, random, amax, amin, log10
import cmath
import matplotlib.gridspec as gridspec
import os
import sys

def C_k(var, k):
	sumN = sum(np.exp(1j * k * var))
	R, Theta = cmath.polar(sumN / (N))
	return R, Theta
	
def constantconverters_MMS2(phi):
	clst = np.array([])
	for i in range(int(N-3)):
		inputlst = [phi[i], phi[i+1], phi[i+2], phi[i+3]]
		c = MMS_nodivider(inputlst)
		clst = np.append(clst, c)
	return clst

def MMS_nodivider(inputlst):
	#cross ratio of four distinct points, according to Marvel Mirollo Strogatz paper section V.A.
	[phi1, phi2, phi3, phi4] = inputlst
	S13 = sin((phi1-phi3)/2.)
	S24 = sin((phi2-phi4)/2.)
	S14 = sin((phi1-phi4)/2.)
	S23 = sin((phi2-phi3)/2.)	
	return [S13, S24, S14, S23]
								
def integration(h, sqrth, Nint, phiin, psdelta, datfile):
	phi = phiin
	Slist0 = constantconverters_MMS2(phi)
	
	R_list = np.array([])
	R2_list =np.array([])
	R3_list =np.array([])
	Err_list = np.array([])
	for i in range(int(Nint)):
		xi = np.random.normal(0, 1)
		eta = np.random.normal(0, 1)
		
		[R, Theta] = C_k(phi, 1e0)
		d0 = R * sin(Theta - phi + psdelta) 
		f0 = phi + sqrth * sigma * (xi * sin(phi) + eta * cos(phi))
		phi2 = phi + h * d0 + 0.5e0 * sqrth * sigma * (xi * (sin(phi) + sin(f0)) + eta * (cos(f0) + cos(phi)))	
#		phi2 = Euler_Heun(phi, h, sqrth, psdelta, xi, eta)
			
		if i % int(10e0/h) == 0:
			Slist = constantconverters_MMS2(phi)
			Elist_nodivider = np.absolute(Slist[0]*Slist[1]*Slist0[2]*Slist0[3] - Slist[2]*Slist[3]*Slist0[0]*Slist0[1])
			maxE_nodivider = amax(Elist_nodivider)
			
			R_list = np.append(R_list, C_k(phi, 1e0)[0])
			R2_list = np.append(R2_list, C_k(phi, 2e0)[0])
			R3_list = np.append(R3_list, C_k(phi, 3e0)[0])			
			Err_list = np.append(Err_list, maxE_nodivider)
		phi = phi2
	DAT = np.column_stack((R_list, R2_list, R3_list, Err_list))
	np.savetxt(datfile, DAT, delimiter=" ")	

def plotevol2(datplotfilelist_METHOD, figpath, rescaletag, column_tag):
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	if rescaletag == 0:
		ax1.set_xlabel("Time",fontsize=18)
	if rescaletag == 1:
		ax1.set_xlabel("$T/h^{a}$ $(a = $" + str(round(a,3)) + ")", fontsize=18)
	if rescaletag == 2:
		ax1.set_xlabel("$T/T^{*}$",fontsize=18)
	ax1.set_ylabel(column_label_list[column_tag],fontsize=18)
	
	ax1.grid(True,linestyle= '-',which='major',color= '0.75')
	ax1.grid(True,linestyle= '-',which='minor',color= '0.75')
	ax1.grid(True, which='both')
	ax1.minorticks_on()	
	
	if column_tag == 4:
		ax1.set_xscale('log')
#		ax1.set_yscale('log')
		ax1.set_ylim(-15, 0)
	for j in range(len(h_list)):
		datfilelist = datplotfilelist_METHOD[j]
		#print(datfilelist)
		plot_list = []
		for casenum in range(len(datfilelist)):
			h = h_list[j]
			R1, R2, R3, Emax_nodivider = loadtxt(datfilelist[casenum], unpack = 1)
			time = np.linspace(0, T, len(R1))
			if rescaletag == 1:
				time = time * h ** (-a)
				ax1.set_xlim(0, 10000)
			else:
				ax1.set_xlim(0, 17500)
			data = [time, R1, R2, R3, Emax_nodivider]
			plot_list.append(data[column])
		if column_tag < 4:
			ax1.plot(data[0], mean(plot_list, axis=0), c = color_list[j], marker = marker_list[casenum], markersize=1.5, linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
		if column_tag == 4:
			if casenum == 0:
				ax1.plot(data[0], log10(mean(data[column_tag], axis=0)), c = color_list[j], marker = marker_list[casenum], markersize=1.5, linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
			else:
				ax1.plot(data[0], log10(mean(data[column_tag], axis=0)), c = color_list[j], marker = marker_list[casenum], markersize=1.5, linestyle = '-', linewidth = 2)				
	if rescaletag == 0:
		legend1 = ax1.legend(loc=9, shadow=True, bbox_to_anchor=(0.9, .25))
	fig.savefig(figpath, dpi=500)
	plt.close()

def plotevol3(datplotfilelist_METHOD, figpath, rescaletag, column_tag):
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	if rescaletag == 0:
		ax1.set_xlabel("Time",fontsize=18)
	if rescaletag == 1:
		ax1.set_xlabel("$T/h^{a}$ $(a = $" + str(round(a,3)) + ")", fontsize=18)
	ax1.set_ylabel(column_label_list[column_tag],fontsize=18)
	
	ax1.grid(True,linestyle= '-',which='major',color= '0.75')
	ax1.grid(True,linestyle= '-',which='minor',color= '0.75')
	ax1.grid(True, which='both')
	ax1.minorticks_on()
	
	plot_list = []
	if column_tag == 4:
		ax1.set_xscale('log')
#		ax1.set_yscale('log')
#		ax1.set_ylim(-15, 0)
	for j in range(len(h_list)):
		datfilelist = datplotfilelist_METHOD[j]
		plot_list.append([])
		for casenum in range(len(datfilelist)):
			h = h_list[j]
			R1, R2, R3,Emax_nodivider = loadtxt(datfilelist[casenum], unpack = 1, skiprows = 2)
			time = np.linspace(0, T, len(R1))
			if rescaletag == 1:
				time = time / (h ** a)
				ax1.set_xlim(0, 17500)
			else:
				ax1.set_xlim(0, 17500)
			data = [time, R1, R2, R3,Emax_nodivider]
			plot_list[j].append(data[column_tag])
		if column_tag < 4:
			ax1.plot(data[0], mean(plot_list[j], axis=0), c = color_list[j], marker = marker_list[j], markersize=1.5, linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
		if column_tag ==4 :
			ax1.plot(data[0], amin(log10(plot_list[j]), axis=0), c = color_list[j], marker = marker_list[j], markersize=1.5, linestyle = '--', linewidth = 2)
			ax1.plot(data[0], amax(log10(plot_list[j]), axis=0), c = color_list[j], marker = marker_list[j], markersize=1.5, linestyle = '--', linewidth = 2)
			ax1.plot(data[0], mean(log10(plot_list[j]), axis=0), c = color_list[j], marker = marker_list[j], markersize=1.5, linestyle = '-', linewidth = 2, label = "h=" + str(h), markeredgecolor = 'none')
		
			ax1.fill_between(data[0], amin(log10(plot_list[j]), axis=0), amax(log(plot_list[j]), axis=0), facecolor=color_list[j], alpha=0.2)			
	if rescaletag == 0:
		legend1 = ax1.legend(loc=9, shadow=True, bbox_to_anchor=(0.9, .4))
	fig.savefig(figpath, dpi=500)
	plt.close()

def plot_hT(datplotlist_total, figpath):
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	ax1.set_xlabel("log(h)",fontsize=18)
	ax1.set_ylabel("log(T)",fontsize=18)	
	
	ax1.grid(True,linestyle= '-',which='major',color= '0.75')
	ax1.grid(True,linestyle= '-',which='minor',color= '0.75')
	ax1.grid(True, which='both')
	ax1.minorticks_on()	
	
#	ax1.set_yscale('log')
#	ax1.set_xscale('log')
	
#	ax1.set_ylim(3, 7.5)
	
	logthreshold_list_tot = []
	logh_list = []	
	for j in range(len(h_list[:3])):
		h = h_list[:3][j]
		datfilelist = datplotlist_total[j]
		logthreshold_list = []
		for casenum in range(len(datfilelist)):	
			R1, R2, R3,Emax_nodivider = loadtxt(datfilelist[casenum], unpack = 1)
			time = np.linspace(0, T, len(R1))
			for i in range(len(time)):
				if R2[i] > 0.85:
					logthreshold_list.append(log(time[i]))
#					if casenum == 0:
#						print (time[i],log(time[i]))
					break
		logh_list.append(log(h))
		logthreshold_list_tot.append(mean(logthreshold_list))
		
#	logthreshold_list_tot = np.array(logthreshold_list_tot)
	[a,b] = np.polyfit(logh_list, logthreshold_list_tot, 1)
	
	ax1.plot(logh_list, logthreshold_list_tot, c = 'b', marker = marker_list[j], linestyle = '-', linewidth = 2, label = "data")
	ax1.plot([logh_list[0], logh_list[-1]], [logh_list[0] * a + b, logh_list[-1] * a + b], c = "black", marker = '.', linestyle = '--', linewidth = 2, label = "fit: T = " + str(round(exp(b),1)) + " $h^{" + str(round(a,3)) +"}$")
#	print np.polyfit(np.log(h_list), threshold_list, 1)
	legend1 = ax1.legend(loc=9, shadow=True,bbox_to_anchor=(1.05, .4))
	fig.savefig(figpath, dpi=500)
	return logthreshold_list, a

if __name__ == "__main__":
	NumofCases = 10
#	h_list = [2e-1, 5e-1, 1e0, 1.5e0, 2e0, 1e-1, 5e-2, 2e-2, 5e-3, 2e-3, 1e-3]
#	h_list = [ 5e-3, 2e-3, 1e-3, 2e-1, 5e-1, 1e0, 1.5e0]
	h_list = [1e-1,5e-2,2e-2,1e-2,5e-3]
#	h_list = [5e-3, 2e-3,1e-3]
	marker_list = ["v", "o", "^", "s", "8", "D", "+", ".", "^", "."]
	color_list = ["blue", "green", "red", "purple", "orange", "brown", "black", "yellow", "magenta", "cyan"]
	column_label_list  = ["time", "R", "$R_{2}$", "R3", "$\mathrm{Err}_{1,\mathrm{MMS}^{*}}(t)$", "Emax", "Emax_ratio", "Emax_nodivider"]
	column_name_list  = ["time", "R", "R2", "R3", "E2", "Emax", "Emax_ratio", "Emax_nodivider"]
	
	alpha = 3e-1
	psdelta = alpha * 2e0 * pi #REPULSIVE REGIME (alpha = 0.5, sigma = 0.01, h = 2, stable six cluster, h=3 stable 3 cluster, EM;< )alpha = 0.5, sigma = 0.01, h = 2, 6 to 4 cluster, MS; no cluster, sRK4)
	sigma = 1e-1
	figname = "_EH_"

	datplotfilelist_total = [] 

#	for casenum in range(NumofCases):
	casenum = int(sys.argv[1])-1	
	phi0datfile = "phiin" + str(casenum) + ".dat"

	N = 100
	phiin = loadtxt(phi0datfile, delimiter=',', unpack=True)

	datplotfilelist = []

	for j in range(len(h_list)):
		h = h_list[j]
		if h == 5e-3:
			T = 500000e0
		else:
			T = int(500000e0 * (h ** (-1e0)))
		datfile = "h=" + str(h) + "_T=" + str(T)  + "_sigma=" + str(sigma)  + "_alpha=" + str(alpha) + "_EH" + str(casenum)+ "_MMS.dat"
		sqrth = sqrt(h)
		Nint = int(T/h)
		
		integration(h, sqrth, Nint, phiin, psdelta, datfile) # Euler-Heun is a Stratonovich scheme, does not need to add shift, should be the same as above

		datplotfilelist.append(datfile)
	datplotfilelist_total.append(datplotfilelist)
#	datplotfilelist_total = list(map(list, zip(*datplotfilelist_total)))
#	for column in [1,2,3,4]:
#		fig_name2 = "alpha=" + str(alpha) + "_sigma=" +str(sigma) + figname + column_name_list[column] + "_hdep_noscaling_ind_2noise.png"
#		fig_name3 = "alpha=" + str(alpha) + "_sigma=" +str(sigma) + figname + column_name_list[column] + "_hdep_noscaling_stat_2noise.png"
#		plotevol2(datplotfilelist_total, fig_name2, 0, column)
##		plotevol3(datplotfilelist_total, fig_name3, 0, column)
#		if column == 2: #R2
#			hTscaleplot = "alpha=" + str(alpha) + "_sigma=" +str(sigma) + "_eulermaru_hT_scaling.jpg"
#			scaledplot = "alpha=" + str(alpha) + "_sigma=" +str(sigma) + "_eulermaru_Z_hdep_rescaled.jpg"			
##			logthreshold_list, a = plot_hT(datplotfilelist_total, hTscaleplot + ".jpg")
#			a = -1.
#			plotevol2(datplotfilelist_total, scaledplot, 1, column)	
