'''
Non phase oscillator model: Van der Pol and Fizhugh-Nagumo oscillators - do they cluster the same way as phase oscillators with Euler, under common noise and repulsive coupling?
'''
from wsroutine import *
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, exp, arccos, sqrt, log, loadtxt, fabs, std, arctan2,  real, imag, mean, random, amax
import cmath
from time import sleep
import matplotlib.gridspec as gridspec
import os
import sys

def wrap(phi):
	#returns angle (vector) phi to interval [0,2pi]
	return np.mod(phi, 1e0)
	
def VdP_Ck(pseudophase, k):
	#mean field amplitude to kth harmonic of N non-phase oscillators
	sumN = sum(exp(1j * k * 2e0 * pi * pseudophase))
	R, Theta = cmath.polar(sumN / (N))
	return R, Theta	
	
def ode_vdP(var):
	return np.array([var[1], mu * (1e0 - var[0] ** 2e0) * var[1] - var[0] - b * (1e0 / N * sum(var[1]) - var[1] ) ])

def RK4(func, var, h):
	#algorithm function of RungeKutta of 4th order, input the initial vector var, outputs the new vector var, h the time step. 
		#var, k1 to k4 are all N-Vector
	k1 = h * func(var)
	k2 = h * func(var + k1 / 2e0)
	k3 = h * func(var + k2 / 2e0)
	k4 = h * func(var + k3)
	var = var + (k1 + 2e0 * k2 + 2e0 * k3 + k4) / 6e0
	return var

def Euler_Maru_vdP_nonoise(var, h):
	var = var + h * ode_vdP(var)
	return var

def sRK4(var, h, sigma, eta, psi):
	#second order
	K1 = var
	K2 = var + TwoThirds * h * ode_vdP(K1) + TwoThirds * eta * sigma
	K3 = var + ThreeHalves * h * ode_vdP(K1) - OneThird * h * ode_vdP(K2) + OneHalf * eta * sigma + OneSixth * eta * sigma - TwoThirds * psi * sigma
	K4 = var + SevenSixth * h * ode_vdP(K1) - OneHalf * eta * sigma + OneHalf * eta * sigma + OneSixth * psi * sigma + OneHalf * psi * sigma
	var = var + h * (OneFourth * ode_vdP(K1) + ThreeFourth * ode_vdP(K2) - ThreeFourth * ode_vdP(K3) + ThreeFourth * ode_vdP(K4)) + eta * (- OneHalf * sigma + ThreeHalves * sigma - ThreeFourth * sigma + ThreeFourth * sigma) + psi * (ThreeHalves * sigma - ThreeHalves * sigma)
	return var
	
def Euler_Maru_vdP_additiv_noise(var, h, sqrth, xi, eta):
	var = var + h * ode_vdP(var) + sqrth * sigma * np.array([xi* np.ones(N), np.zeros(N)])
	return var
	
def integration(h, sqrth, sigma, Nint, xin, yin, datfile, noisetag):
	var = np.array([xin, yin])
	f = open(datfile, 'w')
	f.write('i'+ '\t' + 'R_1' + '\t' + 'R_2' + '\t' + 'R_3'+ '\t' + 'R_4' + '\t' + 'R_5' + '\n')
#	f.close()
	
	crossing_time_lst  = np.zeros(N)
	num_crossing_1 = np.zeros(N)
	crossing_opp0 = np.zeros(N) # make sure all have crossed the poicare section at y = 0, x >0
	counter = 1
	crossing = np.zeros(N)
	crossing_opp = np.ones(N)
	varx_list= []
	vary_list= []
	pseudophase_list = []
	for i in range(int(Nint)):
		time = i * h
#		print(time)
		if noisetag == 0:
#			var2 = Euler_Maru_vdP_nonoise(var, h)
			var2 = RK4(ode_vdP, var, h)
		elif noisetag == 1:
#			xi = xi_list[i]
#			eta = eta_list[i]
#			var2 = Euler_Maru_vdP_additiv_noise(var, h, sqrth, xi, eta) # additive noise, the correct one
			
			u = np.random.normal(0, 1)
			v = np.random.normal(0, 1)
			eta = u * sqrth
			psi = sqrth * (u/2e0 + v/c)
			var2 = sRK4(var, h, sigma, eta, psi)
			
		ysignchange = np.sign(var2[1]) - np.sign(var[1]) # whether the sign has changed in y direction
		xsignchange = np.sign(var2[0]) - np.sign(var[0]) # whether the sign has changed in x direction

#		if i % int(100e0/h) == 0:
#			plotvdP(var2, time, sigma)
#			f = open(datfile, 'a')
#			f.write(str(time) + '\t' + str(std(var2[0][:50])) + '\t' + str(std(var2[0][50:])) + '\t' + str(std(var2[1][:50])) + '\t' + str(std(var2[1][50:])) + '\n')
#			f.close()
			
		if time > 0: #skip some transit where the oscillators are not yet on the limit cycle
			crossing_opp0[np.logical_and(ysignchange > 0, np.sign(var2[0]) < 0)] = 1 # oscillator crosses poincare section (y goes from positive to negative and x > 0 )
#			if i % int(0.1e0/h) == 0:
#				print(var[0][0], var[1][0], time)
			for j in range(N):
				if crossing_opp0[j] == 1: # passing the starting line of poincare section
					if ysignchange[j] < 0 and np.sign(var2[0][j]) > 0 and crossing_opp[j] == 1:
						crossing[j] = 1
						
						num_crossing_1[j] += 1
						
						if max(num_crossing_1) > counter:
							counter += 1
							maxind = np.where(num_crossing_1 == max(num_crossing_1))
							tn_old = crossing_time_lst[maxind[0]] # first to cross
#							print(var2[0][maxind[0]], var2[1][maxind[0]], crossing[maxind[0]], crossing_opp[maxind[0]], crossing_time_lst[maxind[0]], num_crossing_1[maxind[0]])
						crossing_time_lst[j] = time
						crossing_opp[j] = 0
#						if j == 0:
#							print("rezero")
#							print("A", var[0][j], var[1][j])
#							print("A", var2[0][j], var2[1][j], crossing[j], crossing_opp[j], crossing_time_lst[j], num_crossing_1[j])
					if ysignchange[j] > 0 and np.sign(var2[0][j]) < 0 and crossing[j] == 1:
						crossing_opp[j] = 1
						crossing[j] = 0
#						if j == 0:
#							print("B", var[0][j], var[1][j])
#							print("B", var2[0][j], var2[1][j], crossing[j], crossing_opp[j], crossing_time_lst[j], num_crossing_1[j])

			if np.all(crossing_time_lst != 0) and counter > 1:
#				if i % int(2e0/h) == 0:
##					print(var2[0][maxind[0]], var2[1][maxind[0]], crossing[maxind[0]], crossing_opp[maxind[0]], crossing_time_lst[maxind[0]], num_crossing_1[maxind[0]])
#					plotvdP(var2, time, sigma)
			
				tn_new = crossing_time_lst[maxind[0]]
				if tn_new != 0 and tn_old != 0:
					if i % int(5e0/h) == 0:
						period = (tn_new - tn_old)[0]
						pseudophase = wrap(np.array(crossing_time_lst - tn_old)/period) # 2pi is multiplied in the function of VdP_Ck
	#					print(time, VdP_Ck(pseudophase, 1e0)[0], VdP_Ck(pseudophase, 2e0)[0], VdP_Ck(pseudophase, 3e0)[0], VdP_Ck(pseudophase, 4e0)[0], VdP_Ck(pseudophase, 5e0)[0])
	#					print(time, crossing_time_lst, tn_old, tn_new)
	#					print(pseudophase)
	#					for j in range(N):
	#						print(time, crossing_time_lst[j], tn_old, tn_new)
	#						print(time, pseudophase[j])
	#					print('tn_new - tn_old', tn_new - tn_old, tn_new, tn_old)
						R = VdP_Ck(pseudophase, 1e0)[0]
						R2 = VdP_Ck(pseudophase, 2e0)[0]
						R3 = VdP_Ck(pseudophase, 3e0)[0]
						R4 = VdP_Ck(pseudophase, 4e0)[0]
						R5 = VdP_Ck(pseudophase, 5e0)[0]

#						plotcrossT(crossing_time_lst-tn_new, R, R2, R3, R4, R5, time, sigma)
#						simplehist(pseudophase, "phase", 100, time, sigma)
#						plotvdP(var2, time, sigma)
						varx_list.append(var2[0])
						vary_list.append(var2[1])
						pseudophase_list.append(pseudophase)
						
						f.write(str(time) + '\t' + str(period) + '\t' + str(R) + '\t' + str(R2) + '\t' + str(R3) + '\t' + str(R4) + '\t' + str(R5) + '\n')
						
		var = var2
	f.close()
#	sys.exit()

def plotvdP(dat, time, sigma):
	# plotting the phase on a unit circle
	fig = plt.figure(2)
	fig.clf()
	ax = fig.add_subplot(111, aspect='equal', xlim=(-8,8), ylim=(-8,8))
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
#	ax = fig.add_subplot(111, aspect='equal')
	cls2 = ['b','g','r']
	dat = np.transpose(dat)
	for j in range(N):
		plt.scatter(dat[j][0], dat[j][1], c = cls2[0])
#		if j < 10:
#			ax.annotate(str(j), xy=(dat[j][0], dat[j][1]), textcoords='offset points')
	plt.grid()
#	plt.show(block=False)
#	sleep(1)
	fig.savefig(os.path.join(os.getcwd(), 'case'+str(casenum) + '_' +str(time) + '_sigma=' +str(sigma) + "_limit_cycle_vdp.jpg"), dpi=500)
	plt.close()

def simplehist(var, varname, binnum, time, sigma):
	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111)
	ax.set_xlabel(varname,fontsize=12)
#	binlist = np.arange(-1e0, 1e0 + (max(var)-min(var))/binnum, (max(var)-min(var))/float(binnum))
#	xlocs = np.arange(-1e0 + .5*(max(var)-min(var))/binnum, 1e0 + .5*(max(var)-min(var))/binnum, (max(var)-min(var))/binnum)
#	plt.hist(var, bins=binlist, alpha = 0.3, color = 'blue', normed = True)
#	histx, bin_edgesx = np.histogram(var, bins=binlist, normed = True)
#	print(len(xlocs), len(histx))	
	ax.hist(var, bins = binnum, range = [0, 1], color='blue')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 100])
	fig.savefig(os.path.join(os.getcwd(), 'case'+str(casenum) + '_' +str(int(time)) + '_sigma=' +str(sigma) + "_hist_vdp.jpg"), dpi=500)
	
def plotcrossT(crossing_time_lst, R, R2, R3, R4, R5, time, sigma):
	# plotting the phase on a unit circle
	fig = plt.figure(2)
	fig.clf()
	ax = fig.add_subplot(111, xlim=(min(crossing_time_lst),max(crossing_time_lst)), ylim=(-10,10))
	
	for j in range(N):
		plt.plot(crossing_time_lst[j] * np.ones(2), [-7,7], c = 'b')
	ax.annotate('R = ' + str(R), xy=(min(crossing_time_lst), 10))
	ax.annotate('R2 = ' + str(R2), xy=(min(crossing_time_lst), 9))
	ax.annotate('R3 = ' + str(R3), xy=(min(crossing_time_lst), 8))
	ax.annotate('R4 = ' + str(R4), xy=(min(crossing_time_lst), 7))
	ax.annotate('R5 = ' + str(R5), xy=(min(crossing_time_lst), 6))
	
#	plt.show(block=False)
#	sleep(1)
	fig.savefig(os.path.join(os.getcwd(), 'case'+str(casenum) + '_' +str(time) + '_sigma=' +str(sigma) + "_cross_time_vdp.jpg"), dpi=500)
	plt.close()
	
def plotevol2(datplotfilelist, figpath, rescaletag, column_tag):
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	if rescaletag == 0:
		ax1.set_xlabel("Time",fontsize=18)
	if rescaletag == 1:
		ax1.set_xlabel("$T/h^{a}$ $(a = $" + str(round(a,3)) + ")", fontsize=18)
	if rescaletag == 2:
		ax1.set_xlabel("$T/T^{*}$",fontsize=18)
	ax1.set_ylabel(column_label_list[column_tag],fontsize=18)
#	ax1.set_yscale('log')
	ax1.grid(True,linestyle= '-',which='major',color= '0.75')
	ax1.grid(True,linestyle= '-',which='minor',color= '0.75')
	ax1.grid(True, which='both')
	ax1.minorticks_on()	
	
	
	ax1.set_ylim(0.5, 1.1)
	ax1.set_xlim(0, T*3)
#	if column_tag != 1:
#		ax1.set_ylim(0, 1)
	plot_list = []
	for j in range(len(sigma_list)):
		datfilelist = datplotfilelist[j]
		sigma = sigma_list[j]

		time, tn_min_to, R1, R2, R3, R4, R5 = loadtxt(datfilelist, unpack = 1, skiprows = 1)
#		time, var1x, var2x, var1y, var2y = loadtxt(datfilelist, unpack = 1, skiprows = 1)
#		index = np.where(tn_min_to > 4.)
#		time, R1, R2, R3, R4, R5 = loadtxt(datfilelist, unpack = 1, skiprows = 1)
		
		if rescaletag == 1:
			time = time * h ** (-a)
#		data = [time[index[0]], tn_min_to[index[0]], R1[index[0]], R2[index[0]], R3[index[0]], R4[index[0]], R5[index[0]]]
		data = [time, tn_min_to, R1, R2, R3, R4, R5]
#		data = [time, var1x, var2x, var1y, var2y ]
#		data = [time, R1, R2, R3, R4, R5]
		plot_list.append(data[column_tag])
		ax1.plot(data[0], plot_list[j], c = color_list[j], marker = marker_list[j], markersize=1.5, alpha = 0.3, linestyle = '-', linewidth = 2, label = "sigma=" + str(sigma), markeredgecolor = 'none')
		movingavg = np.convolve(plot_list[j], np.ones((400,))/400, mode='same')
		print(movingavg)
		print(len(plot_list[j]))
		ax1.plot(data[0][400:-400], movingavg[400:-400], c = 'red', marker = marker_list[j], markersize=1.5, linestyle = '-', linewidth = 2, markeredgecolor = 'none')
		
	legend1 = ax1.legend(loc=9, shadow=True, bbox_to_anchor=(0.9, .4))
	fig.savefig(figpath, dpi=500)
	plt.close()

def	average_Daido(datplotfilelist, figpath):
	fig = plt.figure(figsize=(10,7))
	ax1 = fig.add_subplot(111)
	ax1.set_xlabel("$\sigma$",fontsize=18)
	
	ax1.grid(True,linestyle= '-',which='major',color= '0.75')
	ax1.grid(True,linestyle= '-',which='minor',color= '0.75')
	ax1.grid(True, which='both')
	ax1.minorticks_on()	
	
	ax1.set_ylim(0, 1)
	
	for i in range(5):
		plot_list = []
		for j in range(len(sigma_list)):
			datfilelist = datplotfilelist[j]
			sigma = sigma_list[j]

			time, tn_min_to, R1, R2, R3, R4, R5 = loadtxt(datfilelist, unpack = 1, skiprows = 1)
			data = [R1, R2, R3, R4, R5]
			
			plot_list.append(mean(data[i]))
			if j == 0 and i ==2:
				print(datfilelist, i, mean(data[i]), sigma)
				print(data[i])		
#		print(sigma_list, plot_list)
		ax1.plot(sigma_list, plot_list, c = color_list[i], marker='o', linewidth = 2, label = column_label_list[2+i])
	legend1 = ax1.legend(loc=9, shadow=True, bbox_to_anchor=(0.9, .4))
	fig.savefig(figpath, dpi=500)
	plt.close()	
	
if __name__ == "__main__":
	Ncasenum = 1
	h = 1e-3
#	
	processornum = 1

	marker_list = ["v", "o", "^", "s", "8", "D", "+", ".", "^", "."]
	color_list = ["blue", "green", "red", "purple", "orange", "brown", "black", "yellow", "magenta", "cyan"]
	column_label_list  = ["time", "period", "$<R>_t$", "$<R_2>_t$", "$<R_3>_t$", "$<R_4>_t$", "$<R_5>_t$"]
#	column_label_list  = ["time", "varx1", "varx2", "vary1", "vary2"]
	
	column_label_list  = ["time", "period", "R", "R2", "R3", "R4", "R5"]
	T = 10000e0
	
	mu = 1. #van der pol nonlinearity parameter
	b = 0.01  #coupling strength (repulsiveness degree) # b cannot be too big because we want to use mu to control the shape of the limit cycle, when b is large it means when the repulsive coupling is large it distorts the limit cycle too much

#	mu = .1
#	b = 0.001
	
#	sigma_list = [0, 0.015, 0.03, 0.05, 0.07]
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
	
	casenum = 6
	datplotfilelist = []
	for l in range(len(sigma_list)):
#	l = int(sys.argv[1])-1
		N = 100
		sigma = sigma_list[l]
	
		if casenum > 1:
			if casenum ==5:
				xy0datfile = "xyin" + str(casenum) + "_fourclusterIC_lesscompact.dat"
				T = 2e5
			
			if casenum ==4:
				xy0datfile = "xyin" + str(casenum) + "_fourclusterIC.dat"				
	#				xin = np.random.uniform(-1.,1.,size = N) * 5.
	#				yin = np.random.uniform(-1.,1.,size = N) * 5.			

	#				xin1 = np.random.uniform(1.49, 1.5,size = int(N/4)) * 5.
	#				yin1 = np.random.uniform(2.49, 2.5,size = int(N/4)) * 5.			

	#				xin2 = np.random.uniform(1.49, 1.5,size = int(N/4)) * 5.
	#				yin2 = np.random.uniform(-.5,-.49,size = int(N/4)) * 5.		

	#				xin3 = np.random.uniform(-1.5,-1.49,size = int(N/4)) * 5.
	#				yin3 = np.random.uniform(-2.5,-2.49,size = int(N/4)) * 5.	

	#				xin4 = np.random.uniform(-1.5,-1.49,size = int(N/4)) * 5.
	#				yin4 = np.random.uniform(-.49,-.5,size = int(N/4)) * 5.
	#				
	#				xin = np.concatenate((xin1, xin2, xin3, xin4))
	#				yin = np.concatenate((yin1, yin2, yin3, yin4))

			if casenum ==6:
				xy0datfile = "xyin" + str(casenum) + "_twoclusterIC.dat"	
								
				xin1 = np.random.uniform(1.95, 2.05,size = int(N/2)) 
				yin1 = np.random.uniform(-0.25, -0.35,size = int(N/2))			

				xin3 = -xin1
				yin3 = -yin1	

				xin = np.concatenate((xin1,xin3))
				yin = np.concatenate((yin1,yin3))
				
#			f = open(xy0datfile, 'w')
#			for i in range(len(xin)):
#				f.write(str(xin[i]) + '\t ' + str(yin[i]) + '\n ')
#			f.close()

		else:
			xy0datfile = "xyin" + str(casenum) + ".dat"
		
		xin, yin = loadtxt(xy0datfile, unpack=True)

		sqrth = sqrt(h)
		Nint = int(T/h)

		if sigma == 0:
			datfilelist = "h=" + str(h) + "_T=" + str(T) + "_sigma=" +str(sigma) + "_RK4_" + str(casenum) + "_vdP_mu=" + str(mu) + "_b=" + str(b) + ".dat"
#			integration(h, sqrth, sigma, Nint, xin, yin, datfilelist, 0)
		else:
			datfilelist = "h=" + str(h) + "_T=" + str(T) + "_sigma=" +str(sigma) + "_sRK4_" + str(casenum) + "_vdP_mu=" + str(mu) + "_b=" + str(b) + "_vdP_specialIC_stable2clusters.dat"
#			integration(h, sqrth, sigma, Nint, xin, yin, datfilelist, 1)

		datplotfilelist.append(datfilelist)
#		print(datplotfilelist)

		for column in range(3,4):
			if sigma == 0:
				fig_name = "h=" +str(h) + "case=" +str(casenum) + column_label_list[column] + "_hdep_noscaling_vdP_RK4_mu=" + str(mu) + "_b=" + str(b) + "_sigma=" + str(sigma) + ".jpg"
			else:
				fig_name = "h=" +str(h) + "case=" +str(casenum) + column_label_list[column] + "_hdep_noscaling_vdP_sRK4_mu=" + str(mu) + "_b=" + str(b) + "_sigma=" + str(sigma) + "_specialIC_stable2clusters.jpg"
			plotevol2(datplotfilelist, fig_name, 0, column)

#	average_Daido(datplotfilelist, 'average_Daido_OP_high_nonlin.jpg')
#	average_Daido(datplotfilelist, 'average_Daido_OP_low_nonlin.jpg')
