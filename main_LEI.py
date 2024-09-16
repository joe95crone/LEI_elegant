# JC 12/12/2023
# Code to MonteCarlo modify electron bunch for laser-electron interaction
# based on Pan et al PRAB 22, 040702 (2019)
# Note that this is currently only accurate for head-on collisions

import time
import numpy as np
from scipy import constants

from scipy.stats import gaussian_kde
from scipy import stats
from scipy.interpolate import interp1d

import os
import argparse
from sddsbeam import sddsbeam

#--------------
# CODE OUTLINE
#--------------
# take in bunch, sample affected particles, calculate modification, write bunch

# 3 stages
# - sample particles & determine if they interact random number test against interaction probability - how?
# - sample theta in range 0 < theta < pi - from the electron distribution
# - sample phi in range 0 < phi < 2pi 

# electron interaction probability is flux per no. electrons for each interaction
# - varies per interaction
# - dependent on beam size and particle losses etc.

# theta distribution is more complicated 
# - sample from the dsig/dEg vs Eg distribution setup for a head-on bunch
# - convert the sampled values to therta 

# azimuthal distribution is simple
# - sample uniform distribution in range 0 to 2pi

#--------------------------------------
# calculate from initial + final bunch
#--------------------------------------

# root mean square
def rms(data):
	return np.sqrt(np.mean(data**2))

#-----------------------------------
# general ICS calculation functions
#-----------------------------------

# reference energy of beam in MeV
# mean or median??
def Ee(p):
	return np.mean(p*constants.physical_constants["electron mass energy equivalent in MeV"][0])*constants.mega

def EL(wavelength):
	return ((constants.physical_constants["Planck constant"][0])*constants.c)/(wavelength*constants.e)

# Lorentz factor 
def gamma(p):
	return ((Ee(p)/constants.mega) + constants.physical_constants["electron mass energy equivalent in MeV"][0])/constants.physical_constants["electron mass energy equivalent in MeV"][0] 

# Lorentz speed factor
def SpeedFac(p):
	return np.sqrt(1-1/gamma(p)**2) 

# centre of mass photon energy (i.e. mid part of ICS process)
# crossing angle = 0 for head-on convention
def EL_cen(p, wavelength, phi_cross):
	return gamma(p)*EL(wavelength)*(1+SpeedFac(p)*np.cos(phi_cross))

# scattered photon energy
def Eg(p, wavelength, phi_cross, theta):
	return (EL(wavelength)*(1+SpeedFac(p)*np.cos(phi_cross)))/(1-SpeedFac(p)*np.cos(theta)+(EL(wavelength)/Ee(p))*(1+np.cos(phi_cross+theta)))

# recoil parameter
def X(p, wavelength, phi):
	return 2*gamma(p)*EL(wavelength)*(1 + SpeedFac(p)*np.cos(phi))/((constants.physical_constants["electron mass energy equivalent in MeV"][0])*constants.mega)

# ICS cross section (in low-recoil approx.)
def sigma_c(p, wavelength, phi):
	return (constants.physical_constants["Thomson cross section"][0])*(1-X(p, wavelength, phi))

# Luminosity
def Ne(Q):
	return Q/constants.e

def NL(Epulse, wavelength):
	return Epulse/(constants.e*EL(wavelength))

def convxy(sigma_e, sigma_L):
	return np.sqrt(sigma_e**2 + sigma_L**2)

def luminosity(Q, Epulse, wavelength, sigma_ex, sigma_ey, sigma_L):
	return (Ne(Q)*NL(Epulse, wavelength))/(2*np.pi*convxy(sigma_ex, sigma_L)*convxy(sigma_ey, sigma_L))

#---------------------------------
# Monte Carlo sampling functions
#---------------------------------
# Monte Carlo interaction probability
def PintMCmacro(p, wavelength, phi, Q, Epulse, sigma_ex, sigma_ey, sigma_L, Nmacro):
	return 1 - ((sigma_c(p, wavelength, phi)*luminosity(Q, Epulse, wavelength, sigma_ex, sigma_ey, sigma_L))/Ne(Q))*(Ne(Q)/Nmacro)

# head on recoil parameter (Lorentz Invariant)
def XHO(p, wavelength):
	return 2*gamma(p)*EL(wavelength)*(1+SpeedFac(p))/(constants.physical_constants["electron mass energy equivalent in MeV"][0]*constants.mega)

# head on Y Lorentz Invariant
def YHO(p, wavelength, EgHO):
	return XHO(p, wavelength)*(SpeedFac(p)*Ee(p)-EgHO)/(SpeedFac(p)*Ee(p)-EL(wavelength))

def dsigdEgHO(p, wavelength, EgHO):
	return ((8*np.pi*constants.physical_constants["classical electron radius"][0]**2)/(XHO(p, wavelength)*(Ee(p)-EL(wavelength))))*(((1/XHO(p,wavelength))-(1/YHO(p, wavelength, EgHO)))**2 + (1/XHO(p, wavelength)) - (1/YHO(p, wavelength, EgHO)) + (1/4)*((XHO(p, wavelength)/YHO(p, wavelength, EgHO))+(YHO(p, wavelength, EgHO)/XHO(p, wavelength))))

#--------------------------------
# Modification of beam particles
#--------------------------------
# uses only individual x'_0, y'_0 and p'_0 values. 
# use convention that crossing angle = 0 is head-on	
def mod_xp(xp_0, p, wavelength, phi_cross, theta, phi):
	return xp_0 - (EL_cen(p,wavelength,phi_cross)/(Ee(p)*SpeedFac(p)))*np.sin(theta)*np.cos(phi)

def xp_var(p, wavelength, phi_cross, theta, phi):
	return (EL_cen(p,wavelength,phi_cross)/(Ee(p)*SpeedFac(p)))*np.sin(theta)*np.cos(phi)

def mod_yp(yp_0, p, wavelength, phi_cross, theta, phi):
	return yp_0 - (EL_cen(p,wavelength,phi_cross)/(Ee(p)*SpeedFac(p)))*np.sin(theta)*np.sin(phi)

# this is a simplified version! may or may not be used
def mod_p_JC(p, wavelength, phi_cross, theta):
	return (Ee(p) - Eg(p, wavelength, phi_cross, theta))/(constants.physical_constants["electron mass energy equivalent in MeV"][0]*constants.mega)

def delta_0(p, pbeam):
	return (p - np.mean(pbeam))/np.mean(pbeam)

# original Zhilong Pan and John Byrd version correctly implemented
# Doesn't include recoil so we don't consider this here! We use my version
def mod_p_ZP(p, pbeam, wavelength, phi_cross, theta):
	return Ee(p) - Ee(pbeam)*((((gamma(p)**2)*EL(wavelength))/(Ee(pbeam)*SpeedFac(pbeam)))*(np.cos(phi_cross)+SpeedFac(p)) + ((gamma(p)*EL_cen(p, wavelength, phi_cross))/(Ee(pbeam)*SpeedFac(pbeam)))*np.cos(theta))

#----------------
# Setup of paths
#----------------

parser = argparse.ArgumentParser(description='Elegant SCRIPT example')
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)

if __name__ == "__main__":
	# VARIABLES
	# p is momentum 
	# Ee is mean electron energy
	# EL is lab frame incident photon energy
	# EL_cen is centre of mass photon energy
	# phi_cross is crossing angle (phi_cross = 0 for head-on)
	# theta is the polar emission angle
	# phi is the azimuthal emission angle 
	
	#--------------
	# IMPORT BUNCH
	#--------------
	tic = time.perf_counter()
	
	# initial bunch is at the IP
	args = parser.parse_args()
	assert os.path.isfile(args.input)
	
	init_beam = sddsbeam()
	init_beam.read_SDDS_file(args.input)
	
	print(args.input)
	
	#------------------
	# Bunch Parameters
	#------------------
	# non-obtainable from the distribution 
	wvl = 1064e-9
	# crossing angle uses the zero is head-on convention 
	cross_ang = 0
	bunch_charge = 73.5e-12
	EpulseNDYAG = 100e-6
	laser_trans_size = 100e-6

	# obtained from the distribution (this should be at the IP but isn't for now)
	beam_size_x = rms(init_beam.beam.x)
	beam_size_y = rms(init_beam.beam.y)
	
	#----------------------
	# Probability Sampling
	#----------------------
	
	# no. macro
	Nmacro = len(init_beam.beam.p)
	print("No. Initial Macroparticles: ", Nmacro)
	
	# generate random numbers
	trials = np.random.rand(Nmacro)
	
	# probabilities are wrong since we are using the initial out-of-thesource bunch for now
	interaction_prob = PintMCmacro(init_beam.beam.p, wvl, cross_ang, bunch_charge, EpulseNDYAG, beam_size_x, beam_size_y, laser_trans_size, Nmacro)
	
	# indexes of particles with succcessful trials i.e. where interactions ha
	successful_trials = np.where(trials > interaction_prob)[0]
	
	NmacroINT = len(successful_trials)
	print("No. Interacted Macroparticles: ", NmacroINT)
	
	#-------------------------------------
	# Azimuthal Scattering Angle Sampling
	#-------------------------------------
	
	azimuthal_angles = np.random.uniform(0, 2*np.pi, NmacroINT)
	
	#---------------------------------
	# Polar Scattering Angle Sampling 
	#---------------------------------
	
	# get data to find distribution fit (0 to Compton Edge)
	max_ICS_energy = Eg(init_beam.beam.p, wvl, cross_ang, 0)
	print("Mean Max Scattered Photon Energy [MeV] :", max_ICS_energy/constants.mega)
	
	EgDAT = np.linspace(0,max_ICS_energy,10**5)
	
	dsigdEgDAT = dsigdEgHO(init_beam.beam.p, wvl, EgDAT)
	EgVSdsigdEgDAT = np.squeeze(np.dstack((EgDAT, dsigdEgDAT))).T
	
	# BW of KDE seems to control adherence to the curve (smaller = better)
	kernel = stats.gaussian_kde(EgVSdsigdEgDAT, bw_method=0.001)
	EgVSdsigdEgSample = kernel.resample(NmacroINT)
	
	# Conversion to angle (by interpolation)
	thetaPADAT = np.linspace(0,np.pi,10**5)
	EgPADAT = Eg(init_beam.beam.p, wvl, 0, thetaPADAT)
	
	# these conditions bound the sampling & interpolation to make sure we don't get particles higher than maximum energy (possible since KDE has bandwidth)
	EgVSdsigdEgSample[0,:][EgVSdsigdEgSample[0,:] > max(EgPADAT)] = max(EgPADAT) 
	EgVSdsigdEgSample[0,:][EgVSdsigdEgSample[0,:] < min(EgPADAT)] = min(EgPADAT)
	
	PA_interp = interp1d(EgPADAT,thetaPADAT, kind='cubic')
	polar_angles = PA_interp(EgVSdsigdEgSample[0,:])
	
	#--------------------
	# Bunch Modification
	#--------------------
	
	p_init = init_beam.beam.p[successful_trials[0]]
	
	# create modified arrays of x', y', p
	final_xp = np.array(init_beam.beam.xp)
	final_yp = np.array(init_beam.beam.yp)
	final_p = np.array(init_beam.beam.p)
	for i in range(len(successful_trials)):
		final_xp[successful_trials[i]] = mod_xp(init_beam.beam.xp[successful_trials[i]], init_beam.beam.p[successful_trials[i]], wvl, cross_ang, polar_angles[i], azimuthal_angles[i])
		final_yp[successful_trials[i]] = mod_yp(init_beam.beam.yp[successful_trials[i]], init_beam.beam.p[successful_trials[i]], wvl, cross_ang, polar_angles[i], azimuthal_angles[i])
		#final_p[successful_trials[i]] = mod_p_ZP(init_beam.beam.p[successful_trials[i]], init_beam.beam.p, wvl, cross_ang, polar_angles[i]) # Zhilong Pan
		final_p[successful_trials[i]] = mod_p_JC(init_beam.beam.p[successful_trials[i]], wvl, cross_ang, polar_angles[i]) # ME
	
	# initialise modified beam (think this also makes init_beam identical to mod_beam)
	mod_beam = sddsbeam()
	mod_beam = init_beam
	
	# change the x', y', p values
	mod_beam.beam.xp = final_xp
	mod_beam.beam.yp = final_yp
	mod_beam.beam.p = final_p
	
	#-------------
	# WRITE BUNCH
	#-------------
	# write_SDDS_file has been modified to use p not cp (as I modify cp here)
	mod_beam.write_SDDS_file(args.output)

	toc = time.perf_counter()
	print("Simulation Time [s]: ", toc-tic)
