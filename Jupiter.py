# %%
from Test import data
from Test import file_name
from Test import SN_LC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import randomcolor
from matplotlib.pyplot import figure
from Test import plot_Amp
from Test import plot_Beta
from Test import plot_t0
from Test import plot_t1
from Test import plot_Trise
from Test import plot_Tfall
from Test import curvefitting_and_plot
from Test import rise1
from Test import fall

# %%
# test for GPC1 in g filter
target = "2018agk"  

file_location = file_name(target)
original_data = data(file_location,'g', 'GPC1') # plot based on the the Instrument gp band
time = np.array(original_data[0]) #MJD

mags = np.array(original_data[1]) #Mag
mag_errors = np.array(original_data[2]) #Magerr

#plot_Amp(counts = 9, x = time)
#plot_Beta(counts = 9, x = time)
#plot_t0(counts = 9 , x = time)
#plot_t1(counts = 9, x = time)
#plot_Trise(counts = 9, x = time)
#plot_Tfall(counts = 9, x = time)
A = 2.8 
B = 3.6e-2
t0 = 5.81967533e+04 # based on all the plots, not based on the filter and instrument
t1 = 5.81972019e+04
Trise = 2.53458254e+00
Tfall = 2.32956652e+02
zero_points = 1.87637350e+01
guessparams_1 = np.array([A, B, t0, t1, Trise])
guessparams_2 = np.array([A, B, t0, t1, Trise, Tfall]) # rise function's params
guessparams_3 = np.array([A, B, t0, t1, Trise, Tfall, zero_points]) # SN_LC function's params
curvefitting_and_plot(SN_LC, time, mags, dy = mag_errors, guessparams= guessparams_3, target_and_filter_inst = 'target = 2018agk, Instrument= GPC1, filter = g' )
"""
Test 

"""
#xsmooth1 = np.linspace(np.min(time), np.max(time), len(time))
#fsmooth1 = rise(xsmooth1, *guessparams_1) # len(fsmooth = len(timd)) =78
#fsmooth2 = rise1(xsmooth1, *guessparams_1) # len(fsmooth2) =36
#fsmooth3 = fall(xsmooth1, *guessparams_2) # len(fsmooth3) = 42
#fsmooth1 = SN_LC(xsmooth1, *guessparams_3)
#SN = np.concatenate((fsmooth2, fsmooth3))
#SN1 =-2.5 * np.log10(SN) + 18
#print(SN1)
#print(len(SN))
#popt, pcov = curve_fit(SN_LC, xsmooth1, mags, p0 = guessparams_3)
#print(popt)


# %%
# test for GPC1 in g filter
target = "2018agk"  

file_location = file_name(target)
original_data = data(file_location,'gp', 'Sinistro') # plot based on the the Instrument gp band
time = np.array(original_data[0]) #MJD

mags = np.array(original_data[1]) #Mag
mag_errors = np.array(original_data[2]) #Magerr

#plot_Amp(counts = 9, x = time)
#plot_Beta(counts = 9, x = time)
#plot_t0(counts = 9 , x = time)
#plot_t1(counts = 9, x = time)
#plot_Trise(counts = 9, x = time)
#plot_Tfall(counts = 9, x = time)
A = 4.493 
B = 0.367
t0 = 5.81917533e+04 # based on all the plots, not based on the filter and instrument
t1 = 5.81916019e+04
Trise = 2.95458254e+00
Tfall = 23.2956652e+00
zero_points = 1.77637350e+01
#guessparams_1 = np.array([A, B, t0, t1, Trise])
#guessparams_2 = np.array([A, B, t0, t1, Trise, Tfall]) # rise function's params
guessparams_3 = np.array([A, B, t0, t1, Trise, Tfall, zero_points]) # SN_LC function's params
curvefitting_and_plot(SN_LC, time, mags, dy = mag_errors, guessparams= guessparams_3, target_and_filter_inst = 'target = 2018agk, Instrument= Sinistro, filter = gp' )
"""
Test 
"""
#xsmooth1 = np.linspace(np.min(time), np.max(time), len(time))
#fsmooth1 = rise(xsmooth1, *guessparams_1) # len(fsmooth = len(timd)) =78
#fsmooth2 = rise1(xsmooth1, *guessparams_1) # len(fsmooth2) =36
#fsmooth3 = fall(xsmooth1, *guessparams_2) # len(fsmooth3) = 42
#fsmooth1 = SN_LC(xsmooth1, *guessparams_3)
#SN = np.concatenate((fsmooth2, fsmooth3))
#SN1 =-2.5 * np.log10(SN) + 18
#print(SN1)
#print(len(SN))
#popt, pcov = curve_fit(SN_LC, xsmooth1, mags, p0 = guessparams_3)
#print(popt)


# %%


# %%


# %%



