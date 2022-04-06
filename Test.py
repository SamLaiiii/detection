import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as ran
import scipy.optimize as curve_fit
import scipy.stats as stat


# create a function to locate the fle data 
def file_name(target):
    return "C:/Users/Sam/Downloads/{target}_data.snana.txt".format(target = target) #file locaiton 


# import the data from the files
def data(file_location, Filter, instrument):
    data = pd.read_csv(file_location, delim_whitespace= "True", comment="#")
    df = pd.DataFrame(data, columns = ['FLT', 'MJD', 'MAG', 'MAGERR', 'FLUXCAL', 'FLUXCALERR', 'INSTRUMENT'])
    filter = df['FLT'].to_numpy()
    MJD = df['MJD'].to_numpy()
    mag = df['MAG'].to_numpy()
    magerr = df['MAGERR'].to_numpy()
    fluxcal = df['FLUXCAL'].to_numpy()
    fluxcalerr = df['FLUXCALERR'].to_numpy()
    Instrument = df['INSTRUMENT'].to_numpy()
    unique_filter = np.unique(filter)
    unique_Instrument = np.unique(Instrument)
#    print(mag)                        # to show all the mag 
    print(unique_filter)              # to show what kind of filters are there
    print(unique_Instrument)          # to show the Instrument
    
    mag_new = [] 
    MJD_new = []
    magerr_new = []
    flux_new = []
    fluxerr_new = []
    for index in range(len(mag)):
        if mag[index] >= 0 and magerr[index] != 1.086 and magerr[index] > 0:
            mag_new.append(mag[index])
            MJD_new.append(MJD[index])
            magerr_new.append(magerr[index])
            flux_new.append(fluxcal[index])
            fluxerr_new.append(fluxcalerr[index])
        else:
            pass
#    print("mag new:", mag_new)
#    print("MJD new:", MJD_new)
#    print("magerr new:", magerr_new)
#    print("fluxcal new:", flux_new)
#    print("fluxcalerr new:", fluxerr_new) 

#To get the mag and MJD list for each filter in different instrument
    mag_list = []
    MJD_list = []
    mag_err_list = []
    for k in range(len(mag_new)):
        if Instrument[k] == instrument:
            mag_list.append(mag_new[k])
            MJD_list.append(MJD_new[k])
            mag_err_list.append(magerr_new[k])
        else:
            pass
        
    return MJD_list, mag_list, mag_err_list



# create LC model
#fit function for Supernovae model
def rise(time, A, B, t0, Trise):
    """
    Parameters:
    
    time == t(MJD)
    
    A == Amplitude
    
    B == Beta
    
    t0 == t0
    
    Trise == tau_rise 
    
    t < t1(platue onset)
    """
    return (A + B * (time - t0)) / ( 1 + np.exp((-(time - t0))/ Trise))

def fall(time, A, B, t0, Trise, Tfall, t1):
    """
    Parameters:
    
    x == t(MJD)
    
    A == Amplitude
    
    B == Beta
    
    t0 == t0

    ti == t1 
    
    Trise== tau_rise 
    
    Tfall == tau_fall 
    
    x >= t1
    """
    return ((A + B * (t1 - t0)) * np.exp((-(time - t1))/Tfall)) / (1 + np.exp((-(time - t0)) / Trise))

def SN_LC(time, A, B, t0, Trise, Tfall, t1, zero_points):
    para_rise = np.array([ A, B, t0, Trise])
    para_fall = np.array([ A, B, t0, Trise, Tfall, t1])
    flux_rise = rise(time, *para_rise)
    flux_fall = fall(time, *para_fall)
    index_change = np.where(time < t1) 
    flux = np.array([flux_rise[:index_change], flux_fall[index_change:]]) # index_change is not an integer
    return -2.5 * np.log10(flux) + zero_points # return to the expected Magitude base on the fit function
    



# test for  in gp filter)
target = "2018agk"
file_location = file_name(target)
original_data = data(file_location,'gp', 'Sinistro')
#print(original_data[0])
xsmooth1 = np.linspace(np.min(original_data[0]), np.max(original_data[0]), 1000)
#print(xsmooth1)


A = 3 
B = 1 
Trise = 20 
Tfall = 120
t0 = 58186
t1 = 58202
zero_points = 0 
fsmooth1 = SN_LC(xsmooth1, A, B, t0, Trise, Tfall, t1, zero_points)