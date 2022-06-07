import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat


# create a function to locate the fle data 
def file_name(target):
    return "/Users/eddie_tang/Desktop/{target}_data.snana.txt".format(target = target) #file locaiton 


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
    #mag = mag[MJD.argsort()[::]]
    #MJD = MJD[MJD.argsort()[::]]
    #magerr = magerr[MJD.argsort()[::]]

#    print(mag)                        # to show all the mag 
#    print(unique_filter)              # to show what kind of filters are there
#    print(unique_Instrument)          # to show the Instrument
    
    mag_new = [] 
    MJD_new = []
    magerr_new = []
    flux_new = []
    fluxerr_new = []
    for index in range(len(mag)):
        if mag[index] >= 0 and fluxcal[index] != fluxcalerr[index] and fluxcal[index] > fluxcalerr[index] and magerr[index] > 0:
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
        if Instrument[k] == instrument and filter[k] == Filter:
            mag_list.append(mag_new[k])
            MJD_list.append(MJD_new[k])
            mag_err_list.append(magerr_new[k])
        else:
            pass
    
    return MJD_list, mag_list, mag_err_list





# create LC model
#fit function for Supernovae model

def rise1(time, A, B, t0, t1, Trise):
    """
    Parameters:
    time == t(MJD) in array
    A == Amplitude
    B == Beta
    t0 == t0
    Trise == tau_rise 
    condition: t < t1(platue onset)
    """
    rise = []
    for i in range(len(time)):
        if time[i] < t1:
            rise.append((A + B * (time[i]- t0)) / ( 1 + np.exp((-(time[i] - t0))/ Trise)))
        else:
            pass   
    return rise
#def rise(time, A, B, t0, t1, Trise):
    """ I thought that is not correct for t < t1
    """
#    return (A + B * (time- t0)) / ( 1 + np.exp((-(time - t0))/ Trise))



def fall(time, A, B, t0, t1, Trise, Tfall):
    """
    Parameters:
    x == t(MJD)
    A == Amplitude
    B == Beta (Plateau slope)
    t0 == t0 (Plateau start)
    ti == t1 (Plateau end)
    Trise== tau_rise 
    Tfall == tau_fall 
    x >= t1
    """
    fall = [] 
    for i in range(len(time)):
        if time[i] >= t1:
            fall.append(((A + B * (t1 - t0)) * np.exp((-(time[i] - t1))/Tfall)) / (1 + np.exp((-(time[i] - t0)) / Trise))) 
        else:
            pass

    return fall



def SN_LC(time, A, B, t0, t1,Trise, Tfall,c):
    para_rise = np.array([ A, B, t0, t1, Trise])
    para_fall = np.array([ A, B, t0, t1, Trise, Tfall])
    flux_rise = rise1(time, *para_rise)
    flux_fall = fall(time, *para_fall)
    flux = np.concatenate((flux_rise, flux_fall))
    result = -2.5 * np.log10(flux/27.5) +c #+17.5  # return to the expected Magitude base on the fit function


    return result
# plot the amplitude for testing

   
def curvefitting_and_plot(fitfunction, x, y, dy, guessparam, xrange= [-1, 1], yrange= [-1, 1], make_image = 0, target_and_filter_inst = 'target') :
    """
    x , y, dy == real data(should be an numpy array)

    fitfunction == SN_LC model 
    """
    #x = x - np.mean(x)
    #y = y - np.mean(y)
    plt.rcParams["figure.figsize"] = (12,4)  
    plt.xlabel("MJD")
    plt.ylabel("Mag") 
    plt.title(target_and_filter_inst)
    if yrange == [-1,1]:
        yspan = max(y) - min(y)
        yrange = [min(y) - yspan/10, max(y) + yspan/10]

    if xrange == [-1,1]:
        xspan = max(x) - min(x)
        xrange = [min(x) - xspan/10, max(x) + xspan/10]

    plt.gca().invert_yaxis()
    plt.scatter(x , y)
    plt.errorbar(x , y, dy, ls='none', fmt ='.')
    xsmooth1 = np.linspace(np.min(x), np.max(x), len(x))
 #   fsmooth1 = fitfunction(xsmooth1 , *guessparam)
 #   plt.plot(xsmooth1, fsmooth1, color = 'red')

    ######guessparam for the scipy.curve_fit:
    A = (np.max(y)-np.min(y) + (np.max(y)-np.min(y))/2)
    B = ( - ((np.min(y) - np.max(y)) / (x[np.argmin(y)] - x[np.argmax(y)])) - 1)
    t0 = (x[np.argmax(y)])# x[np.argmin(y)] + 10
    t1 = (x[np.argmax(y)] + 10)
    Trise = (x[np.argmin(y)] - np.min(x) + (x[np.argmin(y)] - np.min(x))/2)
    Tfall = (np.max(x) - x[np.argmin(y)] + (np.max(x) - x[np.argmin(y)])/2)
    ######

    # t1 >> 0 
    #bounds_sequence = (A, B, t0, t1,Trise, Tfall)
    init_bounds = (0 ,  - ((np.min(y) - np.max(y)) / (x[np.argmin(y)] - x[np.argmax(y)])) - 1, np.min(x)-5,x[np.argmin(y)], 0, 0)
    fina_bounds = (np.max(y)-np.min(y) + (np.max(y)-np.min(y))/2, 0,
    x[np.argmin(y)] + 10, np.max(x), 
    x[np.argmin(y)] - np.min(x) + (x[np.argmin(y)] - np.min(x))/2,
     np.max(x) - x[np.argmin(y)] + (np.max(x) - x[np.argmin(y)])/2)
    
    
    
    xsmooth1 = np.linspace(np.min(x), np.max(x), len(x))
    popt, pcov = opt.curve_fit(fitfunction, x, y, p0 = guessparam, sigma= dy)

    for i in range(len(popt)):
        print('para',i,'=',popt[i],'+/-',np.sqrt(pcov[i,i]))

    fsmooth2 = fitfunction(xsmooth1, *popt)
    plt.plot(xsmooth1, fsmooth2, color = 'orange', label = 'fit: A = %5.3f, B = %5.3f, t0= %5.3f, t1= %5.3f, Trise =%5.3f, Tfall = %5.3f, c = %5.3f' % tuple(popt))
    plt.legend()

    
    if make_image > 0:
        plt.savefig('Test.png', format = 'png')
    plt.show()

#distance for best fit  for model line / distance of the error bar

