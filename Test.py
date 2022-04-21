import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat
import randomcolor

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
        if Instrument[k] == instrument:
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
    rise =[]
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
def SN_LC(time, A, B, t0, t1,Trise, Tfall, zero_points):
    para_rise = np.array([ A, B, t0, t1, Trise])
    para_fall = np.array([ A, B, t0, t1, Trise, Tfall])
    flux_rise = rise1(time, *para_rise)
    flux_fall = fall(time, *para_fall)
    flux = np.concatenate((flux_rise, flux_fall))
    result = (-2.5 * np.log10(flux) + zero_points) # return to the expected Magitude base on the fit function
    return result
# plot the amplitude for testing

def plot_Amp(counts, x):
    """counts = number of plots

        x = MJD in array  
    """
    rand_color = randomcolor.RandomColor()
    xsmooth = np.linspace(np.min(x), np.max(x), 1000)
    plt.figure(figsize = (8,6))
    for i in range(counts):
        fsmooth = SN_LC(xsmooth, A = i * 0.1, B = 1, t0 = 58186, t1 = 58190, Trise = 15, Tfall = 0, zero_points = 20)
        plt.plot(xsmooth, fsmooth, color = rand_color.generate(count = counts)[i], label = 'A = {x}'.format(x = i* 0))
    plt.title('change in A(amplitude)')
    plt.gca().invert_yaxis()
    return plt.legend(loc =(1, 0.5))



def plot_Beta(counts, x):
    """counts = number of plots

        x = MJD in array  
    """
    rand_color = randomcolor.RandomColor()
    xsmooth = np.linspace(np.min(x), np.max(x), 1000)
    plt.figure(figsize = (8,6))
    for i in range(counts):
        fsmooth = SN_LC(xsmooth, A = 1, B = -i * 0.001, t0 = 58186, t1 = 58202, Trise = 120, Tfall = 120, zero_points = 20)
        plt.plot(xsmooth, fsmooth, color = rand_color.generate(count = counts)[i], label = 'B = {x}'.format(x = - i * 0.001))
    plt.title('change in B(Beta)')
    plt.gca().invert_yaxis()
    return plt.legend(loc =(1, 0.5))





def plot_t0(counts, x):
    """counts = number of plots

        x = MJD in array  
    """
    rand_color = randomcolor.RandomColor()
    xsmooth = np.linspace(np.min(x), np.max(x), 1000)
    plt.figure(figsize = (8,6))
    for i in range(counts):
        fsmooth = SN_LC(xsmooth, A = 5, B = 1, t0 = 58186 - 100 * i , t1 = 58202, Trise = 120, Tfall = 120, zero_points = 20)
        plt.plot(xsmooth, fsmooth, color = rand_color.generate(count = counts)[i], label = 't0 = {x}'.format(x = 58186 - 100 * i))
    plt.title('change in t0(start time)')
    plt.gca().invert_yaxis()
    return plt.legend(loc =(1, 0.5))

def plot_t1(counts, x):
    """counts = number of plots

        x = MJD in array  
    """
    rand_color = randomcolor.RandomColor()
    xsmooth = np.linspace(np.min(x), np.max(x), 1000)
    plt.figure(figsize = (8,6))
    for i in range(counts):
        fsmooth = SN_LC(xsmooth, A = 5, B = 1, t0 = 58186  , t1 = 58202  + 10 * i, Trise = 120, Tfall = 120, zero_points = 20)
        plt.plot(xsmooth, fsmooth, color = rand_color.generate(count = counts)[i], label = 't1 = {x}'.format(x = 58186 + 10 * i))
    plt.title('change in t1(end time)')
    plt.gca().invert_yaxis()
    return plt.legend(loc =(1, 0.5))

def plot_Trise(counts, x):
    """counts = number of plots

        x = MJD in array  
    """
    rand_color = randomcolor.RandomColor()
    xsmooth = np.linspace(np.min(x), np.max(x), 1000)
    plt.figure(figsize = (8,6))
    for i in range(counts):
        fsmooth = SN_LC(xsmooth, A = 5, B = 1, t0 = 58186  , t1 = 58202, Trise = 120 - 10 *i , Tfall = 120, zero_points = 20)
        plt.plot(xsmooth, fsmooth, color = rand_color.generate(count = counts)[i], label = 'Trise = {x}'.format(x = 120 - 10 * i))
    plt.title('change in Trise(rising time)')
    plt.gca().invert_yaxis()
    return plt.legend(loc =(1, 0.5))

def plot_Tfall(counts, x):
    """counts = number of plots

        x = MJD in array  
    """
    rand_color = randomcolor.RandomColor()
    xsmooth = np.linspace(np.min(x), np.max(x), 1000)
    plt.figure(figsize = (8,6))
    for i in range(counts):
        fsmooth = SN_LC(xsmooth, A = 5, B = 1, t0 = 58186  , t1 = 58202, Trise = 120  , Tfall = 120 + 10 * i, zero_points = 20)
        plt.plot(xsmooth, fsmooth, color = rand_color.generate(count = counts)[i], label = 'Tfall= {x}'.format(x = 120 + 10 * i))
    plt.title('change in Tfall(falling time)')
    plt.gca().invert_yaxis()
    return plt.legend(loc =(1, 0.5))


    

def curvefitting_and_plot(fitfunction, x, y, dy, guessparams, xrange= [-1, 1], yrange= [-1, 1], make_image = 0, target_and_filter_inst = 'target') :
    """
    x , y, dy == real data

    fitfunction == SN_LC model 
    """
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
    #fsmooth1 = fitfunction(xsmooth1 , *guessparams)
    #plt.plot(xsmooth1, fsmooth1, color = 'red')

    popt, pcov = opt.curve_fit(fitfunction, x, y, p0 = guessparams)
    for i in range(len(popt)):
        print('para',i,'=',popt[i],'+/-',np.sqrt(pcov[i,i]))

    fsmooth2 = fitfunction(xsmooth1, *popt)
    plt.plot(xsmooth1, fsmooth2, color = 'orange', label = 'fit: A = %5.3f, B = %5.3f, t0= %5.3f, t1= %5.3f, Trise =%5.3f, Tfall = %5.3f, Zero_points = %5.3f' % tuple(popt))
    plt.legend()

    
    if make_image > 0:
        plt.savefig('Test.png', format = 'png')
    plt.show()


