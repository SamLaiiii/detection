from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat
import matplotlib
import matplotlib.cm as cm
from  itertools import islice
import random
import species  # package for calculating the mag to flux
from scipy.stats import chisquare

# create a function to locate the fle data 
def file_name(target):
    return "/Users/eddie_tang/Desktop/Photometry file/{target}_data.snana.txt".format(target = target) #read the file locaiton 


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

def multi(filters, Instrument, file_location):#### This is the function that extract specific mag from the name of Instrument and filters 
    MJD = []
    Mag = []
    Magerr = []
    for i in range(len(filters)):
        all_data = data(file_location = file_location, Filter = filters[i]  , instrument = Instrument)
        MJD.append(all_data[0])
        Mag.append(all_data[1])
        Magerr.append(all_data[2])
    return MJD, Mag, Magerr


def mag_to_flux(mag, mag_err, Filter_ID): 
    """This is the function that changing the mag to flux and fluxerr"""
    synphot = species.SyntheticPhotometry(Filter_ID)
    flux = np.array([])
    flux_err = np.array([])
    for i in range(len(mag)):
        for j in range(len(mag[i])):
            Flux,error = synphot.magnitude_to_flux(mag[i][j], error=mag_err[i][j], zp_flux=27.5)
            flux = np.append(flux, Flux)
            flux_err = np.append(flux_err, error)

    input = flux 
    len_to_split = [] 
    for i in range(len(mag)):
        len_to_split.append(len(mag[i]))
    Inputt = iter(input)
    Output_flux = [list(islice(Inputt, elem))
                for elem in len_to_split]

    input1 = flux_err
    len_to_split = [] 
    for i in range(len(mag)):
        len_to_split.append(len(mag[i]))
    Inputt = iter(input1)
    Output_flux_err = [list(islice(Inputt, elem))
                for elem in len_to_split]    

    return Output_flux, Output_flux_err




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



def SN_LC(time, A, B, t0, t1,Trise, Tfall, c):
    para_rise = np.array([ A, B, t0, t1, Trise])
    para_fall = np.array([ A, B, t0, t1, Trise, Tfall])
    flux_rise = rise1(time, *para_rise)
    flux_fall = fall(time, *para_fall)
    flux = np.concatenate((flux_rise, flux_fall)) + c
    return flux
#    result = -2.5 * np.log10(flux/27.5) + c #+17.5  # return to the expected Magitude base on the fit function


    return result
# plot the amplitude for testing

# def multi_curvefit(fitfunction, x, y, dy, title, filter_name):
#     """Thi is the function to draw the plots if multiple filters in the same plots is neccessary"""
#     plt.rcParams["figure.figsize"] = (24,8)  
#     plt.xlabel("MJD")
#     plt.ylabel("Flux") 
#     plt.title(title)   
#     for i in range(len(filter_name)):
# #        x[i] = x[i] - np.mean(x[i])
#         number_of_colors = len(filter_name)
#         Colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#              for i in range(number_of_colors)]


#     """Guess parameters general guess """
#         ### guessparameter
#         # A = np.max(y[i]) - np.min(y[i])
#         # B = (np.max(y[i]) - y[i][np.argmax(x[i])])/ (x[i][np.argmax(y[i])] - np.max(x[i]))
#         # t0 = np.min(x[i])
#         # t1 = x[i][np.argmax(y[i])]
#         # Trise = x[i][np.argmax(y[i])] - np.min(x[i])
#         # Tfall = np.max(x[i]) - x[i][np.argmax(y[i])]
#         # c = np.min(y[i])

#         #g:
#         # A = 5.22e-6
#         # B = 5.987e-7
#         # t0 = 58204.34
#         # t1 = 58206.19
#         # Trise = 31
#         # Tfall = 10.54
#         # c = 1.306e-6


#         #r:
#         A = 8.016e-6
#         B = 5.44e-7
#         t0 = 58200.47
#         t1 = 58199.58
#         Trise = 5.53
#         Tfall = 17.3
#         c = 1.06e-07
#         guessparam = np.array([A, B, t0, t1, Trise, Tfall,c])
#         xsmooth1 = np.linspace(np.min(x[i]), np.max(x[i]), 100000)
#         fsmooth1 = fitfunction(xsmooth1 , *guessparam)
#         plt.plot(xsmooth1, fsmooth1, color = 'red')


#         #boundary 
#         initial_bounds = [3 * np.min(dy[i]), 0, 
#         np.min(x[i]) - 50, np.min(x[i]) - 45, 0.01, 1, -3 * np.min(dy[i])]
#         final_bounds = [100 * np.max(y[i]), (np.max(y[i])/150), np.max(x[i])+ 100, np.max(x[i])+160,
#         50, 300, 3 * np.max(dy[i])]

#         popt, pcov = opt.curve_fit(fitfunction, x[i], y[i],sigma=dy[i], p0 = guessparam ) #bounds = (initial_bounds, final_bounds)
#         for i in range(len(popt)):
#             print('para',i,'=',popt[i],'+/-',np.sqrt(pcov[i,i]))
        
#         fsmooth = fitfunction(xsmooth1, *popt)
#         plt.plot(xsmooth1, fsmooth, c ='orange', label ='fit: A = %5.3f, B = %5.3f, t0= %5.3f, t1= %5.3f, Trise =%5.3f, Tfall = %5.3f, c = %5.3f' % tuple(popt))#filter_name[i]
#         plt.legend()
#         # Expect = fitfunction(x[i], *popt)
#         # score = []
#         # for j in range(len(Expect)):
#         #     score.append((Expect[j]- y[i][j])**2/(dy[i][j])**2)
#         # plt.scatter(x[i], y[i], c = score, cmap = 'summer')
#         # norm = matplotlib.colors.Normalize(vmin = min(score), vmax = max(score), clip = True)
#         # mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
#         # x_color = np.array([(mapper.to_rgba(v)) for v in y[i]])     
#         # plt.errorbar(x[i],y[i],yerr=dy[i], linestyle="None", color= Colors[i])
#         # for x, y, e, color in zip(x[i], y[i], dy[i], x_color):
#         #     plt.plot(x, y, 'o', color=color)
#         #     plt.errorbar(x, y, e, lw=1, capsize=3, color=color)    
#     # plt.colorbar(label = 'least anomalous / the most anomalous', orientation="horizontal")
#     plt.show()



def multi_curvefit_test(fitfunction, x, y, dy, title, filter_name):
    """Thi is the function to draw the plots if multiple filters in the same plots is neccessary
    Also, this is a test function """
    plt.rcParams["figure.figsize"] = (24,12) 
    plt.xlabel("MJD")
    plt.ylabel("Flux") 
    plt.title(title)   
    for i in range(len(filter_name)):
        number_of_colors = len(filter_name)
        Colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
        """guessparameter for curve fitting"""
        # A = np.max(y[i]) - np.min(y[i])
        # B = (np.max(y[i]) - y[i][np.argmax(x[i])])/ (x[i][np.argmax(y[i])] - np.max(x[i]))
        # t0 = np.min(x[i])
        # t1 = x[i][np.argmax(y[i])]
        # Trise = x[i][np.argmax(y[i])] - np.min(x[i])
        # Tfall = np.max(x[i]) - x[i][np.argmax(y[i])]
        # c = np.min(y[i])

        """boundary for the curvefitting"""
        # A == (3 * sigma(np.min(dy[i])) , 100 * F(np.max(y[i])))
        initial_bounds = [3 * np.min(dy[i]), -(np.max(y[i])/150), 
        np.min(x[i]) - 50, np.min(x[i]) - 45, 0.01, 1, -3 * np.min(dy[i])]
        final_bounds = [100 * np.max(y[i]), 0, np.max(x[i])+ 100, np.max(x[i])+160,
        50, 300, 3 * np.max(dy[i])]

        """specific guessparameter for curvefitting"""
        A = 6.576e-6
        B = 5.91e-7
        t0 = 58200.47
        t1 = 58201.58
        Trise = 5.79
        Tfall = 16.73
        c = 2e-07
        guessparam = np.array([A, B, t0, t1, Trise, Tfall,c])
        xsmooth1 = np.linspace(np.min(x[i]), np.max(x[i]), len(x[i]))
        fsmooth1 = fitfunction(xsmooth1, *guessparam)#for plot

        """chi_sqaure test for the best curve fitting"""
        Observation = y[i]
        Expected_red = fitfunction(x[i], *guessparam)
        summation = np.array([])
        for j in range(len(Observation)):
            summation_red = (Observation[j] - Expected_red[j])**2 / Expected_red[j]
            summation = np.append(summation, summation_red)
        chisq = np.sum(summation)
        print('red:', chisq)
        plt.plot(xsmooth1, fsmooth1, color = 'blue', label = 'best: chisquare = {chi_square}'.format(chi_square = chisq))


        """chi_sqaure test the scipy curve fitting"""
        popt, pcov = opt.curve_fit(fitfunction, x[i], y[i],sigma=dy[i], bounds = (initial_bounds, final_bounds), method ='trf') #bounds = (initial_bounds, final_bounds)
        # for i in range(len(popt)):
        #     print('para',i,'=',popt[i])        
        fsmooth = fitfunction(xsmooth1, *popt)
        Expected_g = fitfunction(x[i], *popt)
        summation_g = []
        for j in range(len(y[i])):
            summation_of_g = ((Observation[j]- Expected_g[j])**2)/Expected_g[j]
            summation_g = np.append(summation_g ,summation_of_g)
        chi_square = np.sum(summation_g)     
        plt.plot(xsmooth1, fsmooth, c = Colors[i], label = 'SCIPY: {filter_name} chi square: {chi_square}'.format(filter_name = filter_name[i], chi_square=chi_square))#filter_name[i]
        plt.legend()

        """calculate the anamalous scores """
        Expect = fitfunction(x[i], *popt)
        score = []
        for j in range(len(Expect)):
            score.append((Expect[j]- y[i][j])**2/(dy[i][j])**2)
        Colors_for_cmap = ['spring', 'summer','autumn', 'winter','cool','Wistia']
        plt.scatter(x[i], y[i], c = score, cmap = Colors_for_cmap[i])
        norm = matplotlib.colors.Normalize(vmin = min(score), vmax = max(score), clip = True)
        mapper = cm.ScalarMappable(norm=norm, cmap=Colors_for_cmap[i])
        x_color = np.array([(mapper.to_rgba(v)) for v in y[i]])     
        plt.colorbar(label = '{filter} band :least anomalous / the most anomalous'.format(filter = filter_name[i]), orientation="horizontal")
        plt.errorbar(x[i],y[i],yerr=dy[i], linestyle='None', color= Colors[i])
        for x, y, e, color in zip(x[i], y[i], dy[i], x_color):
            plt.plot(x, y, 'o', color=color)
            plt.errorbar(x, y, e, lw=1, capsize=3, color=color)    

    plt.show()