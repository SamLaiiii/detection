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
from lmfit import Minimizer, Parameters, report_fit, Model


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
    for i in range(np.size(time)):
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
    for i in range(np.size(time)):
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

from scipy.optimize import differential_evolution

def Chi_square(Observation, Expectation, Observation_error):
    """Observation --> flux
    Expectation --> y values of curvefitiing 
    Observation_error --> fluxerror
    """
    chi_square = np.array([])
    for j in range(len(Observation)):
        chisq = (Observation[j] - Expectation[j])**2 / (Observation_error[j]**2)
        chi_square = np.append(chi_square, chisq)
    
    return np.sum(chi_square)


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
        initial_bounds = [3 * np.std(y[i]), -(np.max(y[i])/150), 
        np.min(x[i]) - 50, np.min(x[i]) - 45, 0.01, 1, -3 * np.std(y[i])]
        final_bounds = [100 * np.max(y[i]), 0, np.max(x[i])+ 100, np.max(x[i])+160,
        50, 300, 3 * np.std(y[i])]
        A_bounds = (3 * np.std(y[i]), 100 * np.max(y[i]))
        B_bounds = (-(np.max(y[i])/150), 0)
        t0_bounds = (np.min(x[i]) - 50, np.min(x[i])+ 100)
        t1_bounds = (np.max(x[i]) - 45, np.max(x[i])+160)
        Trise_bounds = (0.01, 50)
        Tfall_bounds  = (1, 300)
        c_bounds = (-3 * np.std(y[i]), 3 * np.std(y[i]))

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
        # print(x[i])
        # print(y[i])

        """chi_sqaure test for the best curve fitting"""
        Observation = y[i]
        Expected_red = fitfunction(x[i], *guessparam)
        chi_square_red = Chi_square(y[i], Expected_red,  dy[i])
    
        plt.plot(xsmooth1, fsmooth1, color = 'blue', label = 'best: chisquare = {chi_square}'.format(chi_square = chi_square_red))


        """chi_sqaure test the scipy curve fitting"""
        popt, pcov = opt.curve_fit(fitfunction, x[i], y[i], sigma=dy[i], bounds = (initial_bounds, final_bounds)) #bounds = (initial_bounds, final_bounds)
        # for i in range(len(popt)):
        #     print('para',i,'=',popt[i])        
        fsmooth = fitfunction(xsmooth1, *popt)
        Expected_scipy = fitfunction(x[i], *popt)
        chi_square_scipy = Chi_square(y[i], Expected_scipy, dy[i])
        minimized_result = differential_evolution(chi_square_scipy, bounds = [A_bounds,B_bounds,t0_bounds,t1_bounds,Trise_bounds,Tfall_bounds,c_bounds] )
        print('x:',minimized_result.x)
        print('fun:', minimized_result.fun)
        plt.plot(xsmooth1, fsmooth, c = Colors[i], label = 'SCIPY: {filter_name} chi square: {chi_square}'.format(filter_name = filter_name[i], chi_square=chi_square_scipy))#filter_name[i]
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
        # for x, y, e, color in zip(x[i], y[i], dy[i], x_color):
        #     plt.plot(x, y, 'o', color=color)
        #     plt.errorbar(x, y, e, lw=1, capsize=3, color=color)    
    plt.show()



from scipy.optimize import differential_evolution
def chi_2(parameters, *data):
    """this function is for finding the least chi_square value"""
    A, B, t0, t1, Trise, Tfall,c = parameters
    x, y, dy = data
    Expectation = SN_LC(x, A , B, t0, t1, Trise, Tfall,c)
    Observation = y
    Observation_error = dy  
    chi_square = 0
    for j in range(np.size(Expectation)):
        chi_square += (Observation[j] - Expectation[j])**2 / (Observation_error[j]**2)
    return chi_square

def find_lq(model, bounds, x, y, dy):
    """use it for find_lq"""
    args = (x, y, dy)
    result = differential_evolution(model, bounds = bounds, args = args)
    return result.x

from lmfit import Minimizer, Parameters, report_fit
def limfit_test_function(fitfunction, x, y, dy, title, filter_name):
    """lmfit instead of scipy curvefitt"""

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
        """Normalized curvefitting"""
        x[i] = x[i] - np.mean(x[i])
        # for element in x[i]:

        model  = Model(fitfunction)
        parameters = model.make_params(A = np.max(y[i]) - np.min(y[i]), B = (np.max(y[i]) - y[i][np.argmax(x[i])])/ (x[i][np.argmax(y[i])] - np.max(x[i])), 
        t0 = np.min(x[i]), t1 = x[i][np.argmax(y[i])] , Trise = x[i][np.argmax(y[i])] - np.min(x[i]), Tfall = np.max(x[i]) - x[i][np.argmax(y[i])], c=np.min(y[i]))
        parameters['A'].set(min= 3 * np.std(y[i]), max= 100 * np.max(y[i]))
        parameters['B'].set(min= -(np.max(y[i]))/150, max= 0)
        parameters['t0'].set(min=np.min(x[i]) - 50 , max=np.max(x[i])+ 100)
        parameters['t1'].set(min=np.min(x[i]) -45 , max= np.max(x[i]) + 160)
        parameters['Trise'].set(min= 0.01 , max= 50)
        parameters['Tfall'].set(min=1, max =300)
        parameters['c'].set(min=-3 * np.std(y[i]), max = 3 * np.std(y[i]))
        result = model.fit(y[i],parameters, time = x[i], weights = 1/np.array(dy[i]), method = 'emcee')
        # print(result.fit_report())
        limfit_params = []
        for p in result.params:
            limfit_params.append(result.params[p].value)
        print(limfit_params)
        xsmooth1 = np.linspace(np.min(x[i]), np.max(x[i]), 100)
        data = [x[i], y[i], dy[i]]
        chi_square_lmfit = chi_2(limfit_params, *data)
        fsmooth = fitfunction(xsmooth1, *limfit_params)
        plt.plot(xsmooth1, fsmooth, c = Colors[i],label = 'limfit: {filter_name} chi square: {chi_square}'.format(filter_name = filter_name[i], chi_square=chi_square_lmfit))
        plt.legend()
        Expect = fitfunction(x[i], *limfit_params)
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
        # for x, y, e, color in zip(x[i], y[i], dy[i], x_color):
        #     plt.plot(x, y, 'o', color=color)
        #     plt.errorbar(x, y, e, lw=1, capsize=3, color=color)    
    plt.show()

def differential_evolution_plots(fitfunction, x, y, dy, title, filter_name):
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
        x[i] = x[i] - np.mean(x[i])
        """tuple bounds for the diffential evolution"""
        A_bounds = (3 * np.std(y[i]), 100 * np.max(y[i]))
        B_bounds = (-(np.max(y[i])/150), 0)
        t0_bounds = (np.min(x[i]) - 50, np.max(x[i])+ 100)
        t1_bounds = (np.min(x[i]) - 45, np.max(x[i])+ 160)
        Trise_bounds = (0.01, 50)
        Tfall_bounds  = (1, 300)
        c_bounds = (-3 * np.std(y[i]), 3 * np.std(y[i]))
        tuple_bounds = [A_bounds,B_bounds,t0_bounds,t1_bounds,Trise_bounds,Tfall_bounds,c_bounds] 


        """specific guessparameter for curvefitting based on MCMC"""
        A = 6.493028363565406e-06
        B = -1.476303070705317e-08
        t0 =  -20.77896968598672
        t1 = -22.97180301849619
        Trise = 2.6828497715049924
        Tfall = 25.04881103498708
        c = -1.0613388894872917e-08
        guessparam = np.array([A, B, t0, t1, Trise, Tfall,c])
        xsmooth1 = np.linspace(np.min(x[i]), np.max(x[i]), len(x[i]))
        fsmooth1 = fitfunction(xsmooth1, *guessparam)#for ideal plot

        """chi_sqaure test for the best curve fitting"""
        data = [x[i], y[i], dy[i]]
        chi_square_red = chi_2(guessparam, *data)
        print('chi_square_red:', chi_square_red)
        plt.plot(xsmooth1, fsmooth1, color = 'red', label = 'best: chisquare = {chi_square}'.format(chi_square = chi_square_red))
        """chi_sqaure test the scipy curve fitting"""
        diffparam = find_lq(model = chi_2,  bounds = tuple_bounds, x = x[i], y=y[i], dy = dy[i])
        fsmooth = fitfunction(xsmooth1, *diffparam)
        chi_square_scipy = chi_2(diffparam, *data)
        print('chi_square_scipy:', chi_square_scipy)
        plt.plot(xsmooth1, fsmooth, c = Colors[i], label = 'SCIPY: {filter_name} chi square: {chi_square}'.format(filter_name = filter_name[i], chi_square=chi_square_scipy))#filter_name[i]
        plt.legend()

        """calculate the anamalous scores """
        Expect = fitfunction(x[i], *diffparam)
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
        # for x, y, e, color in zip(x[i], y[i], dy[i], x_color):
        #     plt.plot(x, y, 'o', color=color)
        #     plt.errorbar(x, y, e, lw=1, capsize=3, color=color)    
    plt.show()


