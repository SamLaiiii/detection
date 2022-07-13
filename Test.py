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
import species

# create a function to locate the fle data 
def file_name(target):
    return "/Users/eddie_tang/Desktop/Photometry file/{target}_data.snana.txt".format(target = target) #file locaiton 


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
#    for i in range(len(Filter)):
    return MJD_list, mag_list, mag_err_list

def multi(filters, Instrument, file_location):
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

   
def curvefitting_and_plot(fitfunction, x, x1,  y, y1, dy, dy1, guessparam, xrange= [-1, 1], yrange= [-1, 1], make_image = 0, target_and_filter_inst = 'target' ) :
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

#    plt.errorbar(x,y,dy,fmt = '.', ecolor = 'blue')
#    plt.gca().invert_yaxis()
    xsmooth1 = np.linspace(np.min(x), np.max(x), len(x))
    fsmooth1 = fitfunction(xsmooth1 , *guessparam)
    plt.plot(xsmooth1, fsmooth1, color = 'red')

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
    xsmooth2 = np.linspace(np.min(x1), np.max(x1), len(x1))
    popt, pcov = opt.curve_fit(fitfunction, x, y, p0 = guessparam, sigma= dy)
    popt1, pcov1 = opt.curve_fit(fitfunction, x1, y1, p0 = guessparam, sigma= dy1)
    for i in range(len(popt)):
        print('para',i,'=',popt[i],'+/-',np.sqrt(pcov[i,i]))

    fsmooth2 = fitfunction(xsmooth1, *popt)
    fsmooth3 = fitfunction(xsmooth2, *popt1)
    plt.plot(xsmooth1, fsmooth2, color = 'orange', label = 'fit: A = %5.3f, B = %5.3f, t0= %5.3f, t1= %5.3f, Trise =%5.3f, Tfall = %5.3f, c = %5.3f' % tuple(popt))
    plt.plot(xsmooth2, fsmooth3, color = 'orange', label = 'fit: A = %5.3f, B = %5.3f, t0= %5.3f, t1= %5.3f, Trise =%5.3f, Tfall = %5.3f, c = %5.3f' % tuple(popt1))##
    plt.legend()
    Expect = fitfunction(x, *popt)
    Expect1 = fitfunction(x1, *popt1)
    score = []
    for i in range(len(Expect)):
        score.append((Expect[i]- y[i])**2/(dy[i])**2)
    score1 = [] 
    for i in range(len(Expect1)):
        score1.append((Expect1[i]- y1[i])**2/(dy1[i])**2)
    plt.scatter(x, y, c = score, cmap = 'summer')
    plt.scatter(x1, y1, c = score1, cmap = 'summer')
    plt.colorbar(label = 'least anomalous / the most anomalous', orientation="horizontal")
    norm = matplotlib.colors.Normalize(vmin = min(y), vmax = max(y), clip = True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
    x_color = np.array([(mapper.to_rgba(v)) for v in y])
    x_color1 = np.array([(mapper.to_rgba(v)) for v in y1])
    for x, y, e, color in zip(x, y, dy, x_color):
        plt.plot(x, y, 'o', color=color)
        plt.errorbar(x, y, e, lw=1, capsize=3, color=color)
    
    for x1, y1, e, color in zip(x1, y1, dy1, x_color1):
        plt.plot(x1, y1, 'o', color=color)
        plt.errorbar(x1, y1, e, lw=1, capsize=3, color=color)
#    plt.errorbar(x,y,yerr=dy, marker=None, mew=0)
    plt.legend()
    plt.gca().invert_yaxis()
#    plt.scatter(x , y)
#    plt.errorbar(x , y, dy, ls='none', fmt ='.')
    
    if make_image > 0:
        plt.savefig('Test.png', format = 'png')
    plt.show()











def multi_curvefit(fitfunction, x, y, dy, title, filter_name):

    plt.rcParams["figure.figsize"] = (24,8)  
    plt.xlabel("MJD")
    plt.ylabel("Flux") 
    plt.title(title)   
    for i in range(len(filter_name)):
#        x[i] = x[i] - np.mean(x[i])
        number_of_colors = len(filter_name)
        Colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
        ### guessparameter
        # A = np.max(y[i]) - np.min(y[i])
        # B = (np.max(y[i]) - y[i][np.argmax(x[i])])/ (x[i][np.argmax(y[i])] - np.max(x[i]))
        # t0 = np.min(x[i])
        # t1 = x[i][np.argmax(y[i])]
        # Trise = x[i][np.argmax(y[i])] - np.min(x[i])
        # Tfall = np.max(x[i]) - x[i][np.argmax(y[i])]
        # c = np.min(y[i])
        plt.errorbar(x[i],y[i],dy[i],fmt = '.', ecolor = 'red')

        #g:
        # A = 5.22e-6
        # B = 5.987e-7
        # t0 = 58204.34
        # t1 = 58206.19
        # Trise = 31
        # Tfall = 10.54
        # c = 1.306e-6


        #r:
        A = 6.576e-6
        B = 5.91e-7
        t0 = 58200.47
        t1 = 58201.58
        Trise = 5.79
        Tfall = 16.73
        c = 1.32e-07
        guessparam = np.array([A, B, t0, t1, Trise, Tfall,c])
        xsmooth1 = np.linspace(np.min(x[i]), np.max(x[i]), 100000)
        fsmooth1 = fitfunction(xsmooth1 , *guessparam)
        plt.plot(xsmooth1, fsmooth1, color = 'red')


        #boundary 
        initial_bounds = [3 * np.min(dy[i]), 0, 
        np.min(x[i]) - 50, np.min(x[i]) - 45, 0.01, 1, -3 * np.min(dy[i])]
        final_bounds = [100 * np.max(y[i]), (np.max(y[i])/150), np.max(x[i])+ 100, np.max(x[i])+160,
        50, 300, 3 * np.max(dy[i])]

        popt, pcov = opt.curve_fit(fitfunction, x[i], y[i],sigma=dy[i], p0 = guessparam ) #bounds = (initial_bounds, final_bounds)
        # for i in range(len(popt)):
        #     print('para',i,'=',popt[i],'+/-',np.sqrt(pcov[i,i]))
        
        fsmooth = fitfunction(xsmooth1, *popt)
#       plt.plot(xsmooth1, fsmooth, c ='orange', label ='fit: A = %5.3f, B = %5.3f, t0= %5.3f, t1= %5.3f, Trise =%5.3f, Tfall = %5.3f, c = %5.3f' % tuple(popt))#filter_name[i]
        plt.legend()
        # Expect = fitfunction(x[i], *popt)
        # score = []
        # for j in range(len(Expect)):
        #     score.append((Expect[j]- y[i][j])**2/(dy[i][j])**2)
        # plt.scatter(x[i], y[i], c = score, cmap = 'summer')
        # norm = matplotlib.colors.Normalize(vmin = min(score), vmax = max(score), clip = True)
        # mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        # x_color = np.array([(mapper.to_rgba(v)) for v in y[i]])     
        # plt.errorbar(x[i],y[i],yerr=dy[i], linestyle="None", color= Colors[i])
        # for x, y, e, color in zip(x[i], y[i], dy[i], x_color):
        #     plt.plot(x, y, 'o', color=color)
        #     plt.errorbar(x, y, e, lw=1, capsize=3, color=color)    
    # plt.colorbar(label = 'least anomalous / the most anomalous', orientation="horizontal")
    plt.show()



def multi_curvefit_test(fitfunction, x, y, dy, title, filter_name):

    plt.rcParams["figure.figsize"] = (24,12) 
    plt.xlabel("MJD")
    plt.ylabel("Flux") 
    plt.title(title)   
    for i in range(len(filter_name)):
#        x[i] = x[i] - np.mean(x[i])
        number_of_colors = len(filter_name)
        Colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
        ### guessparameter
        # A = np.max(y[i]) - np.min(y[i])
        # B = (np.max(y[i]) - y[i][np.argmax(x[i])])/ (x[i][np.argmax(y[i])] - np.max(x[i]))
        # t0 = np.min(x[i])
        # t1 = x[i][np.argmax(y[i])]
        # Trise = x[i][np.argmax(y[i])] - np.min(x[i])
        # Tfall = np.max(x[i]) - x[i][np.argmax(y[i])]
        # c = np.min(y[i])

        #boundary 
        # A == (3 * sigma(np.min(dy[i])) , 100 * F(np.max(y[i])))
        initial_bounds = [3 * np.min(dy[i]), 0, 
        np.min(x[i]) - 50, np.min(x[i]) - 45, 0.01, 1, -3 * np.min(dy[i])]
        final_bounds = [100 * np.max(y[i]), (np.max(y[i])/150), np.max(x[i])+ 100, np.max(x[i])+160,
        50, 300, 3 * np.max(dy[i])]
        A = 6.576e-6
        B = 5.91e-7
        t0 = 58200.47
        t1 = 58201.58
        Trise = 5.79
        Tfall = 16.73
        c = 1.32e-07
        guessparam = np.array([A, B, t0, t1, Trise, Tfall,c])
        xsmooth1 = np.linspace(np.min(x[i]), np.max(x[i]), 100000)
        fsmooth1 = fitfunction(xsmooth1 , *guessparam)
        plt.plot(xsmooth1, fsmooth1, color = 'red')       
        popt, pcov = opt.curve_fit(fitfunction, x[i], y[i],sigma=dy[i], p0=guessparam) #bounds = (initial_bounds, final_bounds)
        # for i in range(len(popt)):
        #     print('para',i,'=',popt[i],'+/-',np.sqrt(pcov[i,i]))        
        # fsmooth = fitfunction(xsmooth1, *popt)
        # plt.plot(xsmooth1, fsmooth, c = Colors[i], label = filter_name[i])#filter_name[i]
        plt.legend()
        Expect = fitfunction(x[i], *guessparam)
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