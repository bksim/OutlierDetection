import numpy as np
import scipy.optimize as optimize
import os

from import_lightcurve import ReadLC_MACHO
from Feature import FeatureSpace
from alignLC import Align_LC
from PreprocessLC import Preprocess_LC
import lomb

import argparse

##### MODEL #####
#1) For i = {1, 2, 3}:

#2) Calculate Lomb-Scargle periodogram $P_f(f)$ for light curve.

#3) Find peak in $P_f(f)$, subtract that model from data.

#4) Update $\chi_{\circ}^2$, return to Step 1.

#Then, the features extracted are given as an amplitude and a phase:
#A_{i,j} &= \sqrt{a_{i,j}^2 + b_{i,j}^2}\\
#\textrm{PH}_{i,j} &= \arctan\left(\frac{b_{i,j}}{a_{i,j}}\right)
def calculate_periodic_features(mjd2, data2):
    A = []
    PH = []
    
    def model(x, a, b, c, freq):
         return a*np.sin(2*np.pi*freq*x)+b*np.cos(2*np.pi*freq*x)+c
        
    for i in range(3):
        wk1, wk2, nout, jmax, prob = lomb.fasper(mjd2, data2, 6., 100.)
    
        fundamental_freq = wk1[jmax]
        
        # fit to a_i sin(2pi f_i t) + b_i cos(2 pi f_i t) + b_i,o
        
        # a, b are the parameters we care about
        # c is a constant offset
        # f is the fundamental frequency
        def yfunc(freq):
            def func(x, a, b, c):
                return a*np.sin(2*np.pi*freq*x)+b*np.cos(2*np.pi*freq*x)+c
            return func
        
        Atemp = []
        PHtemp = []
        popts = []
        
        for j in range(4):
            popt, pcov = optimize.curve_fit(yfunc((j+1)*fundamental_freq), mjd2, data2)
            Atemp.append(np.sqrt(popt[0]**2+popt[1]**2))
            PHtemp.append(np.arctan(popt[1] / popt[0]))
            popts.append(popt)
        
        A.append(Atemp)
        PH.append(PHtemp)

        for j in range(4):
            data2 = np.array(data2) - model(mjd2, popts[j][0], popts[j][1], popts[j][2], (j+1)*fundamental_freq)
    
    scaledPH = []
    for ph in PH:
        scaledPH.append(np.array(ph) - ph[0])

    return A, PH, scaledPH

def calculate_features(lc_fn, feature_list):
    lc = ReadLC_MACHO(lc_fn)
    [data, mjd, error] = lc.ReadLC()
    preprocessed_data = Preprocess_LC(data, mjd, error)
    
    fs = FeatureSpace(featureList=feature_list,
                        Automean=[0,0],
                        #Beyond1Std=[np.array(error)],
                        CAR_sigma=[mjd, error],
                        Eta_e=mjd,
                        LinearTrend=mjd,
                        MaxSlope=mjd,
                        PeriodLS=mjd,
                        Psi_CS=mjd
                        )
    values = fs.calculateFeature(data)
    value_dict = values.result(method='dict')
    A, PH, scaledPH = calculate_periodic_features(mjd, data)
    
    for i in range(len(A)):
        for j in range(len(A[i])):
            value_dict['freq'+str(i+1)+'_harmonics_amplitude_'+str(j)] = A[i][j]
            value_dict['freq'+str(i+1)+'_harmonics_rel_phase_'+str(j)] = scaledPH[i][j]
    return value_dict

if __name__ == "__main__":
	# parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default='I', type=str, help="path of the data")
    parser.add_argument("-o", "--out", default='features_I.csv', type=str, help="output file location")
    parser.add_argument("-s", "--start", default=0, type=int, help="start curve inclusive")
    parser.add_argument("-e", "--end", default=0, type=int, help="end curve not inclusive")

    args = parser.parse_args()
    path=args.path
    outfile=args.out
    start=args.start
    end=args.end
    featureList=['Amplitude',
                'AndersonDarling',
                'Automean',
                'Autocor_length',
                #'Beyond1Std', #this feature doesn't work
                'CAR_sigma',
                'CAR_tmean',
                'CAR_tau',
                'Con',
                'Eta_e',
                'FluxPercentileRatioMid20',
                'FluxPercentileRatioMid35',
                'FluxPercentileRatioMid50',
                'FluxPercentileRatioMid65',
                'FluxPercentileRatioMid80',
                'LinearTrend',
                'MaxSlope',
                'Mean',
                'Meanvariance',
                'MedianAbsDev',
                'MedianBRP',
                'PairSlopeTrend',
                'PercentAmplitude',
                'PercentDifferenceFluxPercentile',
                'PeriodLS',
                'Period_fit',
                'Psi_CS',
                'Psi_eta',
                'Q31',
                'Rcs',
                'Skew',
                'SmallKurtosis',
                'Std',
                'StetsonK'
                ]

    fullfeaturelist = ['Amplitude', 'AndersonDarling', 'Autocor_length', 'Automean', 'CAR_sigma', 'CAR_tau', 'CAR_tmean', 'Con', 'Eta_e', 'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 'LinearTrend', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'PercentDifferenceFluxPercentile', 'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 'Q31', 'Rcs', 'Skew', 'SmallKurtosis', 'Std', 'StetsonK', 'freq1_harmonics_amplitude_0', 'freq1_harmonics_amplitude_1', 'freq1_harmonics_amplitude_2', 'freq1_harmonics_amplitude_3', 'freq1_harmonics_rel_phase_0', 'freq1_harmonics_rel_phase_1', 'freq1_harmonics_rel_phase_2', 'freq1_harmonics_rel_phase_3', 'freq2_harmonics_amplitude_0', 'freq2_harmonics_amplitude_1', 'freq2_harmonics_amplitude_2', 'freq2_harmonics_amplitude_3', 'freq2_harmonics_rel_phase_0', 'freq2_harmonics_rel_phase_1', 'freq2_harmonics_rel_phase_2', 'freq2_harmonics_rel_phase_3', 'freq3_harmonics_amplitude_0', 'freq3_harmonics_amplitude_1', 'freq3_harmonics_amplitude_2', 'freq3_harmonics_amplitude_3', 'freq3_harmonics_rel_phase_0', 'freq3_harmonics_rel_phase_1', 'freq3_harmonics_rel_phase_2', 'freq3_harmonics_rel_phase_3']
    with open(outfile, 'wb') as fout:
        fout.write(",".join(['LC'] + fullfeaturelist) + '\n')
        for fn in os.listdir(path):
            filenumber = float(fn.split('-')[3].split('.')[0])
            if filenumber>=start and filenumber<end:
                features_temp = calculate_features(path+'/'+fn, featureList)
                vals = []
                for k in fullfeaturelist:
                    vals.append(features_temp[k])
                fout.write(",".join([fn.split('.')[0]]+[str(v) for v in vals])+'\n')