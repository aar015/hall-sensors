import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import numpy as np

class Model(object):
    def __init__(self, name):
        self.name = name
        # Fetch Constants
        constants_file = os.path.join(os.getcwd(),'data',name,'constants.csv')
        constants_frame = pd.read_csv(constants_file)
        rh = constants_frame['R_H (m^2/C)'][0]
        n = constants_frame['N'][0]
        r = constants_frame['R (cm)'][0]*1e-2
        z = constants_frame['z (cm)'][0]*1e-2
        rm = constants_frame['R_m (ohms)'] [0]
        ls = constants_frame['L_s (H)'][0]
        lm = constants_frame['L_m (H)'][0]
        c = constants_frame['C (nF)'][0]*1e-9
        h = constants_frame['h_s (mm)'][0]*1e-3
        k = constants_frame['w_s (mm)'][0]*1e-3
        res = constants_frame['R_s (ohm)'][0]
        theta = math.atan(rh*500e-9/res)
        geo = 1-1.045*math.exp(-math.pi*h/k)*theta/math.tan(theta)
        alpha1 = constants_frame['Alpha 1'][0]
        alpha2 = constants_frame['Alpha 2'][0]
        self.constants = [lm, n, r, z, rm, ls, c, rh, geo, alpha1, alpha2]
        # Fetch Data
        data_file = os.path.join(os.getcwd(),'data',name,'data_summary.csv')
        data_frame = pd.read_csv(data_file)
        data_frame['I_bias (mA)'] = data_frame['I_bias (mA)']*1e-3
        self.data = data_frame[['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (mA)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']].values 
        # Initial Parameter Corrections
        parameters = np.zeros(9, dtype=float)
        # Hyperparameters
        # hyper_file = os.path.join(os.getcwd(),'data',name,'hyperparameters.csv')
        # hyper_frame = pd.read_csv(constants_file)
        lr = 1e5
        lz = 1e5
        lgeo = 1e3
        lrm = 45
        hyperparameters = [lr, lz, lrm, lgeo]
        # Fit
        num_trials = 10
        self.best = np.zeros(9, dtype=float)
        for i in range(num_trials):
            min = minimize(self.loss, parameters, args=(self.data, self.constants, hyperparameters), method='Powell')
            self.best += min.x
        self.best /= num_trials
       
    def mag_prediction(self, freq, vm, lm, n, r, z, rm, dlm, dr, dz, drm):
        ''' Calculate Applied Magnetic Field using parameters
        input:
            freq - Frequency of Field
            vm - Lock-In Voltage
            lm - Inductance of magnet
            n - Number of loops in magnet
            r - Radius of magnet
            z - Distance from magnet to sensor
            rm - Resistence of magnet (and lock-in)
            dlm - Correction to Inductance of Magnet
            dr - Correction to Radius of Magnet
            dz - Correcton of Distance from Magnet to Sensor
            drm - Correction of Resistence of Magnet
        output: Predicted Magnetic Field in nT'''
        mu = 1.25e-6 
        omega = 2 * math.pi * freq
        prediction = mu * n * (r + dr)**2 * vm * 1e9
        prediction /= 2**1.5
        prediction /= ((z + dz)**2 + (r + dr)**2)**1.5
        prediction /= ((rm + drm)**2 + ((lm + dlm) * omega)**2)**0.5
        return prediction
    
    def mag_measurment(self, freq, ib, vh, ls, c, rh, geo, dls, dc, dgeo):
        ''' Calculate Measured Magnetic Field given parameters
        input:
            freq - Frequency of Field
            ib - Bias Current
            vh - Hall Voltage
            ls - Inductance of sensor
            c - Capcitence of sensor
            rh - Hall Constant of sample
            geo - Geometric Factor
            dls - Correction to Inductance of Sensor
            dc - Correction to Capcitence of Sensor
            dgeo - Correction to Geometric Factor
        output: Measured Magnetic Field in nT'''
        omega = 2 * math.pi * freq
        measurement = vh * (1 + (6000 * omega * (c + dc))**2)**0.5 * 1e6
        measurement /= 2000 * (((geo + dgeo) * rh * ib)**2 + ((ls + dls) * omega)**2)**0.5
        return measurement

    def phase_prediction(self, freq, ib, lm, rm, ls, c, rh, geo, alpha1, alpha2, dlm,
                     drm, dls, dc, dgeo, dalpha1, dalpha2):
        ''' Calculate predicted phase given parameters
        input:
            freq - Frequency of Field
            ib - Bias Current
            lm - Inductance of magnet
            rm - Resistence of magnet (and lock-in)
            ls - Inductance of Sensor
            c - Capacitence of Sensor
            alpha1 - Phase Offset for positive bias current
            alpha2 - Phase Offset for negative bias current
            rh - Hall Constant of Sample
            geo - Geometric Factor
            dlm - Correction to Inductance of Magnet
            drm - Correction to Resistence of Magnet
            dls - Correction to Inductance of Sensor
            dc - Correction to Capacitence of Sensor
            dgeo - Correction to Geometric Factor
            dalpha1 - Correction to phase offset for positive bias current
            dalpha2 - Correction to phase offset for negative bias current
        output: Predicted Phase'''
        omega = 2 * math.pi * freq
        prediction = np.arctan((ls + dls) * omega / ((geo + dgeo) * rh * ib))
        prediction += np.arctan((lm + dlm) * omega / (rm + drm))
        prediction += np.arctan(6000 * omega * (c + dc))
        prediction *= -180 / math.pi
        prediction[ib>0] += alpha1 + dalpha1
        prediction[ib<0] += alpha2 + dalpha2
        return prediction

    def loss(self, parameters, data, constants, hyperparameters):
        '''Loss function for fit
        input:
            parameters-Corrections to constants based of fit 
                   [dlm, dr, dz, drm, dls, dc, dgeo, dalpha1, dalpha2]
            data-Data to plot
                ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit 
                  [lm, n, r, z, rm, ls, c, rh, geo, alpha1, alpha2]
            hyperparameters-Hyperparameters for fit 
                        [lr, lz, lrm, lgeo]'''
        predicted_mag = self.mag_prediction(data[:, 0], data[:, 1], *np.append(constants[:5], parameters[:4]))
        measured_mag = self.mag_measurment(data[:, 0], data[:, 2], data[:,3],*np.append(constants[5:9], parameters[4:7]))
        arguements = np.concatenate(([constants[0]], constants[4:], [parameters[0]], parameters[3:]))
        predicted_phase = self.phase_prediction(data[:, 0], data[:, 2], *arguements)
        measured_phase = data[:, 4]
        return ((predicted_mag - measured_mag)**2).mean() + 4*((predicted_phase - measured_phase)**2).mean() + hyperparameters[0] * math.fabs(parameters[1]) + hyperparameters[1] * math.fabs(parameters[2]) + hyperparameters[2] * math.fabs(parameters[3]) + hyperparameters[3] * math.fabs(parameters[6])

    def plot_mag_vs_freq(self, chosen_i=1e-3, label='Observed'):
        '''Plot measured and predicted magnet field vs frequency
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, alpha] '''
        data = self.data
        parameters = self.best
        constants = self.constants
        data = data[np.logical_and(data[:,1]==5,np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 0].argsort()]
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *np.append(constants[:5], parameters[:4]))
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3], *np.append(constants[5:9], parameters[4:7]))
        base_line = plt.plot(np.log10(data[data[:, 2]>0, 0]),
                             predicted[data[:,2]>0], linestyle='dashed')[0]
        plt.plot(np.log10(data[data[:, 2]<0, 0]), predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed')
        error = 3 * data[:, 5] / data[:, 3] * measured 
        plt.errorbar(np.log10(data[:, 0]), measured, yerr=error, fmt='o',
                    mfc='none', label=label, color=base_line.get_color())
        plt.xlabel('Log Freq (log Hz)')
        plt.ylabel('Magnitic Field (nT)')
        plt.legend()

    def plot_phase_vs_freq(self, chosen_i=1e-3, label='Observed'):
        '''Plot measured and predicted phase vs frequency
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo, dalpha]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, alpha] '''
        data = self.data
        data = data[np.logical_and(data[:,1]==5,np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 0].argsort()]
        parameters = self.best
        constants = self.constants
        arguements = np.concatenate(([constants[0]], constants[4:], [parameters[0]], parameters[3:]))
        predicted = self.phase_prediction(data[:, 0], data[:, 2], *arguements)
        measured = data[:, 4]
        base_line=plt.plot(np.log10(data[data[:, 2]>0, 0]), predicted[data[:,2]>0], linestyle='dashed')[0]
        plt.plot(np.log10(data[data[:, 2]<0, 0]), predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed')
        plt.errorbar(np.log10(data[:, 0]), measured, yerr=3*data[:, 6], fmt='o',
                     mfc='none', label=label, color=base_line.get_color())
        plt.legend()
        plt.xlabel('Log Freq (log Hz)')
        plt.ylabel('Phase')

    def plot_error_vs_freq(self, chosen_i=1e-3, label='Observed'):
        '''Plot field error vs frequency 
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo, dalpha]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, alpha] '''
        data = self.data
        data = data[np.logical_and(data[:,1]==5,np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 0].argsort()]
        parameters = self.best
        constants = self.constants
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3], *np.append(constants[5:9], parameters[4:7]))
        error = 3 * data[:, 5] / data[:,3] * measured
        plt.scatter(np.log10(data[data[:, 2]>0, 0]), error[data[:, 2]>0], label=label)
        plt.xlabel('Log Freq (log Hz)')
        plt.ylabel('Error (nT)')

    def plot_mag_vs_bias(self, chosen_i=1e-3,chosen_f=1e3, label='Observed'):
        '''Plot measured and predicted magnet field vs bias current
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, dalpha] '''
        data = self.data
        data = data[np.logical_and(data[:,0]==1000, data[:, 1]==5)]
        data = data[data[:, 2].argsort()]
        parameters = self.best
        constants = self.constants
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *np.append(constants[:5], parameters[:4]))
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3],*np.append(constants[5:9], parameters[4:7]))
        base_line=plt.plot(data[data[:, 2]>0, 2]*1000, predicted[data[:, 2]>0],
                           linestyle='dashed')[0]
        plt.plot(data[data[:, 2]<0, 2]*1000, predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed')
        error = 3 * data[:, 5] / data[:, 3] * measured 
        plt.errorbar(data[:, 2]*1000, measured, yerr=error, fmt='o',
                     mfc='none', label=label, color=base_line.get_color())
        plt.xlabel('Bias Current (mA)')
        plt.ylabel('Magnitic Field (nT)')
        plt.legend()

    def plot_mag_vs_mag(self, chosen_i=1e-3, chosen_f=1e3, label='Observed'):
        '''Plot measured vs predicted field 
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo, dalpha]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
        constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, alpha] '''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 1].argsort()]
        parameters = self.best
        constants = self.constants
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *np.append(constants[:5], parameters[:4]))
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3], *np.append(constants[5:9], parameters[4:7]))
        error = 3 * data[:, 5] / data[:, 3] * measured 
        base_line=plt.plot(predicted, predicted, linestyle='dashed')[0]
        plt.errorbar(predicted, measured, yerr=error, fmt='o', mfc='none',
                     color=base_line.get_color(), label=label)
        plt.xlabel('Applied Field (nT)')
        plt.ylabel('Observed Field (nT)')
        plt.legend()

    def plot_phase_vs_mag(self, chosen_i=1e-3, chosen_f=1e3, label='Observed'):
        '''Plot measured and predicted phase vs applied field
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo, dalpha]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, alpha] '''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 1].argsort()]
        parameters = self.best
        constants = self.constants
        arguements = np.concatenate(([constants[0]], constants[4:], [parameters[0]], parameters[3:]))
        predicted = self.phase_prediction(data[:, 0], data[:, 2], *arguements)
        measured = data[:, 4]
        predicted_mag = self.mag_prediction(data[:, 0], data[:, 1], *np.append(constants[:5], parameters[:4]))
        base_line=plt.plot(predicted_mag[data[:,2]>0], predicted[data[:, 2]>0], linestyle='dashed')[0]
        plt.plot(predicted_mag[data[:, 2]<0], predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed')
        plt.errorbar(predicted_mag, measured, yerr=3*data[:, 6], fmt='o',
                     mfc='none', label='Observed', color=base_line.get_color())
        plt.legend()
        plt.xlabel('Applied Field (nT)')
        plt.ylabel('Phase')

    def plot_error_vs_mag(self, chosen_i=1e-3, chosen_f=1e3, label='Observed'):
        '''Plot field error vs applied field
        input:
            parameters-Corrections to constants based of fit [dlm, dr, dz, drm, dls, dc, dgeo, dalpha]
            data-Data to plot ['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (A)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']
            constants-Constatnts needed for fit [lm, n, r, z, rm, ls, c, rh, geo, alpha] '''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 1].argsort()]
        parameters = self.best
        constants = self.constants
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *np.append(constants[:5], parameters[:4]))
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3], *np.append(constants[5:9], parameters[4:7]))
        error = 3 * data[:, 5] / data[:,3] * measured
        plt.scatter(predicted[data[:, 2]>0], error[data[:, 2]>0], label=label)
        plt.xlabel('Applied Field (nT)')
        plt.ylabel('Error (nT)')

    def print_parameters(self):
        constants = self.constants
        best = self.best
        print(self.name)
        print('The radius of the magnet is %.3f cm' % ((constants[2] + best[1])*100))
        print('The the distance from the magnet to the sensor is  %.3f cm' %
              ((constants[2] + best[2])*100))
        print('The hall constant of the sensor is  %.3f m^2/C' % constants[7])
        print('The geometric factor of the sensor is %.3f' % (constants[8] + best[6]))
        print('The resistance of the magnet is %.3f ohm' % (constants[4] + best[3]))
        print('The inductance of the sensor is %.3f mH' % ((constants[5] + best[4])*1e3))
        print('The inductance of the magnet is %.3f mH' % ((constants[0] + best[0])*1e3))
        print('The capacitance of the sensor is %.3f nF' % ((constants[6] + best[5]*1e9))) 
        print('The phase offset for positive bias is %.3f degrees' %
              (constants[9] + best[7]))
        print('The phase offset for negative bias is %.3f degrees' %
              (constants[10] + best[8]))

def do_qw():
    qw2 = Model('B052 2mm')
    qw5 = Model('B052 5mm')
    qw10 = Model('M145 10mm')
    qw2.print_parameters()
    qw5.print_parameters()
    qw10.print_parameters()
    qw2.plot_mag_vs_freq(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_mag_vs_freq(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_mag_vs_freq(chosen_i=1e-3, label='QW 10mm')
    plt.show()
    qw2.plot_phase_vs_freq(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_phase_vs_freq(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_phase_vs_freq(chosen_i=1e-3, label='QW 10mm')
    plt.show()
    qw2.plot_mag_vs_mag(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_mag_vs_mag(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_mag_vs_mag(chosen_i=1e-3, label='QW 10mm')
    plt.show()
    qw2.plot_phase_vs_mag(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_phase_vs_mag(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_phase_vs_mag(chosen_i=1e-3, label='QW 10mm')
    plt.show()
    qw2.plot_error_vs_freq(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_error_vs_freq(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_error_vs_freq(chosen_i=1e-3, label='QW 10mm')
    plt.legend()
    plt.show()
    qw2.plot_error_vs_mag(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_error_vs_mag(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_error_vs_mag(chosen_i=1e-3, label='QW 10mm')
    plt.legend()
    plt.show()

def do_epilayer():
    epilayer = Model('M148 10mm')
    epilayer.print_parameters()
    epilayer.plot_mag_vs_freq(chosen_i=1e-2, label='Epilayer 10mm')
    plt.show()
    epilayer.plot_phase_vs_freq(chosen_i=1e-2, label='Epilayer 10mm')
    plt.show()
    epilayer.plot_mag_vs_mag(chosen_i=1e-2, label='Epilayer 10mm')
    plt.show()
    epilayer.plot_phase_vs_mag(chosen_i=1e-2, label='Epilayer 10mm')
    plt.show()
    epilayer.plot_error_vs_freq(chosen_i=1e-2, label='Epilayer 10mm')
    plt.legend()
    plt.show()
    epilayer.plot_error_vs_mag(chosen_i=1e-2, label='Epilayer 10mm')
    plt.legend()
    plt.show()


def run():
    do_qw()
    do_epilayer()
    
if __name__ == '__main__':
    run()
