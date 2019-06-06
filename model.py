import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize

class Model(object):
    def __init__(self, name):
        '''Initialize model object
        input:
            name - Name of model and file to find data'''
        # Record Name
        self.name = name
        # Fetch Constants
        constants_file = os.path.join(os.getcwd(),'data',name,'constants.csv')
        constants_frame = pd.read_csv(constants_file)
        self.rh = constants_frame['R_H (m^2/C)'][0]
        self.n = constants_frame['N'][0]
        h = constants_frame['h_s (mm)'][0]*1e-3
        k = constants_frame['w_s (mm)'][0]*1e-3
        res = constants_frame['R_s (ohm)'][0]
        theta = math.atan(self.rh*500e-9/res)
        self.r0 = constants_frame['R (cm)'][0]*1e-2
        self.z0 = constants_frame['z (cm)'][0]*1e-2
        self.rm0 = constants_frame['R_m (ohms)'] [0]
        self.ls0 = constants_frame['L_s (H)'][0]
        self.lm0 = constants_frame['L_m (H)'][0]
        self.c0 = constants_frame['C (nF)'][0]*1e-9
        self.geo0 = 1-1.045*math.exp(-math.pi*h/k)*theta/math.tan(theta)
        self.alpha10 = constants_frame['Alpha 1'][0]
        self.alpha20 = constants_frame['Alpha 2'][0]
        # Fetch Data
        data_file = os.path.join(os.getcwd(),'data',name,'data_summary.csv')
        data_frame = pd.read_csv(data_file)
        data_frame['I_bias (mA)'] = data_frame['I_bias (mA)']*1e-3
        self.data = data_frame[['freq (Hz)', 'Lock-In Voltage (V)', 'I_bias (mA)', 'R Ave (mV)', 'Phase Avg', 'R Std Dev (mV)', 'Phase Std Dev']].values 
        # Initial Parameters
        parameters = np.array([self.r0, self.z0, self.rm0, self.ls0, self.lm0,
                               self.c0, self.geo0, self.alpha10,
                               self.alpha20], copy=True)
        # Define indices for parameters
        self.indices = {'r':0, 'z':1, 'rm':2, 'ls':3, 'lm':4, 'c':5, 'geo':6,
                        'alpha1':7, 'alpha2':8}
        # Fetch Hyperparameters
        hyper_file = os.path.join(os.getcwd(),'data',name,'hyperparameters.csv')
        hyper_frame = pd.read_csv(hyper_file)
        self.lr = hyper_frame['lr'][0]
        self.lz = hyper_frame['lz'][0]
        self.lrm = hyper_frame['lrm'][0]
        self.lls = hyper_frame['lls'][0]
        self.lc = hyper_frame['lc'][0]
        self.llm = hyper_frame['llm'][0]
        self.lgeo = hyper_frame['lgeo'][0]
        self.lalpha1 = hyper_frame['lalpha1'][0]
        self.lalpha2 = hyper_frame['lalpha2'][0]
        self.lmag = hyper_frame['lmag'][0]
        self.lphase = hyper_frame['lphase'][0]
        # Fit
        num_trials = 10
        self.best = np.zeros(9, dtype=float)
        for i in range(num_trials):
            min = minimize(self.loss, parameters,  method='Powell')
            self.best += min.x
        self.best /= num_trials

    def loss(self, parameters):
        '''Loss function for fit
        input: 
            parameters - array of parameters
        output: Loss value for these parameters'''
        loss = 0
        loss += self.mag_loss(parameters)
        loss += self.phase_loss(parameters)
        loss += self.param_loss(*parameters)
        return loss
       
    def mag_loss(self, parameters):
        '''Loss due to magnetic field measurments is the mean squared difference
        between predicted and measured fields times a hyperparameter 
        input: 
            parameters - Array of Parameters
        output: Part of loss due to the magnetic field measurements'''
        predicted_mag = self.mag_prediction(self.data[:,0], self.data[:, 1], *parameters)
        measured_mag = self.mag_measurment(self.data[:,0], self.data[:,2], self.data[:,3], *parameters)
        return self.lmag*((predicted_mag-measured_mag)**2).mean()

    def phase_loss(self, parameters):
        '''Loss due to phase measurments is the mean squared difference
        between predicted and measured phases times a hyperparameter 
        input: 
            parameters - Array of Parameters
        output: Part of loss due to the phase measurements'''
        predicted_phase = self.phase_prediction(self.data[:, 0], self.data[:, 2], *parameters)
        measured_phase = self.data[:, 4]
        return self.lphase*((predicted_phase - measured_phase)**2).mean()

    def param_loss(self, r, z, rm, ls, c, lm, geo, alpha1, alpha2):
        '''Loss due to changes in parameters. This forces the model to adjust
        to our previous knowledge of the system. The loss for each parameter is
        the square difference between the parameter and our original guess
        times a hyperparameter.
        input: 
            r - Radius of the magnet
            z - Distance from magnet to sensor
            rm - Resistence of Magnet
            ls - Inductance of Sensor
            c - Capacitence of the Sensor
            lm - Inductance of the Magnet
            geo - Geometric Factor
            alpha1 - Phase offset for positive bias
            alpha2 - Phase offset for negative bias
        output: Part of loss due to changes in parameters'''
        loss = 0
        loss += self.lr * (r - self.r0)**2
        loss += self.lz * (z - self.z0)**2
        loss += self.lrm * (rm - self.rm0)**2
        loss += self.lls * (ls - self.ls0)**2
        loss += self.lc * (c - self.c0)**2
        loss += self.llm * (lm - self.lm0)**2
        loss += self.lgeo * (geo - self.geo0)**2
        loss += self.lalpha1 * (alpha1 - self.alpha10)**2
        loss += self.lalpha2 * (alpha2 - self.alpha20)**2
        return loss

    def mag_prediction(self, freq, vm, r, z, rm, ls, c, lm, geo, alpha1, alpha2):
        ''' Calculate Applied Magnetic Field using parameters, frequency, and
        voltage
        input:
            freq - Frequency of Field
            vm - Lock-In Voltage
            r - Radius of the magnet
            z - Distance from magnet to sensor
            rm - Resistence of Magnet
            ls - Inductance of Sensor
            c - Capacitence of the Sensor
            lm - Inductance of the Magnet
            geo - Geometric Factor
            alpha1 - Phase offset for positive bias
            alpha2 - Phase offset for negative bias
        output: Predicted Magnetic Field in nT'''
        mu = 1.25e-6 
        omega = 2 * math.pi * freq
        prediction = mu * self.n * r**2 * vm * 1e9
        prediction /= 2**1.5
        prediction /= (z**2 + r**2)**1.5
        prediction /= (rm**2 + (lm * omega)**2)**0.5
        return prediction
    
    def mag_measurment(self, freq, ib, vh, r, z, rm, ls, c, lm, geo, alpha1, alpha2):
        ''' Calculate Measured Magnetic Field given parameters, frequency, 
        bias, and measurement
        input:
            freq - Frequency of Field
            ib - Bias Current
            vh - Hall Voltage
            freq - Frequency of Field
            vm - Lock-In Voltage
            r - Radius of the magnet
            z - Distance from magnet to sensor
            rm - Resistence of Magnet
            ls - Inductance of Sensor
            c - Capacitence of the Sensor
            lm - Inductance of the Magnet
            geo - Geometric Factor
            alpha1 - Phase offset for positive bias
            alpha2 - Phase offset for negative bias
        output: Measured Magnetic Field in nT'''
        omega = 2 * math.pi * freq
        measurement = vh * (1 + (6000 * omega * c)**2)**0.5 * 1e6
        measurement /= 2000 * ((geo * self.rh * ib)**2 + (ls * omega)**2)**0.5
        return measurement

    def phase_prediction(self, freq, ib, r, z, rm, ls, c, lm, geo, alpha1, alpha2):
        ''' Calculate predicted phase given parameters
        input:
            freq - Frequency of Field
            ib - Bias Current
            freq - Frequency of Field
            vm - Lock-In Voltage
            r - Radius of the magnet
            z - Distance from magnet to sensor
            rm - Resistence of Magnet
            ls - Inductance of Sensor
            c - Capacitence of the Sensor
            lm - Inductance of the Magnet
            geo - Geometric Factor
            alpha1 - Phase offset for positive bias
            alpha2 - Phase offset for negative bias
        output: Predicted Phase'''
        omega = 2 * math.pi * freq
        prediction = np.arctan(ls * omega / (geo * self.rh * ib))
        prediction += np.arctan(lm * omega / rm)
        prediction += np.arctan(6000 * omega * c)
        prediction *= -180 / math.pi
        prediction[ib>0] += alpha1
        prediction[ib<0] += alpha2
        return prediction

    def plot_mag_vs_freq(self, chosen_i=1e-3, chosen_vm=5, label1='Observed', label2='Predicted'):
        '''Plot measured and predicted magnet field vs frequency
        input:
            chosen_i - Chosen bias current to plot
            chosen_vm - Chosen magnet voltage to plot 
            label1 - Label for measurements
            label2 - Label for predictions'''
        data = self.data
        data = data[np.logical_and(data[:,1]==chosen_vm,np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 0].argsort()]
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *self.best)
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3], *self.best)
        base_line = plt.plot(np.log10(data[data[:, 2]>0, 0]),
                             predicted[data[:,2]>0], linestyle='dashed')[0]
        plt.plot(np.log10(data[data[:, 2]<0, 0]), predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed', label=label2)
        error = 3 * data[:, 5] / data[:, 3] * measured 
        plt.errorbar(np.log10(data[:, 0]), measured, yerr=error, fmt='o',
                    mfc='none', label=label1, color=base_line.get_color())
        plt.xlabel('Log Freq (log Hz)')
        plt.ylabel('Magnitic Field (nT)')
        plt.legend()

    def plot_phase_vs_freq(self, chosen_i=1e-3, chosen_vm=5, label1='Observed', label2='Predicted'):
        '''Plot measured and predicted phase vs frequency
        input:
            chosen_i - Chosen bias current to plot
            chosen_vm - Chosen magnet voltage to plot 
            label1 - Label for measurements
            label2 - Label for predictions'''
        data = self.data
        data = data[np.logical_and(data[:,1]==chosen_vm,np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 0].argsort()]
        predicted = self.phase_prediction(data[:, 0], data[:, 2], *self.best)
        measured = data[:, 4]
        base_line=plt.plot(np.log10(data[data[:, 2]>0, 0]), predicted[data[:,2]>0], linestyle='dashed')[0]
        plt.plot(np.log10(data[data[:, 2]<0, 0]), predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed', label=label2)
        plt.errorbar(np.log10(data[:, 0]), measured, yerr=3*data[:, 6], fmt='o',
                     mfc='none', label=label1, color=base_line.get_color())
        plt.legend()
        plt.xlabel('Log Freq (log Hz)')
        plt.ylabel('Phase')

    def plot_error_vs_freq(self, chosen_i=1e-3, chosen_vm=5, label=None):
        '''Plot field error vs frequency 
        input:
            chosen_i - Chosen bias current to plot
            chosen_vm - Chosen magnet voltage to plot 
            label - Label for errors'''
        data = self.data
        data = data[np.logical_and(data[:,1]==chosen_vm,np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 0].argsort()]
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3],
                                       *self.best)
        error = 3 * data[:, 5] / data[:,3] * measured
        plt.scatter(np.log10(data[data[:, 2]>0, 0]), error[data[:, 2]>0], label=label)
        plt.xlabel('Log Freq (log Hz)')
        plt.ylabel('Error (nT)')

    def plot_mag_vs_bias(self, chosen_f=1e3, chosen_vm=5, label1='Observed', label2='Predicted'):
        '''Plot measured and predicted magnet field vs bias current
        input:
            chosen_f - Chosen frequency to plot
            chosen_vm - Chosen magnet voltage to plot 
            label1 - Label for measurements
            label2 - Label for predictions'''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, data[:, 1]==chosen_vm)]
        data = data[data[:, 2].argsort()]
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *self.best)
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:,3], *self.best)
        base_line=plt.plot(data[data[:, 2]>0, 2]*1000, predicted[data[:, 2]>0],
                           linestyle='dashed')[0]
        plt.plot(data[data[:, 2]<0, 2]*1000, predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed', label=label2)
        error = 3 * data[:, 5] / data[:, 3] * measured 
        plt.errorbar(data[:, 2]*1000, measured, yerr=error, fmt='o',
                     mfc='none', label=label1, color=base_line.get_color())
        plt.xlabel('Bias Current (mA)')
        plt.ylabel('Magnitic Field (nT)')
        plt.legend()

    def plot_mag_vs_mag(self, chosen_i=1e-3, chosen_f=1e3, label1='Observed', label2='Predicted'):
        '''Plot measured vs predicted field 
        input:
            chosen_i - Chosen bias current to plot
            chosen_f - Chosen frequency to plot
            label1 - Label for measurements
            label2 - Label for predictions'''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 1].argsort()]
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *self.best)
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3],
                                       *self.best)
        error = 3 * data[:, 5] / data[:, 3] * measured 
        base_line=plt.plot(predicted, predicted, linestyle='dashed',
                           label=label2)[0]
        plt.errorbar(predicted, measured, yerr=error, fmt='o', mfc='none',
                     color=base_line.get_color(), label=label1)
        plt.xlabel('Applied Field (nT)')
        plt.ylabel('Observed Field (nT)')
        plt.legend()

    def plot_phase_vs_mag(self, chosen_i=1e-3, chosen_f=1e3, label1='Observed', label2='Predicted'):
        '''Plot measured and predicted phase vs applied field
        input: 
            chosen_i - Chosen bias current to plot
            chosen_f - Chosen frequency to plot
            label1 - Label for measurements
            label2 - Label for predictions'''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 1].argsort()]
        predicted = self.phase_prediction(data[:, 0], data[:, 2], *self.best)
        measured = data[:, 4]
        predicted_mag = self.mag_prediction(data[:, 0], data[:, 1], *self.best)
        base_line=plt.plot(predicted_mag[data[:,2]>0], predicted[data[:, 2]>0], linestyle='dashed')[0]
        plt.plot(predicted_mag[data[:, 2]<0], predicted[data[:, 2]<0],
                 color=base_line.get_color(), linestyle='dashed', label=label2)
        plt.errorbar(predicted_mag, measured, yerr=3*data[:, 6], fmt='o',
                     mfc='none', label=label1, color=base_line.get_color())
        plt.xlabel('Applied Field (nT)')
        plt.ylabel('Phase')
        plt.legend()

    def plot_error_vs_mag(self, chosen_i=1e-3, chosen_f=1e3, label=None):
        '''Plot field error vs applied field
        input: 
            chosen_i - Chosen bias current to plot
            chosen_f - Chosen frequency to plot
            label - Label for error'''
        data = self.data
        data = data[np.logical_and(data[:,0]==chosen_f, np.fabs(data[:,2])==chosen_i)]
        data = data[data[:, 1].argsort()]
        predicted = self.mag_prediction(data[:, 0], data[:, 1], *self.best)
        measured = self.mag_measurment(data[:, 0], data[:, 2], data[:, 3], *self.best)
        error = 3 * data[:, 5] / data[:,3] * measured
        plt.scatter(predicted[data[:, 2]>0], error[data[:, 2]>0], label=label)
        plt.xlabel('Applied Field (nT)')
        plt.ylabel('Error (nT)')

    def print_parameters(self, r=None, z=None, rm=None, ls=None, c=None,
                         lm=None, geo=None, alpha1=None, alpha2=None):
        '''Print Parameters in a nice way
        input: 
            r - Radius of the magnet
            z - Distance from magnet to sensor
            rm - Resistence of Magnet
            ls - Inductance of Sensor
            c - Capacitence of the Sensor
            lm - Inductance of the Magnet
            geo - Geometric Factor
            alpha1 - Phase offset for positive bias
            alpha2 - Phase offset for negative bias'''
        if r is None:
            self.print_parameters(*self.best)
        else:
            print(self.name)
            print('The radius of the magnet is %.3f cm' % (r*100))
            print('The the distance from the magnet to the sensor is  %.3f cm' % (z*100))
            print('The hall constant of the sensor is  %.3f m^2/C' % self.rh)
            print('The geometric factor of the sensor is %.3f' % geo)
            print('The resistance of the magnet is %.3f ohm' % rm)
            print('The inductance of the sensor is %.3f mH' % (ls*1e3))
            print('The inductance of the magnet is %.3f mH' % (lm*1e3))
            print('The capacitance of the sensor is %.3f nF' % (c*1e9))
            print('The phase offset for positive bias is %.3f degrees' % alpha1)
            print('The phase offset for negative bias is %.3f degrees' % alpha2)

def analyze_device(name, chosen_i=1e-4):
    '''Perform analysis on a device
    input:
        name - Device to perform analysis on  
        chosen_i - Chosen bias current to plot'''
    model = Model(name)
    model.print_parameters()
    model.plot_mag_vs_freq(chosen_i=chosen_i)
    plt.savefig(os.path.join(os.getcwd(), 'figures', name, 'mag_vs_freq.png'))
    plt.close()
    model.plot_phase_vs_freq(chosen_i=chosen_i)
    plt.savefig(os.path.join(os.getcwd(), 'figures', name, 'phase_vs_freq.png'))
    plt.close()
    model.plot_mag_vs_mag(chosen_i=chosen_i)
    plt.savefig(os.path.join(os.getcwd(), 'figures', name, 'mag_vs_mag_1kHz.png'))
    plt.close()
    model.plot_phase_vs_mag(chosen_i=chosen_i)
    plt.savefig(os.path.join(os.getcwd(), 'figures', name, 'phase_vs_mag_1kHz.png'))
    plt.close()
    model.plot_error_vs_freq(chosen_i=chosen_i)
    plt.savefig(os.path.join(os.getcwd(), 'figures', name, 'error_vs_freq.png'))
    plt.close()
    model.plot_error_vs_mag(chosen_i=chosen_i)
    plt.savefig(os.path.join(os.getcwd(), 'figures', name, 'error_vs_mag_1kHz.png'))
    plt.close()

def analyze_all():
    '''Perform analysis on all devices at once'''
    qw2 = Model('QW 2mm')
    qw5 = Model('QW 5mm')
    qw10 = Model('QW 10mm')
    epilayer = Model('Epilayer 10mm')
    qw2.print_parameters()
    qw5.print_parameters()
    qw10.print_parameters()
    epilayer.print_parameters()
    qw2.plot_mag_vs_freq(chosen_i=1e-4, label1='QW 2mm', label2=None)
    qw5.plot_mag_vs_freq(chosen_i=1e-4, label1='QW 5mm', label2=None)
    qw10.plot_mag_vs_freq(chosen_i=1e-3, label1='QW 10mm', label2=None)
    epilayer.plot_mag_vs_freq(chosen_i=1e-2, label1='Epilayer 10mm', label2=None)
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'Composite', 'mag_vs_freq.png'))
    plt.close()
    qw2.plot_phase_vs_freq(chosen_i=1e-4, label1='QW 2mm', label2=None)
    qw5.plot_phase_vs_freq(chosen_i=1e-4, label1='QW 5mm', label2=None)
    qw10.plot_phase_vs_freq(chosen_i=1e-3, label1='QW 10mm', label2=None)
    epilayer.plot_phase_vs_freq(chosen_i=1e-2, label1='Epilayer 10mm', label2=None)
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'Composite', 'phase_vs_freq.png'))
    plt.close()
    qw2.plot_mag_vs_mag(chosen_i=1e-4, label1='QW 2mm', label2=None)
    qw5.plot_mag_vs_mag(chosen_i=1e-4, label1='QW 5mm', label2=None)
    qw10.plot_mag_vs_mag(chosen_i=1e-3, label1='QW 10mm', label2=None)
    epilayer.plot_mag_vs_mag(chosen_i=1e-2, label1='Epilayer 10mm', label2=None)
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'Composite', 'mag_vs_mag_1kHz.png'))
    plt.close()
    qw2.plot_phase_vs_mag(chosen_i=1e-4, label1='QW 2mm', label2=None)
    qw5.plot_phase_vs_mag(chosen_i=1e-4, label1='QW 5mm', label2=None)
    qw10.plot_phase_vs_mag(chosen_i=1e-3, label1='QW 10mm', label2=None)
    epilayer.plot_phase_vs_mag(chosen_i=1e-2, label1='Epilayer 10mm', label2=None)
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'Composite', 'phase_vs_mag_1kHz.png'))
    plt.close()
    qw2.plot_error_vs_freq(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_error_vs_freq(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_error_vs_freq(chosen_i=1e-3, label='QW 10mm')
    epilayer.plot_error_vs_freq(chosen_i=1e-2, label='Epilayer 10mm')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'Composite', 'error_vs_freq.png'))
    plt.close()
    qw2.plot_error_vs_mag(chosen_i=1e-4, label='QW 2mm')
    qw5.plot_error_vs_mag(chosen_i=1e-4, label='QW 5mm')
    qw10.plot_error_vs_mag(chosen_i=1e-3, label='QW 10mm')
    epilayer.plot_error_vs_mag(chosen_i=1e-2, label='Epilayer 10mm')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'Composite', 'error_vs_mag_1kHz.png'))
    plt.close()

def run():
    analyze_device('QW 2mm')
    analyze_device('QW 5mm')
    analyze_device('QW 10mm', chosen_i=1e-3)
    analyze_device('Epilayer 10mm', chosen_i=1e-2)
    analyze_all()

if __name__ == '__main__':
    run()
