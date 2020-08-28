
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from DAC import dac_param, give_an_input_get_analog_output_dac
from ADC import adc_param, give_vmav_get_yout
# Read data from DAC csv file
#### DAC ####
data = csv.reader(open('DAC_transfer.csv', newline=''), delimiter=',')
digitalBIN, digitalDEC, Va, Va_bar = [], [], [], []
# convert column into array
for row in data:
    # print(row)
    digitalBIN.append(float(row[0]))
    digitalDEC.append(float(row[1]))
    Va.append(float(row[2]))
    Va_bar.append(float(row[3]))

#### ADC ####
data = csv.reader(open('ADC_transfer.csv', newline=''), delimiter=',')
Vmav, Yout = [], []
for row in data:
    # print(row)
    Vmav.append(float(row[0]))
    Yout.append(float(row[1]))

X1 = 1
W1 = 2

#### DAC ####
# Va / Va_bar generation
Va_fit_param, Va_bar_fit_param = dac_param(digitalDEC, Va, Va_bar)
print(Va_fit_param)
va, va_bar = give_an_input_get_analog_output_dac(250, Va_bar_fit_param, Va_bar_fit_param)
print(va)
print(va_bar)


#### ADC ####
yout_param = adc_param(Vmav, Yout)
yout = give_vmav_get_yout(600, yout_param)
print(yout)





