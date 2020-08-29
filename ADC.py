import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

def adc_param():
    data = csv.reader(open('ADC_transfer.csv', newline=''), delimiter=',')
    Vmav, Yout = [], []
    for row in data:
        # print(row)
        Vmav.append(float(row[0]))
        Yout.append(float(row[1]))

    # print(Vmav,Yout)

    x = np.linspace(0,1200, num=1000)

    # get equation
    Yout_fit_param = polyfit(Vmav, Yout, 1)
    Yout_equation = Yout_fit_param[1]*x + Yout_fit_param[0]

    # plt.figure()
    # plt.xlabel('Vmav [V]')
    # plt.ylabel('Yout - Digital Output [DEC]')
    #
    # #### Va Bar ####
    # plt.scatter(Vmav,Yout)
    # plt.plot(x,Yout_equation)
    # plt.show()
    return Yout_fit_param

def give_vmav_get_yout(vmav, Yout_fit_param):
    yout = Yout_fit_param[1] * vmav + Yout_fit_param[0]
    return yout


# testing
# yout_param = adc_param(Vmav, Yout)
# yout = give_vmav_get_yout(600, yout_param)
# print(yout)

