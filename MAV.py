import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

def mav_transfer():
    number_of_va = 7
    number_of_weight = 128
    # Read data from DAC csv file
    data = csv.reader(open('MAV_transfer.csv', newline=''), delimiter=',')
    weight_dec, Vmav_pos, Vmav_neg = [], [], []

    x = np.linspace(0,128, num=100)

    # convert column into array
    for row in data:
        # print(row)
        weight_dec.append((float(row[0])))
        for index in range(number_of_va):
            Vmav_pos.append([])
            Vmav_pos[index].append(float(row[index+1]))
        for index in range(number_of_va):
            Vmav_neg.append([])
            Vmav_neg[index].append(float(row[index + 8]))

    va_bar = np.linspace(0.0,0.6, num=number_of_va)
    va = np.linspace(0,0.6,num=number_of_va)
    ################### Neg ######################
    va_bar_fit_params, va_bar_equations = [], []
    for i in range(number_of_va):
        va_bar_fit_params.append([])
        va_bar_equations.append([])
        va_bar_fit_params[i].append(polyfit(weight_dec, Vmav_neg[i], 1))
        va_bar_equations[i].append(va_bar_fit_params[i][0][1] * x + va_bar_fit_params[i][0][0])

    # plt.figure()
    # plt.xlabel('Weight DEC')
    # plt.ylabel('Vmav [V]')
    # for i in range(6):
    #     plt.plot(weight_dec, va_bar[i], '--')
    #     plt.plot(x, va_bar_equations[i][0])
    # plt.show()

    Vmav_va_bar = []
    weights = np.linspace(0,number_of_weight-1, num=number_of_weight)
    for i in range(number_of_va):
        Vmav_va_bar.append(va_bar_fit_params[i][0][1] * weights + va_bar_fit_params[i][0][0])

    Vmav_forSpecificWeight_neg = []
    for thisWeight in range(number_of_weight):
        Vmav_forSpecificWeight_neg.append([])
        for thisVa in range(number_of_va):
            Vmav_forSpecificWeight_neg[thisWeight].append(Vmav_va_bar[thisVa][thisWeight])

    # final result
    Va_bar_vs_Vmav, Va_bar_vs_Vmav_param=[],[]
    for i in range(number_of_weight):
        Va_bar_vs_Vmav.append([])
        Va_bar_vs_Vmav_param.append(([]))
        Va_bar_vs_Vmav_param[i].append(polyfit(va_bar, Vmav_forSpecificWeight_neg[i], 1))
        Va_bar_vs_Vmav[i].append(Va_bar_vs_Vmav_param[i][0][1]*va_bar + Va_bar_vs_Vmav_param[i][0][0])

    # plt.figure()
    # plt.xlabel('Va [V]')
    # plt.ylabel('Vmav [V]')
    # for i in range(number_of_weight):
    #     plt.plot(va_bar, Va_bar_vs_Vmav[i][0])
    # plt.show()

    # ################### Pos ######################
    va_fit_params, va_equations = [], []
    for i in range(number_of_va):
        va_fit_params.append([])
        va_equations.append([])
        va_fit_params[i].append(polyfit(weight_dec, Vmav_pos[i], 1))
        va_equations[i].append(va_fit_params[i][0][1]*x + va_fit_params[i][0][0])

    # plt.figure()
    # plt.xlabel('Weight DEC')
    # plt.ylabel('Vmav [V]')
    # for i in range(6):
    #     plt.plot(weight_dec, va[i], '--')
    #     plt.plot(x, va_equations[i][0])
    # plt.show()

    # given a specific Va, plot 000~111 (totally 128 lines) of weight vs Vmav
    weights = np.linspace(0,number_of_weight-1, num=number_of_weight)
    Vmav_va = []
    for i in range(number_of_va):
        Vmav_va.append(va_fit_params[i][0][1] * weights + va_fit_params[i][0][0])

    Vmav_forSpecificWeight_pos = []
    for thisWeight in range(number_of_weight):
        Vmav_forSpecificWeight_pos.append([])
        for thisVa in range(number_of_va):
            Vmav_forSpecificWeight_pos[thisWeight].append(Vmav_va[thisVa][thisWeight])
    # final result
    Va_vs_Vmav, Va_vs_Vmav_param=[],[]
    for i in range(number_of_weight):
        Va_vs_Vmav.append([])
        Va_vs_Vmav_param.append(([]))
        Va_vs_Vmav_param[i].append(polyfit(va, Vmav_forSpecificWeight_pos[i], 1))
        Va_vs_Vmav[i].append(Va_vs_Vmav_param[i][0][1]*va + Va_vs_Vmav_param[i][0][0])

    # plt.figure()
    # plt.xlabel('Va [V]')
    # plt.ylabel('Vmav [V]')
    # for i in range(number_of_weight):
    #     plt.plot(va, Va_vs_Vmav[i][0])
    # plt.show()

    return Va_vs_Vmav_param, Va_bar_vs_Vmav_param


def give_weight_get_vmav(weight_in, va, va_bar, Va_vs_Vmav_param, Va_bar_vs_Vmav_param):
    if weight_in > 0: # positive - use va
        vmav = Va_vs_Vmav_param[weight_in][0][1] * va + Va_vs_Vmav_param[weight_in][0][0]
    else: # negative - use va-bar
        vmav = Va_bar_vs_Vmav_param[abs(weight_in)][0][1] * va_bar + Va_bar_vs_Vmav_param[abs(weight_in)][0][0]
    return vmav

# Testing
# Va_vs_Vmav_param, Va_bar_vs_Vmav_param = mav_transfer()
# print(Va_vs_Vmav_param, Va_bar_vs_Vmav_param)
# vmav = give_weight_get_vmav(63, 0.9-0.6,0.3, Va_vs_Vmav_param, Va_bar_vs_Vmav_param)
# print(vmav)



