
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from DAC import dac_param, give_an_input_get_analog_output_dac
from ADC import adc_param, give_vmav_get_yout
from MAV import mav_transfer, give_weight_get_vmav
from keras.utils import np_utils

classes = 10

#### DATA ####
######################
from data.cifar_test_xarray import X_test
# print(X_test.shape, 'test samples')


# Weight
weight_data = []
with open('converted_weight_CONV1_dec.txt', 'r') as weight_file:
    reader = csv.reader(weight_file)
    for row in reader:
        row.pop()
        weight_data.append( [int(i) for i in row] )
weight_file.close()

# for i in range(0, len(weight_data)):
#     weight_data[i] = int(weight_data[0][i])
# print(weight_data[0][0])

# x_test = np.random.randint(0,256,size=(3,32,32))
# weight_data1 = np.random.randint(-126,126,size=(2,27))
# print(weight_data1)


def cim_conv(Xin, Win, Va_fit_param, Va_bar_fit_param, Va_vs_Vmav_param, Va_bar_vs_Vmav_param, yout_param):
    #### DAC ####
    # Va / Va_bar generation
    # print('Xin = ' + str(Xin))
    va, va_bar = give_an_input_get_analog_output_dac(Xin, Va_fit_param, Va_bar_fit_param)
    # print('Va = ' + str(va))
    # print('Va_bar = ' + str(va_bar))
    va = va/1000
    va_bar = va_bar / 1000
    #### MAV ####
    vmav = give_weight_get_vmav(Win, 1.2-va, va_bar, Va_vs_Vmav_param, Va_bar_vs_Vmav_param)
    # print('Vmav = ' + str(vmav))
    #### ADC ####
    yout = int(give_vmav_get_yout(vmav*1000, yout_param))
    # print('Digital Output = ' + str(yout))
    return yout


def conv1(X_test, weight_data):
    ### Parameters ###
    channel = X_test.shape[1]
    filter_size = 3
    dividor = 27
    # next_layer_xy = X_test.shape[0][1]-weight_data.shape[0]+1
    next_layer_xy = 30
    number_of_img = X_test.shape[0]
    number_of_filter = 32
    # print('channel = ' + str(channel) + 'filter_size = ' + str(filter_size))
    Va_fit_param, Va_bar_fit_param = dac_param()
    Va_vs_Vmav_param, Va_bar_vs_Vmav_param = mav_transfer()
    yout_param = adc_param()

    partial_sum_counter = -1
    next_layer_input = np.zeros(shape=(number_of_img,number_of_filter,30,30))

    # yout = cim_conv(180, 70)
    for this_img in range(number_of_img):
        for this_filter in range(number_of_filter):
            for this_many_y in range(next_layer_xy): # loop thru image y axis
                # next_layer_input.append([])
                for this_many_x in range(next_layer_xy): # loop thru image x axis
                    partial_sum_single = 0
                    for this_channel in range(channel): # internal loop within filter 3*3*3
                        for this_row in range(filter_size):
                            for this_col in range(filter_size):
                                # first index = which image
                                partial_sum_single = partial_sum_single + cim_conv(
                                    X_test[this_img][this_channel][this_row + this_many_y][this_col + this_many_x],
                                    weight_data[this_filter][this_col + 3*this_row + 9*this_channel],
                                    Va_fit_param, Va_bar_fit_param, Va_vs_Vmav_param, Va_bar_vs_Vmav_param, yout_param)

                                # print('this_col = ' + str(this_col) + '  this_row = ' + str(this_row) + '  this_channel = ' + str(this_channel))
                                # print('index === ' + str(this_col + 3*this_row + 9*this_channel))
                                # print('weight data ====== ' + str(weight_data[this_filter][this_col + 3*this_row + 9*this_channel]))

                    # convert 27 into one average result
                    partial_sum_avg = partial_sum_single/dividor
                    # print('partial sum average ===== ' + str(partial_sum_avg))
                    next_layer_input[this_img][this_filter][this_many_y][this_many_x] = partial_sum_avg
                    # print('index = ' + str(this_many_y + next_layer_xy*this_filter))

    # with open('partial_sum.txt', 'w') as f:
    #     for item in partial_sum:
    #         f.write("%s\n" % item)

    return next_layer_input


# testing
# next_layer_input = conv1(X_test, weight_data)
# print(next_layer_input)
# print('shape = ' + str(next_layer_input.shape))
# print('shape = ' + str(len(next_layer_input)) + ' , ' + str(len(next_layer_input[0])))







