
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from DAC import dac_param, give_an_input_get_analog_output_dac
from ADC import adc_param, give_vmav_get_yout
from MAV import mav_transfer, give_weight_get_vmav

x_test = np.random.randint(0,256,size=(3,32,32))
weight_data = np.random.randint(-126,126,size=(3,3,3))
# print(x_test)
# print(x_test[0,0,1])
# print(x_test[0,1,0])
# print(x_test[0,1,1])
with open('input1.txt', 'w') as f:
    for x in range(x_test.shape[0]):
        f.write("\n")
        for y in range(x_test.shape[1]):
            f.write("\n")
            for z in range(x_test.shape[2]):
                f.write("%s," % x_test[x][y][z])
with open('weight.txt', 'w') as f:
    for x in range(weight_data.shape[0]):
        f.write("\n")
        for y in range(weight_data.shape[1]):
            f.write("\n")
            for z in range(weight_data.shape[2]):
                f.write("%s," % weight_data[x][y][z])
next_layer_xy = x_test.shape[1]-weight_data.shape[0]+1
# print('shape = ' + str(next_layer_xy) )

### Parameters ###
channel = x_test.shape[0]
filter_size = weight_data.shape[0]
# print('channel = ' + str(channel) + 'filter_size = ' + str(filter_size))


def cim_conv(Xin, Win):
    #### DAC ####
    # Va / Va_bar generation
    # print('Xin = ' + str(Xin))
    Va_fit_param, Va_bar_fit_param = dac_param()
    va, va_bar = give_an_input_get_analog_output_dac(Xin, Va_fit_param, Va_bar_fit_param)
    # print('Va = ' + str(va))
    # print('Va_bar = ' + str(va_bar))
    va = va/1000
    va_bar = va_bar / 1000
    #### MAV ####
    Va_vs_Vmav_param, Va_bar_vs_Vmav_param = mav_transfer()
    vmav = give_weight_get_vmav(Win, 1.2-va, va_bar, Va_vs_Vmav_param, Va_bar_vs_Vmav_param)
    # print('Vmav = ' + str(vmav))
    #### ADC ####
    yout_param = adc_param()
    yout = int(give_vmav_get_yout(vmav*1000, yout_param))
    # print('Digital Output = ' + str(yout))
    return yout


# yout = cim_conv(180, 70)

partial_sum_counter = -1
partial_sum = []
next_layer_input = []

dividor = 27
for this_many_y in range(next_layer_xy):
    next_layer_input.append([])
    for this_many_x in range(next_layer_xy):
        partial_sum_single = 0
        partial_sum.append([])
        for this_channel in range (channel):
            for this_row in range (filter_size):
                for this_col in range(filter_size):

                    partial_sum[partial_sum_counter].append(cim_conv(x_test[this_channel][this_row+this_many_y][this_col+this_many_x],
                                                 weight_data[this_channel][this_row][this_col]))

                    partial_sum_single = partial_sum_single + cim_conv(x_test[this_channel][this_row + this_many_y][this_col + this_many_x],
                                                                weight_data[this_channel][this_row][this_col])

        # convert 27 into one average result
        partial_sum_avg = partial_sum_single/dividor
        print('partial sum average ===== ' + str(partial_sum_avg))
        next_layer_input[this_many_y].append(partial_sum_avg)

# with open('partial_sum.txt', 'w') as f:
#     for item in partial_sum:
#         f.write("%s\n" % item)

print(next_layer_input)
print('len0 = ' + str(len(next_layer_input)))
print('len1 = ' + str(len(next_layer_input[0])))










