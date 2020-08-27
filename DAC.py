import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

# Read data from DAC csv file
data = csv.reader(open('DAC_transfer.csv', newline=''), delimiter=',')
digitalBIN, digitalDEC, Va, Va_bar = [], [], [], []
# convert column into array
for row in data:
    # print(row)
    digitalBIN.append(float(row[0]))
    digitalDEC.append(float(row[1]))
    Va.append(float(row[2]))
    Va_bar.append(float(row[3]))

print(digitalBIN)
print(Va)

plt.figure();
#### Va_bar ####
plt.suptitle('DAC Transfer Curve')
plt.xlabel('Digital (DEC)')
plt.ylabel('Va_bar Analog Votlage [mV]')
plt.scatter(digitalDEC,Va_bar)

x = np.linspace(0,256, num=100)
# Va bar equation
Va_bar_fit_param = polyfit(digitalDEC, Va_bar, 1)
Va_bar_equation = Va_bar_fit_param[1]*x + Va_bar_fit_param[0]
plt.suptitle('y = %.1fx + %.1f'%(Va_bar_fit_param[1], Va_bar_fit_param[0]))
plt.plot(x, Va_bar_equation, '-')

#### Va ####
threshold = 190
plt.scatter(digitalDEC, Va)
# Va equation
digitalDEC_forVaFit = digitalDEC[:12]
Va_forFit = Va[:12]
Va_fit_param = polyfit(digitalDEC_forVaFit, Va_forFit, 1)
Va_equation=[]
for element in x:
    if element < threshold:
        Va_equation.append(Va_fit_param[1] * element + Va_fit_param[0])
    else:
        Va_equation.append(1028)

plt.title('y = %.1fx + %.1f'%(Va_fit_param[1], Va_fit_param[0]))
plt.plot(x, Va_equation, '-')

plt.show()






