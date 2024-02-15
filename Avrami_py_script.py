import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#load the data
file_path = r'C:\Users\name\Documents\SN_TEST.csv'

df = pd.read_csv(file_path)

print(df.to_string()) 

# clean the csv file from spaces 
df.columns = [col.strip() for col in df.columns]
print(df)


# scatter plot of the raw points
df.plot(x='time', y='Area', marker='o', linestyle='-')
plt.xlabel('Time (min)')
plt.ylabel('DH (J/g)')
plt.title('Enthalpy vs Time')
plt.show()


# conver the data frame into an array
t_data = np.array(df['time'])
a_data = np.array(df['Area'])


# Define the DH function
def DH(t_data, DHinf, k, tzero, n):
    return DHinf * (1 - np.exp(-k * np.abs(t_data - tzero)**n))

# Provide the intial guess 
initial_guess = (6.19, 0.05, 1, 1)

# Fit the function
params, covariance = curve_fit(DH, t_data, a_data, p0=initial_guess)

# Fitted parameters estraction 
DHinf_fit, k_fit, tzero_fit, n_fit = params

print("Fitted DHinf:", DHinf_fit)
print("Fitted k:", k_fit)
print("Fitted tzero:", tzero_fit)
print("Fitted n:", n_fit)
print(covariance)

def fitted_DH(t_data, DHinf_fit, k_fit, tzero_fit, n_fit):
    return DH(t_data, DHinf_fit, k_fit, tzero_fit, n_fit)

# Plot the original data and the fitted function
plt.scatter(t_data, a_data, label='Data')
plt.plot(t_data, fitted_DH(t_data), label='Fitted DH Function', color='red')
plt.xlabel('Time (min)')
plt.ylabel('DH (J/g)')
plt.title('Enthalpy vs Time Fit')
plt.legend()
plt.show()

# Define two new arrays for the linear fit
x = np.abs(t_data - tzero_fit)
y = np.log(1 - a_data / DHinf_fit)

# Linear equation
def LogDHn(x, k_lin, n_lin):
    return -k_lin * np.abs(x**n_lin)

# Calculate the coefficients
coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef)
print(coef)

# Linear plot
plt.plot(x,y, 'ro', label='Data')
plt.plot(x, poly1d_fn(x), '--k', label='Fit')
plt.legend()
plt.xlabel('t-tzero')
plt.ylabel('1-ln(H1/H0)')
plt.title('Linear fit')






