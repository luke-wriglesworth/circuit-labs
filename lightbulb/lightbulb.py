import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 16, 'font.family': 'serif', 'figure.figsize': (12, 6)})

# read in data
header = ["voltage","current"]
raw_data = np.genfromtxt('lightbulb-data.csv', delimiter=',', skip_header=3, names=header)
voltage = raw_data["voltage"]
current = raw_data["current"]

# sort data by voltage 
sort_index = np.argsort(voltage)
voltage = voltage[sort_index]
current = current[sort_index]

# Uncertainties
# we are in the 30mA range, so the uncertainty is 0.2% of the reading + 5*(0.001). All measurements have same number of digits.
unc_current = 0.002*current + 5*0.001

# fit data to power law, power law with b=3/5, and fit to log of power law
def I(V,a,b):
    return a*V**b
def I_ideal(V,a):
    return a*V**(3/5)
def log_I(V,a,b):
    return a+b*V
def chiSqr(data, model, unc):
    return np.sum((data - model)**2 / unc**2)
def reducedChiSqr(data, model, unc, nParams):
    return chiSqr(data,model,unc) / (len(data) - nParams)
def model_uncertainty(a,b,unc_a,unc_b,V):
    return np.sqrt((a*b*V**(b-1)*unc_b)**2 + (V**b*unc_a)**2)
def residual_uncertainty(popt,param_unc,voltage):
    return np.sqrt(unc_current**2 + model_uncertainty(*popt,*param_unc,voltage)**2)

# take log of data for log fitting
log_current = np.log(current)
log_voltage = np.log(voltage)

popt, pcov = curve_fit(I, voltage, current, sigma=unc_current, absolute_sigma=True)
popt_ideal, pcov_ideal = curve_fit(I_ideal, voltage, current, sigma=unc_current, absolute_sigma=True)
popt_log, pcov_log = curve_fit(log_I, log_voltage, log_current)

red_chi_sqr = reducedChiSqr(current,I(voltage,*popt),unc_current,2)
red_chi_sqr_ideal = reducedChiSqr(current,I_ideal(voltage,*popt_ideal),unc_current,1)
red_chi_sqr_log = reducedChiSqr(current,I(voltage,np.exp(popt_log[0]),popt_log[1]),unc_current,2)   

param_unceratinty = np.sqrt(np.diag(pcov))
param_uncertainty_ideal = np.sqrt(np.diag(pcov_ideal))
param_uncertainty_log = np.sqrt(np.diag(pcov_log))

print("Ideal Power Law")
print("a = ", popt_ideal[0], "+/-", param_uncertainty_ideal[0])
print(f"b={3/5}")
print(f"Reduced Chi Squared: {red_chi_sqr_ideal}")
print("--------------------------------------")
print("I(V) = a*V^b")
print("a = ", popt[0], "+/-", param_unceratinty[0])
print("b = ", popt[1], "+/-", param_unceratinty[1])
print(f"Reduced Chi Squared: {red_chi_sqr}")
print("--------------------------------------")
print("log(I(V)) = log(a) + b*log(V)")
print("a = ", np.exp(popt_log[0]), "+/-", param_uncertainty_log[0]*np.exp(popt_log[0]))
print("b = ", popt_log[1], "+/-", param_uncertainty_log[1])
print(f"Reduced Chi Squared: {red_chi_sqr_log}")

# power law residuals
residuals = current - I(voltage, *popt)
res_uncertainty = residual_uncertainty(popt,param_unceratinty,voltage)

# power law w/ b=3/5 residuals
residuals_ideal = current - I_ideal(voltage, *popt_ideal)
res_uncertainty_ideal = residual_uncertainty([popt_ideal,3/5],[param_uncertainty_ideal,0],voltage)

# log fitting residuals
residuals_log = current - I(voltage, np.exp(popt_log[0]), popt_log[1])
log_model_uncertainty = current*np.sqrt((np.log(voltage)*param_uncertainty_log[1])**2 + (param_uncertainty_log[0]/np.exp(popt_log[0]))**2)
res_uncertainty_log = np.sqrt(unc_current**2 + log_model_uncertainty**2)

# plot fit and residuals
fig, axs = plt.subplots(2,1,layout="constrained",gridspec_kw={'height_ratios': [3, 2]},sharex=True,figsize=(12,8))
plotVals = np.linspace(min(voltage)-1,max(voltage)+1,1000)
axs[0].set_title(r"$I(V)$ Power Law for Lightbulb")
axs[0].errorbar(voltage,current,label="Measurements & Uncertainties",color="black",yerr=unc_current,fmt="o",capsize=6,markersize=4)
axs[0].plot(voltage,I(voltage,*popt),label=r"$I(V)=aV^b$",color="red")
axs[0].plot(voltage,I_ideal(voltage,*popt_ideal),label="b=3/5 Fitting",color="green")
axs[0].plot(voltage,I(voltage,np.exp(popt_log[0]),popt_log[1]),label="Log Fitting",color="blue", ls="--")
axs[0].set_ylabel("Current (mA)")
axs[0].legend()
axs[0].grid()
axs[1].errorbar(voltage,residuals,label=r"$I(V)=aV^b$",color="red",yerr=res_uncertainty,fmt="o",capsize=3)
axs[1].errorbar(voltage,residuals_ideal,label="b=3/5 Fitting",color="green",yerr=res_uncertainty_ideal,fmt="o",capsize=3)
axs[1].errorbar(voltage,residuals_log,label="Log Fitting",color="blue",yerr=res_uncertainty_log,fmt="o",capsize=3)
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Residuals (mA)")
axs[1].grid()
axs[1].set_ylim(np.min(residuals)-0.5,np.max(residuals)+0.5)
axs[1].legend(fontsize=12,loc="lower left")
fig.savefig("lightbulb_plots.png")

# plot fit and residuals (log scale)
fig = plt.figure(figsize=(12,6))
plotVals = np.linspace(min(voltage)-1,max(voltage)+1,1000)
plt.xscale("log")
plt.yscale("log")
plt.title(r"$I(V)$ Power Law for Lightbulb [Log Scale]")
plt.errorbar(voltage,current,label="Measurements & Uncertainties",color="black",fmt="o",yerr=unc_current,capsize=6,markersize=4)
plt.plot(voltage,I(voltage,*popt),label=r"$I(V)=aV^b$",color="red")
plt.plot(voltage,I_ideal(voltage,*popt_ideal),label="b=3/5 Fitting",color="green")
plt.plot(voltage,I(voltage,np.exp(popt_log[0]),popt_log[1]),label="Log Fitting",color="blue",ls="--")
plt.xlabel(r"Log(V)")
plt.ylabel(r"Log(I)")
plt.xlim(5.2,17)
plt.ylim(12,30)
plt.legend()
plt.grid()
plt.show()
fig.savefig("lightbulb_plots_log.png")




