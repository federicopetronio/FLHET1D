import numpy as np
import scipy.constants as phy_const
import matplotlib.pyplot as plt
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import glob
import sys
import configparser



##########################################################
#           POST-PROC PARAMETERS
##########################################################
Results     = sys.argv[1]
PLOT_VARS   = True
ResultConfig = Results+'/Configuration.cfg'

##########################################################
#           CONFIGURE PHYSICAL PARAMETERS
##########################################################

configFile = ResultConfig
config = configparser.ConfigParser()
config.read(configFile)

physicalParameters = config['Physical Parameters']

VG       = float(physicalParameters['Gas velocity'])                 # Gas velocity
M        = float(physicalParameters['Ion Mass'])*phy_const.m_u       # Ion Mass
m        = phy_const.m_e                                             # Electron mass
R1       = float(physicalParameters['Inner radius'])                 # Inner radius of the thruster
R2       = float(physicalParameters['Outer radius'])                 # Outer radius of the thruster
A0       = np.pi * (R2 ** 2 - R1 ** 2)                               # Area of the thruster
LENGTH   = float(physicalParameters['Length of axis'])               # length of Axis of the simulation
L0       = float(physicalParameters['Length of thruster'])           # length of thruster (position of B_max)
alpha_B  = float(physicalParameters['Anomalous transport alpha_B']) # Anomalous transport
mdot     = float(physicalParameters['Mass flow'])                    # Mass flow rate of propellant
Te_Cath  = float(physicalParameters['Temperature Cathode'])          # Electron temperature at the cathode
Rext     = float(physicalParameters['Ballast resistor'])             # Resistor of the ballast
V0       = float(physicalParameters['Voltage'])                      # Potential difference

##########################################################
#           NUMERICAL PARAMETERS
##########################################################
NumericsConfig = config['Numerical Parameteres']

NBPOINTS  = int(NumericsConfig['Number of points'])             # Number of cells
SAVERATE  = int(NumericsConfig['Save rate'])                    # Rate at which we store the data
CFL       = float(NumericsConfig['CFL'])                        # Nondimensional size of the time step
TIMEFINAL = float(NumericsConfig['Final time'])                 # Last time of simulation
Results   = NumericsConfig['Result dir']                        # Name of result directory
TIMESCHEME = NumericsConfig['Time integration']                        # Name of result directory

Delta_x  = LENGTH/NBPOINTS

##########################################################
#           Make the plots
##########################################################


plt.style.use('classic')
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.weight"] = 'normal'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["font.family"]   = 'Times New Roman'
plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 2

ResultsFigs = Results+"/Figs"
ResultsData = Results+"/Data"
if not os.path.exists(ResultsFigs):
    os.makedirs(ResultsFigs)

# open all the files in the directory and sort them to do the video in order
files       = glob.glob(ResultsData + "/*.pkl")
filesSorted = sorted(files, key = lambda x: os.path.getmtime(x), reverse=True)
files.sort(key=os.path.getmtime)


Current = np.zeros(np.shape(files)[0])
Voltage = np.zeros(np.shape(files)[0])
time    = np.zeros(np.shape(files)[0])

def compute_phi(P, Current):
    def cumtrapz(y, d):
        return np.concatenate((np.zeros(1), np.cumsum(d * (y[1:] + y[:-1]) / 2.0)))
    
    E   = compute_E(P, Current)
    phi = V - Current * Rext - cumtrapz(E, d=Delta_x)  # Discharge electrostatic potential
    return phi
    
def compute_E(P, Current):
    
    def trapz(y, d):
        return np.sum( (y[1:] + y[:-1]) )*d/2.0
    def gradient(y, d):
        dp_dz = np.empty_like(y)
        dp_dz[1:-1] = (y[2:] - y[:-2]) / (2 * d)
        dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
        dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

        return dp_dz

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0,:]
    ni = P[1,:]
    ui = P[2,:]
    Te = P[3,:]
    ve = P[4,:]
    Gamma_i = ni*ui
    wce     = phy_const.e*B/m              # electron cyclotron frequency
    
    #############################
    #       Compute the rates   #
    #############################
    Eion    = 12.1  # Ionization energy
    gamma_i = 3     # Excitation coefficient
    Estar   = 50    # Crossover energy

    Kiz = 1.8e-13*(((1.5*Te)/Eion)**0.25)*np.exp(- 4*Eion/(3*Te))  # Ion - neutral  collision rate          TODO: Replace by better
    Kel = 2.5e-13                                                  # Electron - neutral  collision rate     MARTIN: Check this

    sigma = 2.*Te/Estar  # SEE yield
    sigma[sigma > 0.986] = 0.986

    nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate
    #Limit the collisions to inside the thruster
    index_L0         = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.
    
    nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate

    nu_m   = ng*Kel + alpha_B*wce + nu_ew                          # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))       # Effective mobility
    dp_dz  = gradient(ni*Te, d = Delta_x)
    
    I0 = Current/(phy_const.e*A0)
    
    E = (I0 - Gamma_i) / (mu_eff * ni) - dp_dz / ni  # Discharge electric field
    return E

#####################################
#           Plot variables
#####################################

for i_save, file in enumerate(files):
    
    with open(file, 'rb') as f:
        [t, P, U, P_Inlet, P_Outlet, J, V, B, x_center] = pickle.load(f)
    
    # Save the current
    Current[i_save] = J
    Voltage[i_save] = V
    time[i_save]    = t

#####################################
#           Plot current
#####################################

f, ax = plt.subplots(figsize=(8,3))

ax.plot(time/1e-3, Current)
ax.set_xlabel(r'$t$ [ms]', fontsize=18, weight = 'bold')
ax.set_ylabel(r'Current [A]', fontsize=18)
ax_V=ax.twinx()
ax_V.plot(time/1e-3, Voltage,'r')
ax_V.set_ylabel(r'Voltage [V]', fontsize=18)
ax.grid(True)
plt.tight_layout()
plt.savefig(ResultsFigs+"/Current.pdf", bbox_inches='tight')
    
for i_save, file in enumerate(files):

    print("Preparing plot for i = ", i_save)
    #
    with open(file, 'rb') as f:
        [t, P, U, P_Inlet, P_Outlet, J, V, B, x_center] = pickle.load(f)

    if PLOT_VARS:
        E = compute_E(P,J)
        phi = compute_phi(P,J)
        
#        f, ax = plt.subplots(5, 1, figsize = (5,8))
#
#        ax_b=ax[0].twinx()
#        ax[0].plot(x_center, P[0,:], color=(255/255,97/255,3/255), linewidth=1.8, markersize=3)
#        ax_b.plot(x_center, B, ':', color=(0,128./255.,0.), linewidth=0.9, markersize=3)
#        ax_b.set_yticks([0, max(B)])
#        ax_b.set_ylabel(r'$B$ [T]', color=(0,128./255.,0.))
#        ax[0].set_ylabel(r'$n_g$ [m$^{-3}$]', fontsize=18)
#        ax[0].set_xticklabels([])
#        ax[0].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
#        ax[0].legend(loc = 'lower center', fontsize = 12)
#        ax[0].set_ylim([0,max(P[0,:])*1.1])
#
#        ax_b=ax[1].twinx()
#        ax[1].plot(x_center, P[1,:], color=(255/255,97/255,3/255), linewidth=1.8, markersize=3)
#        ax_b.plot(x_center, B, ':', color=(0,128./255.,0.), linewidth=0.9, markersize=3)
#        ax_b.set_yticklabels([])
#        #ax[1].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
#        ax[1].set_ylabel(r'$n_i$ [m$^{-3}$]', fontsize=18)
#        #ax[1].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
#        ax[1].set_xticklabels([])
#        ax[1].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
#        max_ni = 1e17
#        if max(P[1,:]) > 1e18:
#            max_ni = max(P[1,:])*1.5
#        elif max(P[1,:]) > 1e17:
#            max_ni = 1e18
#        else:
#            max_ni = 1e17
#        ax[1].set_ylim([0, max_ni])
#
#        ax_b=ax[2].twinx()
#        ax[2].plot(x_center, P[2,:]/1000, color=(255/255,97/255,3/255), linewidth=1.8, markersize=3)
#        ax_b.plot(x_center, B, ':', color=(0,128./255.,0.), linewidth=0.9, markersize=3)
#        ax_b.set_yticklabels([])
#        #ax[2].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
#        ax[2].set_ylabel(r'$v_i$ [km/s]', fontsize=18)
#        #ax[2].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
#        ax[2].set_xticklabels([])
#        ax[2].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
#        ax[2].legend(loc = 'lower center', fontsize = 12)
#
#        ax_b=ax[3].twinx()
#        ax[3].plot(x_center, P[3,:], color=(255/255,97/255,3/255), linewidth=1.8, markersize=3)
#        ax_b.plot(x_center, B, ':', color=(0,128./255.,0.), linewidth=0.9, markersize=3)
#        ax_b.set_yticklabels([])
#        #ax[3].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
#        ax[3].set_ylabel(r'$T_e$ [eV]', fontsize=18)
#        #ax[3].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
#        ax[3].set_xticklabels([])
#        ax[3].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
#        ax[3].legend(loc = 'lower center', fontsize = 12)
#        f.suptitle('t = ', fontname = 'Times New Roman',fontsize=16, weight = 'bold')
#
#        ax[4].plot(x_center*100, P[4,:]/1000., color=(255/255,97/255,3/255), linewidth=1.8, markersize=3)
#        ax[4].set_xlabel(r'$x~[cm]$', fontsize=18, weight = 'bold')
#        ax[4].set_ylabel(r'$v_e$ [km/s]', fontsize=18)
#        ax[4].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
#        ax[4].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
#        ax[4].legend(loc = 'lower center', fontsize = 12)
#        title = 'time = '+ str(round(t/1e-6, 4))+'$\mu$s'
#        plt.suptitle(title, y=1.05)
#
#        for axis in ax:
#            axis.grid(True)
#        plt.tight_layout()
#
#        plt.subplots_adjust(hspace=0.1)
#        plt.savefig(ResultsFigs+"/MacroscopicVars_"+str(i_save)+".png", bbox_inches='tight')
#        plt.close()
#
#
        
        f = plt.subplots(figsize = (8,7))

        ax1 = plt.subplot(4,2,1)
        ax2 = plt.subplot(4,2,2)
        ax3 = plt.subplot(4,2,3)
        ax4 = plt.subplot(4,2,4)
        ax5 = plt.subplot(4,2,5)
        ax6 = plt.subplot(4,2,6)
        ax7 = plt.subplot(4,1,4)

        ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

        ax_b=ax[0].twinx()
        ax[0].plot(x_center, P[0,:], linewidth=1.8, markersize=3)
        ax_b.plot(x_center, B, 'r:', linewidth=0.9, markersize=3)
        ax_b.set_yticks([0, max(B)])
        #ax_b.set_ylabel(r'$B$ [T]', color='r')
        #ax[0].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
        ax[0].set_ylabel(r'$n_g$ [m$^{-3}$]', fontsize=18)
        #ax[0].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
        ax[0].set_xticklabels([])
        ax[0].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        #ax[0].legend(loc = 'lower center', fontsize = 12)
        ax[0].set_ylim([0,max(P[0,:])*1.1])

        ax_phi=ax[1].twinx()
        ax[1].plot(x_center*100, E, linewidth=1.8, markersize=3)
        ax_phi.plot(x_center*100, phi, color='r', linewidth=1.8, markersize=3)
        ax_phi.set_ylabel(r'$V$ [V]', color='r')
        ax[1].set_ylabel(r'$E$ [V/m]', fontsize=18)
        ax[1].set_xticklabels([])
        ax[1].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        ax[1].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)

        ax_b=ax[2].twinx()
        ax[2].plot(x_center, P[1,:], linewidth=1.8, markersize=3)
        ax_b.plot(x_center, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])
        #ax[2].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
        ax[2].set_ylabel(r'$n_i$ [m$^{-3}$]', fontsize=18)
        #ax[2].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
        ax[2].set_xticklabels([])
        ax[2].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        #ax[1].legend(loc = 'lower center', fontsize = 12)
        
        max_ni = 1e17
        if max(P[1,:]) > 1e18:
            max_ni = max(P[1,:])*1.5
        elif max(P[1,:]) > 1e17:
            max_ni = 1e18
        else:
            max_ni = 1e17
            
        ax[2].set_ylim([0, max_ni])
        
        ax_b=ax[3].twinx()
        ax[3].plot(x_center, P[3,:], linewidth=1.8, markersize=3)
        ax_b.plot(x_center, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])
        #ax[3].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
        ax[3].set_ylabel(r'$T_e$ [eV]', fontsize=18)
        #ax[3].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
        ax[3].set_xticklabels([])
        ax[3].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        #ax[3].legend(loc = 'lower center', fontsize = 12)
        #f.suptitle('t = ', fontname = 'Times New Roman',fontsize=16, weight = 'bold')

        ax_b=ax[4].twinx()
        ax[4].plot(x_center*100, P[2,:]/1000, linewidth=1.8, markersize=3)
        ax[4].plot(x_center*100, np.sqrt(phy_const.e*P[3,:]/(131.293*phy_const.m_u))/1000.,'g--', linewidth=1.8, markersize=3)
        ax_b.plot(x_center, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])
        #ax[4].set_xlabel(r'$x~[m]$', fontsize=18, weight = 'bold')
        ax[4].set_ylabel(r'$v_i$ [km/s]', fontsize=18)
        ax[4].set_xlabel(r'$x~[cm]$', fontsize=18, weight = 'bold')
        #ax[4].xaxis.set_tick_params(which='both', size=0, width=1.5, labelsize=0)
        #ax[4].set_xticklabels([])
        #ax[4].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        ax[4].legend(loc = 'lower center', fontsize = 12)

        ax[5].plot(x_center*100, P[4,:]/1000, linewidth=1.8, markersize=3)
        ax[5].set_xlabel(r'$x~[cm]$', fontsize=18, weight = 'bold')
        ax[5].set_ylabel(r'$v_e$ [km/s]', fontsize=18)
        ax[5].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        ax[5].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=13)
        #ax[4].legend(loc = 'lower center', fontsize = 12)
        title = 'time = '+ str(round(t/1e-6, 4))+'$\mu$s'
        f[0].suptitle(title, y=1.05)

        ax[6].plot(time/1e-3, Current)
        ax[6].set_ylabel(r'Current [A]', fontsize=18)
        ax[6].set_xlabel(r'time [ms]', fontsize=18)
        ax[6].plot(time[i_save]/1e-3, Current[i_save], 'ro', markersize=10)
        ax[6].grid(True)
        ax[6].set_xlim([0,3.])

        for axis in ax:
            axis.grid(True)
            #axis.get_legend().remove()
            
        ax[0].legend(fontsize = 10, loc='lower right')
        plt.tight_layout()

        plt.subplots_adjust(wspace = 0.4, hspace=0.2)
        
        plt.savefig(ResultsFigs+"/MacroscopicVars_New_"+str(i_save)+".png", bbox_inches='tight')
        plt.close()
    

os.system("ffmpeg -r 10 -i "+ResultsFigs+"/MacroscopicVars_New_%d.png -vcodec mpeg4 -y -vb 20M "+ResultsFigs+"Evolution.mp4")

