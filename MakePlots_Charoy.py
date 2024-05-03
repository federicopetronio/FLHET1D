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
Results     = "Results/test_Charoy_codecheck"
PLOT_VARS   = True
ResultConfig = Results+'/Configuration.cfg'

##########################################################
#           CONFIGURE PHYSICAL PARAMETERS
##########################################################

configFile = ResultConfig
config = configparser.ConfigParser()
config.read(configFile)

physicalParameters = config['Physical Parameters']

VG       = float(physicalParameters['Gas velocity'])                # Gas velocity
M        = float(physicalParameters['Ion Mass'])*phy_const.m_u      # Ion Mass
m        = phy_const.m_e                                            # Electron mass
R1       = float(physicalParameters['Inner radius'])                # Inner radius of the thruster
R2       = float(physicalParameters['Outer radius'])                # Outer radius of the thruster
A0       = np.pi * (R2 ** 2 - R1 ** 2)                              # Area of the thruster
LENGTH   = float(physicalParameters['Length of axis'])              # length of Axis of the simulation
L0       = float(physicalParameters['Length of thruster'])          # length of thruster (position of B_max)
alpha_B1 = float(physicalParameters['Anomalous transport alpha_B1'])# Anomalous transport 1st coeff.
alpha_B2 = float(physicalParameters['Anomalous transport alpha_B2'])# Anomalous transport 2nd coeff.
mdot     = float(physicalParameters['Mass flow'])                   # Mass flow rate of propellant
Te_Cath  = float(physicalParameters['e- Temperature Cathode'])         # Electron temperature at the cathode
Rext     = float(physicalParameters['Ballast resistor'])            # Resistor of the ballast
V0       = float(physicalParameters['Voltage'])                     # Potential difference
LX       = float(physicalParameters["Length of axis"])              # length of Axis of the simulation
LTHR     = float(physicalParameters["Length of thruster"])          # length of thruster (position of B_max)
mdot     = float(physicalParameters["Mass flow"])                   # Mass flow rate of propellant
Te_Cath  = float(physicalParameters["e- Temperature Cathode"])      # Electron temperature at the cathode
TE0      = float(physicalParameters["Initial e- temperature"])      # Initial electron temperature at the cathode.
NI0      = float(physicalParameters["Initial plasma density"])      # Initial plasma density.
NG0      = float(physicalParameters["Initial neutrals density"])    # Initial neutrals density
Rext     = float(physicalParameters["Ballast resistor"])            # Resistor of the ballast
V        = float(physicalParameters["Voltage"])                     # Potential difference
Circuit  = bool(
    config.getboolean("Physical Parameters", "Circuit", fallback=False))# RLC Circuit
Estar    = float(physicalParameters["Crossover energy"])            # Crossover energy

##########################################################
#           NUMERICAL PARAMETERS
##########################################################
NumericsConfig = config['Numerical Parameteres']

NBPOINTS  = int(NumericsConfig['Number of points'])             # Number of cells
SAVERATE  = int(NumericsConfig['Save rate'])                    # Rate at which we store the data
CFL       = float(NumericsConfig['CFL'])                        # Nondimensional size of the time step
TIMEFINAL = float(NumericsConfig['Final time'])                 # Last time of simulation
Results   = NumericsConfig['Result dir']                        # Name of result directory
TIMESCHEME= NumericsConfig['Time integration']                  # Name of result directory

Delta_x  = LENGTH/NBPOINTS

##########################################################
#           Compute preliminary parameters
##########################################################
x_center = np.linspace(Delta_x, LX - Delta_x, NBPOINTS)  # Mesh in the center of cell

alpha_B = (np.ones(NBPOINTS) * alpha_B1)  # Anomalous transport coefficient inside the thruster
alpha_B = np.where(x_center < LTHR, alpha_B, alpha_B2)  # Anomalous transport coefficient in the plume
alpha_B_smooth = np.copy(alpha_B)
# smooth between alpha_B1 and alpha_B2
for index in range(10, NBPOINTS - 9):
    alpha_B_smooth[index] = np.mean(alpha_B[index-10:index+10])
alpha_B = alpha_B_smooth


##########################################################
#           Make the plots
##########################################################

# plt.style.use('classic')
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.weight"] = 'normal'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["font.family"]   = 'Times New Roman'
plt.rcParams["font.size"] = 12
plt.rcParams["lines.linewidth"] = 2
plt.rc('axes', unicode_minus=False)
axisfontsize = 12
tickfontsize = 8


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
    Estar   = 50    # Crossover energy

    Kel = 0.                                                        # Electron - neutral  collision rate
    # Kel value before I changed the code for Charoy's test cases.
    # Kel = 2.5e-13

    sigma = 2.*Te/Estar  # SEE yield
    sigma[sigma > 0.986] = 0.986

    nu_iw = np.zeros(Te.shape, dtype=float)                         # Ion - wall collision rate
    # nu_iw value before Martin changed the code for Charoy's test cases.    
    # nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)
    #   Limit the collisions to inside the thruster
    # index_L0         = np.argmax(x_center > L0)
    # nu_iw[index_L0:] = 0.
    
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
plt.close()

for i_save, file in enumerate(files):

    print("Preparing plot for i = ", i_save)
    with open(file, 'rb') as f:
        [t, P, U, P_Inlet, P_Outlet, J, V, B, x_center] = pickle.load(f)

    if PLOT_VARS:
        E = compute_E(P,J)
        phi = compute_phi(P,J)
        
        f = plt.figure(figsize = (12,9))

        ax1 = plt.subplot((4,2), (0,0))
        ax2 = plt.subplot((4,2), (1,0))
        ax3 = plt.subplot((4,2), (2,0))
        ax4 = plt.subplot((4,2), (3,0))
        ax5 = plt.subplot((4,2), (0,1))
        ax6 = plt.subplot((4,2), (1,1))
        ax7 = plt.subplot((4,2), (2,1))
        ax8 = plt.subplot((4,2), (3,1))

        ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        ax_b=ax[0].twinx()
        ax[0].plot(x_center, P[0,:], linewidth=1.8, markersize=3)
        ax[0].set_ylabel(r'$n_g$ [m$^{-3}$]', fontsize=axisfontsize)
        ax[0].set_xticklabels([])
        ax[0].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[0].set_ylim([0,max(P[0,:])*1.1])
        ax_b.plot(x_center, B, 'r:', linewidth=0.9, markersize=3)
        ax_b.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.set_ylabel("B-field [Tesla]", color='r', fontsize=axisfontsize)

        ax_phi=ax[1].twinx()
        ax[1].plot(x_center*100, E, linewidth=1.8, markersize=3)
        ax[1].set_ylabel(r'$E$ [V/m]', fontsize=axisfontsize)
        ax[1].set_xticklabels([])
        ax[1].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[1].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_phi.plot(x_center*100, phi, color='r', linewidth=1.8, markersize=3)
        ax_phi.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_phi.set_ylabel(r'$V$ [V]', color='r')

        ax_b=ax[2].twinx()
        ax[2].plot(x_center, P[1,:], linewidth=1.8, markersize=3)
        ax[2].set_ylabel(r'$n_i$ [m$^{-3}$]', fontsize=axisfontsize)
        ax[2].set_xticklabels([])
        ax[2].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])
        
        max_ni = 1e17
        if max(P[1,:]) > 1e18:
            max_ni = max(P[1,:])*1.5
        elif max(P[1,:]) > 1e17:
            max_ni = 1e18
        else:
            max_ni = 1e17
            
        ax[2].set_ylim([0, max_ni])
        
        ax[3].plot(time/1e-3, Current)
        ax[3].set_ylabel(r'Current [A]', fontsize=axisfontsize)
        ax[3].set_xlabel(r'time [ms]', fontsize=axisfontsize)
        ax[3].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)            
        ax[3].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)        
        ax[3].plot(time[i_save]/1e-3, Current[i_save], 'ro', markersize=10)

        ax_b=ax[4].twinx()
        ax[4].plot(x_center, P[3,:], linewidth=1.8, markersize=3)
        ax[4].set_ylabel(r'$T_e$ [eV]', fontsize=axisfontsize)
        ax[4].set_xticklabels([])
        ax[4].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])

        ax_b=ax[5].twinx()
        ax[5].plot(x_center*100, P[2,:]/1000, linewidth=1.8, markersize=3)
        ax[5].plot(x_center*100, np.sqrt(phy_const.e*P[3,:]/(131.293*phy_const.m_u))/1000.,'g--', linewidth=1.8, markersize=3)
        ax[5].set_ylabel('$v_i$ (b-),\n$v_B$ (g--) [km/s]', fontsize=axisfontsize)
        ax[5].set_xticklabels([])
        ax[5].set_xlabel(r'$x~[cm]$', fontsize=axisfontsize, weight = 'bold')
        ax_b.plot(x_center*100, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])

        ax_b=ax[6].twinx()
        ax[6].plot(x_center*100, P[4,:]/1000, linewidth=1.8, markersize=3)
        ax[6].set_xlabel(r'$x~[cm]$', fontsize=axisfontsize, weight = 'bold')
        ax[6].set_ylabel(r'$v_e$ [km/s]', fontsize=axisfontsize)
        ax[6].set_xticklabels([])
        ax[6].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:', linewidth=0.7, markersize=3)
        ax_b.set_yticklabels([])

        j_of_x = P[1,:]*phy_const.e(*P[4,:] - P[2,:])
        ax[7].plot(time/1e-3, Current)
        ax[7].set_ylabel(r'Current density [$A.m^{-2}$]', fontsize=axisfontsize)
        ax[7].set_xlabel(r'time [ms]', fontsize=axisfontsize)
        ax[7].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[7].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)        
        ax[7].plot(time[i_save]/1e-3, Current[i_save], 'ro', markersize=10)

        title = 'time = '+ str(round(t/1e-6, 4))+'$\mu$s'
        f.suptitle(title, y=1.05)

        for axis in ax:
            axis.grid(True)
        
        plt.tight_layout()

        plt.subplots_adjust(wspace = 0.4, hspace=0.2)
        
        plt.savefig(ResultsFigs+"/MacroscopicVars_New_"+str(i_save)+".png", bbox_inches='tight')
        plt.close()
    

os.system("ffmpeg -r 10 -i "+ResultsFigs+"/MacroscopicVars_New_%d.png -vcodec mpeg4 -y -vb 20M "+ResultsFigs+"Evolution.mp4")

