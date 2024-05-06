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
LX       = float(physicalParameters['Length of axis'])               # length of Axis of the simulation
LTHR     = float(physicalParameters['Length of thruster'])           # length of thruster (position of B_max)
alpha_B1 = float(
    physicalParameters["Anomalous transport alpha_B1"]
)  # Anomalous transport
alpha_B2 = float(
    physicalParameters["Anomalous transport alpha_B2"]
)  # Anomalous transport
mdot     = float(physicalParameters['Mass flow'])                    # Mass flow rate of propellant
Te_Cath  = float(physicalParameters['e- Temperature Cathode'])          # Electron temperature at the cathode
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

Delta_x  = LX/NBPOINTS


# creates the array resulting 2 regions alpha_B
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


#plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 2
tickfontsize = 13
axisfontsize = 14

ResultsFigs = Results+"/Figs"
ResultsData = Results+"/Data"
if not os.path.exists(ResultsFigs):
    os.makedirs(ResultsFigs)

# open all the files in the directory and sort them to do the video in order
files       = glob.glob(ResultsData + "/*.pkl")
filesSorted = sorted(files, key = lambda x: os.path.getmtime(x), reverse=True)
files.sort(key=os.path.getmtime)


Current = np.zeros(np.shape(files)[0])
CurrentDensity = np.zeros(np.shape(files)[0])
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
    Estar   = 50    # Crossover energy

    Kel = 0.     # Electron - neutral  collision rate     MARTIN: Check this

    sigma = 2.*Te/Estar  # SEE yield
    sigma[sigma > 0.986] = 0.986

    nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate
    #Limit the collisions to inside the thruster
    index_L0         = np.argmax(x_center > LX)
    nu_iw[index_L0:] = 0.
    
    nu_ew   = nu_iw/(1 - sigma)                                      # Electron - wall collision rate

    nu_m    = ng*Kel + alpha_B*wce + nu_ew                          # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))       # Effective mobility
    dp_dz  = np.gradient(ni*Te, Delta_x)

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
    CurrentDensity[i_save] = np.mean(P[1,:]*phy_const.e*(P[2,:] - P[4,:]))
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

    with open(file, 'rb') as f:
        [t, P, U, P_Inlet, P_Outlet, J, V, B, x_center] = pickle.load(f)

    if PLOT_VARS:
        E = compute_E(P,J)
        phi = compute_phi(P,J)
        
        f = plt.figure(figsize = (8,7))

        ax1 = plt.subplot2grid((4,2),(0,0))
        ax2 = plt.subplot2grid((4,2),(1,0))
        ax3 = plt.subplot2grid((4,2),(2,0))
        ax4 = plt.subplot2grid((4,2),(3,0))
        ax5 = plt.subplot2grid((4,2),(0,1))
        ax6 = plt.subplot2grid((4,2),(1,1))
        ax7 = plt.subplot2grid((4,2),(2,1))
        ax8 = plt.subplot2grid((4,2),(3,1))

        ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        ax_b=ax[0].twinx()
        ax[0].plot(x_center*100, P[0,:])
        ax[0].set_ylabel(r'n_g [m$^{-3}$]', fontsize=axisfontsize)
        ax[0].set_xticklabels([])
        ax[0].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[0].set_ylim([0,max(P[0,:])*1.1])
        ax[0].set_ylabel('B-field [T]', color='r', fontsize=axisfontsize)
        ax_b.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')

        ax_phi=ax[1].twinx()
        ax[1].plot(x_center*100, E)
        ax[1].set_ylabel(r'E [V/m]', fontsize=axisfontsize)
        ax[1].set_xticklabels([])
        ax[1].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[1].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_phi.plot(x_center*100, phi, color='r')
        ax_phi.set_ylabel('V [V]', color='r', fontsize=axisfontsize)
        ax_phi.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)

        ax_b=ax[2].twinx()
        ax[2].plot(x_center*100, P[1,:])
        ax[2].set_ylabel('n_i [m$^{-3}$]', fontsize=axisfontsize)
        ax[2].set_xlabel('x [cm]', fontsize=axisfontsize)        
        ax[2].set_xticklabels([])
        ax[2].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[2].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        max_ni = 1e17
        if max(P[1,:]) > 1e18:
            max_ni = max(P[1,:])*1.5
        elif max(P[1,:]) > 1e17:
            max_ni = 1e18
        else:
            max_ni = 1e17
            
        ax[2].set_ylim([0, max_ni])
        
        ax_b=ax[4].twinx()
        ax[4].plot(x_center*100, P[3,:])
        ax[4].set_ylabel('T_e [eV]', fontsize=axisfontsize)
        ax[4].set_xticklabels([])
        ax[4].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        ax_b=ax[5].twinx()
        ax[5].plot(x_center*100, P[2,:]/1000)
        ax[5].plot(x_center*100, np.sqrt(phy_const.e*P[3,:]/(131.293*phy_const.m_u))/1000.,'g--')
        ax[5].set_ylabel('$v_i$ [km/s]', fontsize=axisfontsize)
        ax[5].set_xticklabels([])
        ax_b.plot(x_center, B, 'r:')
        ax_b.set_yticklabels([])

        ax_b = ax[6].twinx()
        ax[6].plot(x_center*100, P[4,:]/1000)
        ax[6].set_ylabel('$v_e$ [km/s]', fontsize=axisfontsize)
        ax[6].set_xticklabels([])
        ax[6].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)


        j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
        ax[6].plot(x_center*100, j_of_x)
        ax[6].set_ylabel('Current Density [A.m$^{-2}$]', fontsize=axisfontsize)
        ax[6].set_xlabel('x [cm]', fontsize=axisfontsize)
        ax[6].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[6].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)

        title = 'time = '+ str(round(t/1e-6, 4))+'$\mu$s'
        f.suptitle(title)

        for axis in ax:
            axis.grid(True)
            #axis.get_legend().remove()
            
        ax[0].legend(fontsize = 10, loc='lower right')
        plt.tight_layout()

        plt.subplots_adjust(wspace = 0.4, hspace=0.2)
        
        plt.savefig(ResultsFigs+"/MacroscopicVars_New_"+str(i_save)+".png", bbox_inches='tight')
        plt.close()
    

os.system("ffmpeg -r 10 -i "+ResultsFigs+"/MacroscopicVars_New_%d.png -vcodec mpeg4 -y -vb 20M "+ResultsFigs+"Evolution.mp4")

