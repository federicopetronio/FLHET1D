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
from numba import njit



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

VG = float(physicalParameters["Gas velocity"])  # Gas velocity
M = float(physicalParameters["Ion Mass"]) * phy_const.m_u  # Ion Mass
m = phy_const.m_e  # Electron mass
R1 = float(physicalParameters["Inner radius"])  # Inner radius of the thruster
R2 = float(physicalParameters["Outer radius"])  # Outer radius of the thruster
A0 = np.pi * (R2**2 - R1**2)  # Area of the thruster
LX = float(physicalParameters["Length of axis"])  # length of Axis of the simulation
LTHR = float(
    physicalParameters["Length of thruster"]
)  # length of thruster (position of B_max)
alpha_B1 = float(
    physicalParameters["Anomalous transport alpha_B1"]
)  # Anomalous transport
alpha_B2 = float(
    physicalParameters["Anomalous transport alpha_B2"]
)  # Anomalous transport
mdot = float(physicalParameters["Mass flow"])  # Mass flow rate of propellant
Te_Cath = float(
    physicalParameters["e- Temperature Cathode"]
)  # Electron temperature at the cathode
TE0 = float(physicalParameters["Initial e- temperature"]) # Initial electron temperature at the cathode.
NI0 = float(physicalParameters["Initial plasma density"]) # Initial plasma density.
Rext = float(physicalParameters["Ballast resistor"])  # Resistor of the ballast
V = float(physicalParameters["Voltage"])  # Potential difference
Circuit = bool(
    config.getboolean("Physical Parameters", "Circuit", fallback=False)
)  # RLC Circuit


# Magnetic field configuration
MagneticFieldConfig = config["Magnetic field configuration"]

if MagneticFieldConfig["Type"] == "Default":
    print(MagneticFieldConfig["Type"] + " Magnetic Field")

    BMAX = float(MagneticFieldConfig["Max B-field"])  # Max Mag field
    B0 = float(MagneticFieldConfig["B-field at 0"])  # Mag field at x=0
    BLX = float(MagneticFieldConfig["B-field at LX"])  # Mag field at x=LX
    LB1 = float(MagneticFieldConfig["Length B-field 1"])  # Length for magnetic field
    LB2 = float(MagneticFieldConfig["Length B-field 2"])  # Length for magnetic field
    saveBField = bool(MagneticFieldConfig["Save B-field"])


# Ionization source term configuration
IonizationConfig = config["Ionization configuration"]
if IonizationConfig["Type"] == "SourceIsImposed":
    print("The ionization source term is imposed as specified in T.Charoy's thesis, section 2.2.2.")
SIZMAX  = float(IonizationConfig["Maximum S_iz value"])  # Max Mag field
LSIZ1   = float(IonizationConfig["Position of 1st S_iz zero"])  # Mag field at x=0
LSIZ2   = float(IonizationConfig["Position of 2nd S_iz zero"])  # Mag field at x=LX
assert(LSIZ2 >= LSIZ1)

# Collisions parameters
CollisionsConfig = config["Collisions"]
KEL = float(CollisionsConfig["Elastic collisions reaction rate"])

# Wall interactions
WallInteractionConfig = config["Wall interactions"]
ESTAR = float(WallInteractionConfig["Crossover energy"])  # Crossover energy
assert((WallInteractionConfig["Type"] == "Default")|(WallInteractionConfig["Type"] == "None"))


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
#plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 2
plt.rc('axes', unicode_minus=False)
tickfontsize = 11
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

def GetImposedSiz(x_center):
    xm = (LSIZ1 + LSIZ2)/2
    Siz = SIZMAX*np.cos(np.pi*(x_center - xm)/(LSIZ2 - LSIZ1))
    Siz = np.where((x_center < LSIZ1)|(x_center > LSIZ2), 0., Siz)

    return Siz


@njit
def gradient(y, d):
    dp_dz = np.zeros(y.shape)
    dp_dz[1:-1] = (y[2:] - y[:-2]) / (2 * d)
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    return dp_dz


@njit
def compute_E(fP, fB, fESTAR, wall_inter_type:str, fR1, fR2, fM, fx_center, fLTHR, fKEL, falpha_B, fDelta_x, fJ):

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0,:]
    ni = fP[1,:]
    ui = fP[2,:]
    Te = fP[3,:]
    ve = fP[4,:]
    Gamma_i = ni*ui
    me = phy_const.m_e
    wce     = phy_const.e*fB/me   # electron cyclotron frequency
    
    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / fESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(fR2 - fR1))*np.sqrt(phy_const.e*Te/fM)
        # Limit the wall interactions to the inner channel
        nu_iw[fx_center > fLTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate        
    
    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)     # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)     # Electron - wall collision rate



    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_LTHR = np.argmax(z > LTHR)
    #        index_ind = index_L1 - index_LTHR + 1
    #
    #        nu_iw[index_LTHR: index_L1] = nu_iw[index_LTHR] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0

    ##################################################
    #       Compute the electron properties          #
    ##################################################

    nu_m = (
        ng * fKEL + falpha_B * wce + nu_ew
        )  # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e / (me* nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
        )  # Effective mobility    dp_dz  = np.gradient(ni*Te, Delta_x)

    I0 = fJ/(phy_const.e*A0)
    dp_dz  = gradient(ni*Te, fDelta_x)

    
    E = (I0 - Gamma_i) / (mu_eff * ni) - dp_dz / ni  # Discharge electric field
    return E


@njit
def cumTrapz(y, d):
    n = y.shape[0]
    cuminteg = np.zeros(y.shape, dtype=float)
    
    for i in range(1, n):
        cuminteg[i] = cuminteg[i-1] + d * (y[i] + y[i-1]) / 2.0

    return cuminteg


@njit
def compute_phi(fP, fB, fESTAR, wall_inter_type:str, fR1, fR2, fM, fx_center, fLTHR, fKEL, falpha_B, fDelta_x, fJ, fV, fRext):
    
    E   = compute_E(fP, fB, fESTAR, wall_inter_type, fR1, fR2, fM, fx_center, fLTHR, fKEL, falpha_B, fDelta_x, fJ)
    phi = fV - fJ * fRext - cumTrapz(E, Delta_x)  # Discharge electrostatic potential
    return phi


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
ax.set_xlabel('$t$ [ms]', fontsize=18, weight = 'bold')
ax.set_ylabel('$I_d$ [A]', fontsize=18)
ax_V=ax.twinx()
ax_V.plot(time/1e-3, Voltage,'r')
ax_V.set_ylabel('$V$ [V]', fontsize=18)
ax.grid(True)
plt.tight_layout()
plt.savefig(ResultsFigs+"/Current.pdf", bbox_inches='tight')
    
for i_save, file in enumerate(files):

    print("Preparing plot for i = ", i_save)

    with open(file, 'rb') as f:
        [t, P, U, P_Inlet, P_Outlet, J, V, B, x_center] = pickle.load(f)

    if PLOT_VARS:
        E = compute_E(P, B, ESTAR, WallInteractionConfig['Type'], R1, R2, M, x_center, LTHR, KEL, alpha_B, Delta_x, J)
        phi = compute_phi(P, B, ESTAR, WallInteractionConfig['Type'], R1, R2, M, x_center, LTHR, KEL, alpha_B, Delta_x, J, V, Rext)
        
        f = plt.figure(figsize = (12,9))

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
        ax[0].plot(x_center*100, P[0,:]/1e19)
        ax[0].set_ylabel('$n_g$ [10$^{19}$ m$^{-3}$]', fontsize=axisfontsize)
        ax[0].set_xticklabels([])
        ax[0].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)        
        ax[0].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[0].set_ylim([0, 3.0])
        ax_b.set_ylabel('$B$ [mT]', color='r', fontsize=axisfontsize)
        ax_b.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B*1000, 'r:')

        ax_phi=ax[1].twinx()
        ax[1].plot(x_center*100, E/1000.)
        ax[1].set_ylabel('$E$ [kV/m]', fontsize=axisfontsize)
        ax[1].set_xticklabels([])
        ax[1].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[1].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_phi.plot(x_center*100, phi, color='r')
        ax_phi.set_ylabel('$V$ [V]', color='r', fontsize=axisfontsize)
        ax_phi.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)

        ax_b=ax[2].twinx()
        ax[2].plot(x_center*100, P[1,:]/1e18)
        ax[2].set_ylabel('$n_i$ [10$^{18}$ m$^{-3}$]', fontsize=axisfontsize)
        ax[2].set_xlabel('$x$ [cm]', fontsize=axisfontsize)        
        ax[2].set_xticklabels([])
        ax[2].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[2].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])
        ax[2].set_ylim([0., 2.0])

        ax[3].plot(time*1000., Current, label='Eff. $I_d$')
        ax[3].plot(time*1000., CurrentDensity*A0, label='$I_d = A_0 e (\Gamma_i - \Gamma_e)$')
        ax[3].set_ylabel('$I_d$ [A]', fontsize=axisfontsize)
        ax[3].set_xlabel('$t$ [ms]', fontsize=axisfontsize)
        ax[3].plot([time[i_save]*1000., time[i_save]*1000.],
                   [Current[i_save], CurrentDensity[i_save]*A0],
                   'ko', markersize=5)
        ax[3].yaxis.set_tick_params(size=5, width=1.5, labelsize=tickfontsize)
        ax[3].xaxis.set_tick_params(size=5, width=1.5, labelsize=tickfontsize)
        ax[3].legend(fontsize=tickfontsize)
        ax[3].set_ylim([0., 10.0])
        
        ax_b=ax[4].twinx()
        ax[4].plot(x_center*100, P[3,:])
        ax[4].set_ylabel('$T_e$ [eV]', fontsize=axisfontsize)
        ax[4].set_xticklabels([])
        ax[4].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[4].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        ax_b=ax[5].twinx()
        ax[5].plot(x_center*100, P[2,:]/1000, label="$v_i$")
        ax[5].plot(x_center*100, np.sqrt(phy_const.e*P[3,:]/(131.293*phy_const.m_u))/1000.,'g--', label="$v_{Bohm}$")
        ax[5].set_ylabel('$v_i$ [km/s]', fontsize=axisfontsize)
        ax[5].set_xticklabels([])
        ax[5].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[5].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])
        ax[5].legend(fontsize=tickfontsize)

        ax_b = ax[6].twinx()
        ax[6].plot(x_center*100, P[4,:]/1000)
        ax[6].set_ylabel('$v_e$ [km/s]', fontsize=axisfontsize)
        ax[6].set_xticklabels([])
        ax[6].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[6].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
        ax_b = ax[7].twinx()
        ax[7].plot(x_center*100, j_of_x)
        ax[7].set_ylabel('$J_d$ [A.m$^{-2}$]', fontsize=axisfontsize)
        ax[7].set_xlabel('$x$ [cm]', fontsize=axisfontsize)
        ax[7].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[7].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[7].set_ylim([0.,max(800.,j_of_x.max())])
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        title = 'time = '+ str(round(t/1e-6, 4))+'$\mu$s'
        f.suptitle(title)

        for axis in ax:
            axis.grid(True)
            
        plt.tight_layout()

        #plt.subplots_adjust(wspace = 0.4, hspace=0.2)
        
        plt.savefig(ResultsFigs+"/MacroscopicVars_New_"+str(i_save)+".png", bbox_inches='tight')
        plt.close()
    

os.system("ffmpeg -r 10 -i "+ResultsFigs+"/MacroscopicVars_New_%d.png -vcodec mpeg4 -y -vb 20M "+ResultsFigs+"Evolution.mp4")

