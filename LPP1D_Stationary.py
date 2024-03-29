import numpy as np
import scipy.constants as phy_const
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import configparser
import sys
from scipy import interpolate
from scipy.optimize import fsolve 

from numba import njit

#########################################################
# We solve for a system of equations written as
# dU/dt + dF/dx = S
# with a Finite Volume Scheme.
#
# The conservative variables are
# U = [rhog, rhoi, rhoUi],
# F = [rhog*Vg, rhoUi, rhoUi*Ui + n*e*Ti].
#
# We use the following primitive variables
# P = [ng, ni, ui,  Te, ve]              TODO: maybe add , E
#
# At the boundaries we impose
# Inlet:
#       ng = mdot/(M*A0*VG)*M
#       ui = -u_bohm
# Outlet:
#       Nothing (everything goes out)
#
# The user can change the PHYSICAL PARAMETERS
# or the NUMERICAL PARAMETERS
#
# TODO: Test with Thomas' benchmark, add circuit
#
##########################################################

##########################################################
#           CONFIGURE PHYSICAL PARAMETERS
##########################################################

configFile = sys.argv[1]
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
CURRENT  = float(physicalParameters['Current'])                      # Current
TION       = float(physicalParameters['Ion temperature'])            # Ion temperature eV
alpha_B  = float(physicalParameters['Anomalous transport alpha_B']) # Anomalous transport
mdot     = float(physicalParameters['Mass flow'])                    # Mass flow rate of propellant
Te_Cath  = float(physicalParameters['Temperature Cathode'])          # Electron temperature at the cathode
Rext     = float(physicalParameters['Ballast resistor'])             # Resistor of the ballast
V        = float(physicalParameters['Voltage'])                      # Potential difference
WALLCOLLS= bool(config.getboolean('Physical Parameters', 'Wall collisions', fallback=False))               # Wall collisions
Circuit  = bool(config.getboolean('Physical Parameters', 'Circuit', fallback=False)) # RLC Circuit

# Magnetic field configuration
MagneticFieldConfig = config['Magnetic field configuration']

if MagneticFieldConfig['Type'] == 'Default':
    print(MagneticFieldConfig['Type'] + ' Magnetic Field')
    
    Bmax     = float(MagneticFieldConfig['Max B-field'])                  # Max Mag field
    LB       = float(MagneticFieldConfig['Length B-field'])               # Length for magnetic field
    LBMax    = float(MagneticFieldConfig['Position maximum B-Field'])           # length of thruster (position of B_max)
    saveBField = bool(MagneticFieldConfig['Save B-field'])

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
if 'Initial field' in NumericsConfig:
    INITIALFIELD = NumericsConfig['Initial field']                        # Name of result directory
    WITH_INITIALFIELD = True
else:
    WITH_INITIALFIELD = False


if not os.path.exists(Results):
    os.makedirs(Results)
with open(Results+'/Configuration.cfg', 'w') as configfile:
    config.write(configfile)

##########################################################
#           Allocation of large vectors                  #
##########################################################

Delta_t  = 1.                                                   # Initialization of Delta_t (do not change)
Delta_x  = LENGTH/NBPOINTS

x_mesh   = np.linspace(0, LENGTH, NBPOINTS + 1)                 # Mesh in the interface
x_center = np.linspace(Delta_x, LENGTH - Delta_x, NBPOINTS)     # Mesh in the center of cell
# B0       = Bmax*np.exp(-((x_center - L0)/LB)**2.)               # Magnetic field
B0       = Bmax*np.exp(-((x_center - LBMax)/LB)**2.)               # Magnetic field


# Allocation of vectors
P        = np.ones((5, NBPOINTS))                               # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
U        = np.ones((3, NBPOINTS))                               # Conservative vars U = [rhog, rhoi, rhoUi]
S        = np.ones((3, NBPOINTS))                               # Source Term
F_cell   = np.ones((3, NBPOINTS + 2))                           # Flux at the cell center. We include the Flux of the Ghost cells
F_interf = np.ones((3, NBPOINTS + 1))                           # Flux at the interface
U_Inlet  = np.ones((3, 1))                                      # Ghost cell on the left
P_Inlet  = np.ones((5, 1))                                      # Ghost cell on the left
U_Outlet = np.ones((3, 1))                                      # Ghost cell on the right
P_Outlet = np.ones((5, 1))                                      # Ghost cell on the right
if TIMESCHEME == 'TVDRK3':
    P_1        = np.ones((5, NBPOINTS))                         # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
    U_1        = np.ones((3, NBPOINTS))                         # Conservative vars U = [rhog, rhoi, rhoUi]
    
if Circuit:
    R = float(physicalParameters['R'])
    L = float(physicalParameters['L'])
    C = float(physicalParameters['C'])
    V0 = V

    
    X_Volt0 = np.zeros(2)       # [DeltaV, dDeltaV/dt]
    X_Volt1 = np.zeros(2)
    X_Volt2 = np.zeros(2)
    X_Volt3 = np.zeros(2)
    
    RHS_Volt0 = np.zeros(2)
    RHS_Volt1 = np.zeros(2)
    RHS_Volt2 = np.zeros(2)
    
    A_Volt  = np.zeros([2,2])
    A_Volt[0,0] = 0.
    A_Volt[0,1] = 1.
    A_Volt[1,1] = -1/(L*C)
    A_Volt[1,0] = -1/(R*C)
    
    dJdt        = 0.
    J0          = 0.


##########################################################
#           Formulas defining our model                  #
##########################################################


@njit
def PrimToCons(P, U):
    U[0,:] = P[0,:]*M                                           # rhog
    U[1,:] = P[1,:]*M                                           # rhoi
    U[2,:] = P[2,:]*P[1,:]*M                                    # rhoiUi

@njit
def ConsToPrim(U, P, CURRENT = CURRENT):
    P[0,:] = U[0,:]/M                                           # ng
    P[1,:] = U[1,:]/M                                           # ni
    P[2,:] = U[2,:]/U[1,:]                                      # Ui = rhoUi/rhoi
    # Te Computed in compute_Te
    P[4,:] = P[2,:] - CURRENT/(A0*phy_const.e*P[1,:])                 # ve

@njit
def InviscidFlux(P, F):
    F[0,:] = P[0,:]*VG*M                                        # rho_g*v_g
    F[1,:] = P[1,:]*P[2,:]*M                                    # rho_i*v_i
    F[2,:] = M*P[1,:]*P[2,:]*P[2,:] + P[1,:]*phy_const.e*TION    # M*n_i*v_i**2 + p_i *ions at 0.1 eV for the moment
    
# def Source(P, S):

#     #############################################################
#     #       We give a name to the vars to make it more readable
#     #############################################################
#     ng = P[0,:]
#     ni = P[1,:]
#     ui = P[2,:]
#     Te = P[3,:]
#     ve = P[4,:]
    
#     energy  = 3./2.*ni*phy_const.e*Te       # Electron internal energy
#     #Gamma_E = 3./2.*ni*phy_const.e*Te*ve    # Flux of internal energy
#     wce     = phy_const.e*B0/m              # electron cyclotron frequency
    
#     #############################
#     #       Compute the rates   #
#     #############################
#     Eion    = 12.1  # Ionization energy
#     gamma_i = 1     # Excitation coefficient
#     Estar   = 50    # Crossover energy

#     Kiz = 1.8e-13*(((1.5*Te)/Eion)**0.25)*np.exp(- 4*Eion/(3*Te))  # Ion - neutral  collision rate          TODO: Replace by better
#     Kel = 2.5e-13                                                  # Electron - neutral  collision rate     TODO: Replace by good one

#     sigma = 2.*Te/Estar  # SEE yield
#     sigma[sigma > 0.986] = 0.986

#     nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate
#     #Limit the collisions to inside the thruster
#     index_L0         = np.argmax(x_center > L0)
#     nu_iw[index_L0:] = 0.
    
#     nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate


#     # TODO: Put decreasing wall collisions (Not needed for the moment)
#     #    if decreasing_nu_iw:
#     #        index_L1 = np.argmax(z > L1)
#     #        index_L0 = np.argmax(z > L0)
#     #        index_ind = index_L1 - index_L0 + 1
#     #
#     #        nu_iw[index_L0: index_L1] = nu_iw[index_L0] * np.arange(index_ind, 1, -1) / index_ind
#     #        nu_iw[index_L1:] = 0.0


#     ##################################################
#     #       Compute the electron properties          #
#     ##################################################
#     phi_W = Te*np.log(np.sqrt(M/(2*np.pi*m))*(1 - sigma))       # Wall potential
#     Ew    = 2*Te + (1 - sigma)*phi_W                            # Energy lost at the wall

#     c_s    = np.sqrt(phy_const.e*Te/M)                          # Sound velocity
#     nu_m   = ng*Kel + alpha_B*wce #+ nu_ew                       # Electron momentum - transfer collision frequency TODO
#     mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))    # Effective mobility

#     #DeltaG = Gamma_e / ni
#     #grdI = gradient(DeltaG, dz)

#     S[0,:] = (-ng[:]*ni[:]*Kiz[:] + nu_iw[:]*ni[:])*M                                                      # Gas Density
#     S[1,:] = (ng[:]*ni[:]*Kiz[:] - nu_iw[:]*ni[:])*M                                                       # Ion Density
#     S[2,:] = (ng[:]*ni[:]*Kiz[:]*VG - (phy_const.e/(mu_eff[:]*M))*ni[:]*ve[:] - nu_iw[:]*ni[:]*ui[:])*M    # Momentum

@njit
def Source(P, S):

    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0,:]
    ni = P[1,:]
    ui = P[2,:]
    Te = P[3,:]
    ve = P[4,:]
    
    wce     = phy_const.e*B0/m              # electron cyclotron frequency
    
    #############################
    #       Compute the rates   #
    #############################
    Eion    = 12.127# Ionization energy
    E_exc   = 11.6
    Estar   = 80    # Crossover energy
    h_l     = 0.5

    Kexc  = 1.2921e-13*np.exp(-E_exc/Te)
    v_the = np.sqrt(8.*phy_const.e*Te/(np.pi*phy_const.m_e))
    Kiz   = v_the*(-1.024e-24*Te**2 + 6.386e-20*np.exp(-Eion/Te))  
    Kel   = 2.5e-13                                                  # Electron - neutral  collision rate     TODO: Replace by good one

    sigma = 0.5+Te/Estar  # SEE yield
    sigma[sigma > 0.97] = 0.97

    nu_iw = h_l*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate
    #Limit the collisions to inside the thruster
    index_L0         = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.
    
    nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate

    if WALLCOLLS:

        ##################################################
        #       Compute the electron properties          #
        ##################################################
        phi_W = Te*np.log(np.sqrt(M/(2*np.pi*m))*(1 - sigma))       # Wall potential
        Ew    = 2*Te + (1 - sigma)*phi_W                            # Energy lost at the wall

        c_s    = np.sqrt(phy_const.e*Te/M)                          # Sound velocity
        nu_m   = ng*Kel + alpha_B*wce + nu_ew                       # Electron momentum - transfer collision frequency TODO
        mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))    # Effective mobility


        S[0,:] = (-ng[:]*ni[:]*Kiz[:] + nu_iw[:]*ni[:])*M                                                      # Gas Density
        S[1,:] = (ng[:]*ni[:]*Kiz[:] - nu_iw[:]*ni[:])*M                                                       # Ion Density
        S[2,:] = (ng[:]*ni[:]*Kiz[:]*VG - (phy_const.e/(mu_eff[:]*M))*ni[:]*ve[:] - nu_iw[:]*ni[:]*ui[:])*M    # Momentum
    
    else:
        ##################################################
        #       Compute the electron properties          #
        ##################################################
        nu_m   = ng*Kel + alpha_B*wce                               # Electron momentum - transfer collision frequency TODO
        mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))    # Effective mobility

        S[0,:] = (-ng[:]*ni[:]*Kiz[:])*M                                                      # Gas Density
        S[1,:] = (ng[:]*ni[:]*Kiz[:])*M                                                       # Ion Density
        S[2,:] = (ng[:]*ni[:]*Kiz[:]*VG - (phy_const.e/(mu_eff[:]*M))*ni[:]*ve[:])*M          # Momentum

# # Compute the Current
# def compute_Te(P):

#     #############################################################
#     #       We give a name to the vars to make it more readable
#     #############################################################
#     ng = P[0,:]
#     ni = P[1,:]
#     ui = P[2,:]
#     ve = P[4,:]
#     Gamma_i = ni*ui
#     wce     = phy_const.e*B0/m              # electron cyclotron frequency

    
#     #############################
#     #       Compute the rates   #
#     #############################

#     def func(Te, iCell):
#         Eion    = 12.1  # Ionization energy
#         gamma_i = 1     # Excitation coefficient TODO
#         Estar   = 50    # Crossover energy

#         Kiz = 1.8e-13*(((1.5*Te)/Eion)**0.25)*np.exp(- 4*Eion/(3*Te))  # Ion - neutral  collision rate          TODO: Replace by better
#         Kel = 2.5e-13                                                  # Electron - neutral  collision rate     TODO: Replace by good one

#         sigma = 2.*Te/Estar                                            # SEE yield
#         sigma[sigma > 0.986] = 0.986

#         nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate d
#         #Limit the collisions to inside the thruster
#         index_L0         = np.argmax(x_center > L0)
#         nu_iw[index_L0:] = 0.
        
#         nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate


#         nu_m   = ng*Kel + alpha_B*wce #+ nu_ew                          # Electron momentum - transfer collision frequency TODO
        
#         mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))       # Effective mobility
        
#         phi_W = Te*np.log(np.sqrt(M/(2*np.pi*m))*(1 - sigma))       # Wall potential
#         Ew    = 2*Te + (1 - sigma)*phi_W                            # Energy lost at the wall

        
#         return -ng[iCell]*ni[iCell]*Kiz*Eion*gamma_i - nu_ew*ni[iCell]*Ew + 1./mu_eff[iCell]*(ni[iCell]*ve[iCell])**2./ni[iCell]
        
#     for iCell,Te in enumerate(P[3,:]):
#         P[3, iCell] = fsolve(func, Te, args=(iCell))

# Compute the Temperature
# @njit
def compute_Te(P, B0 = B0, WALLCOLLS=WALLCOLLS, R1 = R1, R2 = R2, M = M, L0 = L0):

    from scipy.optimize import fsolve 

    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0,:]
    ni = P[1,:]
    ui = P[2,:]
    Te = P[3,:]
    ve = P[4,:]
    
    Gamma_i = ni*ui
    wce     = phy_const.e*B0/m              # electron cyclotron frequency

    def func(Te, B0 = B0, WALLCOLLS=WALLCOLLS, R1 = R1, R2 = R2, M = M, L0 = L0):

        #############################
        #       Compute the rates   #
        #############################
        Eion    = 12.127# Ionization energy
        E_exc   = 11.6
        Estar   = 80    # Crossover energy
        h_l     = 0.5

        Kexc  = 1.2921e-13*np.exp(-E_exc/Te)
        v_the = np.sqrt(8.*phy_const.e*Te/(np.pi*phy_const.m_e))
        Kiz   = v_the*(-1.024e-24*Te**2 + 6.386e-20*np.exp(-Eion/Te))  
        Kel   = 2.5e-13                                                  # Electron - neutral  collision rate     TODO: Replace by good one

        sigma = 0.5+Te/Estar  # SEE yield
        sigma[sigma > 0.97] = 0.97

        nu_iw = h_l*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate
        #Limit the collisions to inside the thruster
        index_L0         = np.argmax(x_center > L0)
        nu_iw[index_L0:] = 0.
        
        nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate

        if WALLCOLLS:

            nu_m   = ng*Kel + alpha_B*wce  + nu_ew                  # Electron momentum - transfer collision frequency TODO
            
            mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))       # Effective mobility
                        
            phi_W = Te*np.log(np.sqrt(M/(2*np.pi*m))*(1 - sigma))       # Wall potential
            Ew    = 2*Te + (1 - sigma)*phi_W                            # Energy lost at the wall

            
            return -ng*ni*Kiz*Eion - nu_ew*ni*Ew + 1./mu_eff*(ni*ve)**2./ni
        
        else:

            nu_m   = ng*Kel + alpha_B*wce                                 # Electron momentum - transfer collision frequency TODO
            
            mu_eff = (phy_const.e/(phy_const.m_e*nu_m))*(1./(1 + (wce/nu_m)**2))       # Effective mobility

            E_total = Eion + E_exc*Kexc/Kiz + 3.*(phy_const.m_e/M)*(Kel/Kiz)*Te

            # print("alpha_B = ", alpha_B)
            # print("Te    = ", Te)
            # print("nu_m = ", nu_m)
            # print("mu_effu = ", mu_eff)
            # print("Kiz = ", Kiz)
            # print("E_total = ", E_total)
            # print("res_1       = ", -ng*Kiz*E_total)
            # print("res_2       = ",  1./mu_eff*ve**2.)
            # print("res_1/res_2       = ",  ng*Kiz*E_total/(1./mu_eff*ve**2.))

            # exit(0)

            return -ng*Kiz*E_total + 1./mu_eff*ve**2.

        
    # for iCell, Te in enumerate(P[3,:]):
    #     P[3, iCell] = fsolve(func, Te, args=(iCell))
    # P[3, :] = fsolve(func, np.ones_like(Te)*10.)
    T_eNotSmooth = fsolve(func, np.ones_like(Te)*10.)
    tck     = interpolate.splrep(x_center, T_eNotSmooth, s=0)
    P[3, :] = interpolate.splev(x_center, tck, der=0)

@njit
def SetInlet(P_In, U_ghost, P_ghost, CURRENT = CURRENT):
    
    # U_Bohm  = np.sqrt(phy_const.e*P_In[3]/M) TODO: Done with an initial value
    U_Bohm  = 200.
    
    U_ghost[0] = mdot/(M*A0*VG)*M
    # U_ghost[1] = P_In[1]*M
    # U_ghost[2] = -2.*P_In[1]*U_Bohm*M - P_In[1]*P_In[2]*M
    U_ghost[1] = 1e17*M
    U_ghost[2] = 2.*P_In[1]*U_Bohm*M - P_In[1]*P_In[2]*M
    
    P_ghost[0] = U_ghost[0]/M                                     # ng
    P_ghost[1] = U_ghost[1]/M                                     # ni
    P_ghost[2] = U_ghost[2]/U_ghost[1]                            # Ui
    P_ghost[3] = P_In[3]                                          # Te
    P_ghost[4] = P_ghost[2] - CURRENT/(A0*phy_const.e*P_ghost[1])       # ve

@njit 
def SetOutlet(P_In, U_ghost, P_ghost):

    U_ghost[0] = P_In[0]*M
    U_ghost[1] = P_In[1]*M
    U_ghost[2] = P_In[1]*P_In[2]*M
    
    P_ghost[0] = U_ghost[0]/M                                     # ng
    P_ghost[1] = U_ghost[1]/M                                     # ni
    P_ghost[2] = U_ghost[2]/U_ghost[1]                            # Ui
    P_ghost[3] = P_In[3]                                          # Te
    P_ghost[4] = P_ghost[2] - CURRENT/(A0*phy_const.e*P_ghost[1])       # ve

    
##########################################################
#           Functions defining our numerics              #
##########################################################

# TODO: These are vector. Better allocate them 
@njit
def computeMaxEigenVal_i(P):

    # U_Sound_i   = np.sqrt(phy_const.e*TION/M)
    U_Sound_i   = np.sqrt(phy_const.e*P[3,:]/M)  # Using the Bohm speed for the numerical diffusion

    return np.maximum(np.abs(U_Sound_i - P[2, :]), np.abs(U_Sound_i + P[2, :]))

@njit
def NumericalFlux(P, U, F_cell, F_interf):

    # Compute the max eigenvalue
    lambda_max_i_R  = computeMaxEigenVal_i(P[:,1:NBPOINTS+2])
    lambda_max_i_L  = computeMaxEigenVal_i(P[:,0:NBPOINTS+1])
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)
        
    # Compute the flux at the interface
    F_interf[0,:] = 0.5*(F_cell[0,0:NBPOINTS+1] + F_cell[0,1:NBPOINTS+2]) - 0.5*VG*(U[0,1:NBPOINTS+2] - U[0,0:NBPOINTS+1])
    F_interf[1,:] = 0.5*(F_cell[1,0:NBPOINTS+1] + F_cell[1,1:NBPOINTS+2]) - 0.5*lambda_max_i_12*(U[1,1:NBPOINTS+2] - U[1,0:NBPOINTS+1])
    F_interf[2,:] = 0.5*(F_cell[2,0:NBPOINTS+1] + F_cell[2,1:NBPOINTS+2]) - 0.5*lambda_max_i_12*(U[2,1:NBPOINTS+2] - U[2,0:NBPOINTS+1])

@njit
def ComputeDelta_t(P):
    # Compute the max eigenvalue
    lambda_max_i_R  = computeMaxEigenVal_i(P[:,1:NBPOINTS+2])
    lambda_max_i_L  = computeMaxEigenVal_i(P[:,0:NBPOINTS+1])
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)
    
    Delta_t = CFL*Delta_x/(max(lambda_max_i_12))
    return Delta_t
    
    
    
##########################################################################################
#                                                                                        #
#                               SAVE RESULTS                                             #
#                                                                                        #
##########################################################################################

i_save = 0

def SaveResults(P, U, P_Inlet, P_Outlet, CURRENT, V, x_center, time, i_save):
    if not os.path.exists(Results):
        os.makedirs(Results)
    ResultsFigs = Results+"/Figs"
    if not os.path.exists(ResultsFigs):
        os.makedirs(ResultsFigs)
    ResultsData = Results+"/Data"
    if not os.path.exists(ResultsData):
        os.makedirs(ResultsData)
        
    # Save the data
    filenameTemp = ResultsData+"/MacroscopicVars_"+str(i_save)+".pkl"
    pickle.dump([time, P, U, P_Inlet, P_Outlet, CURRENT, V, B0, x_center], open(filenameTemp, 'wb')) # TODO: Save the current and the electric field


 
##########################################################################################################
#           Initial field                                                                                #
#           P := Primitive vars [0: ng, 1: ni, 2: ui, 3: Te, 4: ve]                                      #
#           U := Conservative vars [0: rhog, 1: rhoi, 2: rhoiui]                                         #
#                                                                                                        #
##########################################################################################################

NI0 = 1e17
TE0 = 5.


time = 0.
iter = 0

if WITH_INITIALFIELD:
    with open(INITIALFIELD, 'rb') as f:
        [t_INIT, P_INIT, U_INIT, P_Inlet_INIT, P_Outlet_INIT, J_INIT, V_INIT, B_INIT, x_center_INIT] = pickle.load(f)

    # Interpolation to new mesh
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate

    NBPOINTS_initialField = np.size(P_INIT[0,:])
    Delta_x_initialField  = LENGTH/NBPOINTS_initialField
    x_mesh_initialField   = np.linspace(0, LENGTH, NBPOINTS_initialField + 1)                 # Mesh in the interface
    x_center_initialField = np.linspace(Delta_x_initialField, LENGTH - Delta_x_initialField, NBPOINTS_initialField)     # Mesh in the center of cell

    P0_INTERP = interpolate.interp1d(x_center_initialField, P_INIT[0,:], fill_value=(P_INIT[0,0], P_INIT[0,-1]), bounds_error=False)
    P1_INTERP = interpolate.interp1d(x_center_initialField, P_INIT[1,:], fill_value=(P_INIT[1,0], P_INIT[1,-1]), bounds_error=False)
    P2_INTERP = interpolate.interp1d(x_center_initialField, P_INIT[2,:], fill_value=(P_INIT[2,0], P_INIT[2,-1]), bounds_error=False)
    P3_INTERP = interpolate.interp1d(x_center_initialField, P_INIT[3,:], fill_value=(P_INIT[3,0], P_INIT[3,-1]), bounds_error=False)
    P4_INTERP = interpolate.interp1d(x_center_initialField, P_INIT[4,:], fill_value=(P_INIT[4,0], P_INIT[4,-1]), bounds_error=False)


    # We initialize the primitive variables
    P[0,:] = P0_INTERP(x_center)                           # Initial propellant density ng TODO
    P[1,:] = P1_INTERP(x_center)                           # Initial ni
    P[2,:] = P2_INTERP(x_center)                           # Initial vi
    P[3,:] = P3_INTERP(x_center)                           # Initial Te
    P[4,:] = P4_INTERP(x_center)                           # Initial Ve

else:
    # We initialize the primitive variables
    P[0,:] *= mdot / (M * A0 * VG)                  # Initial propellant density ng TODO
    P[1,:] *= NI0                                   # Initial ni
    P[2,:] *= 10.                                   # Initial vi
    P[3,:] *= TE0                                   # Initial Te
    P[4,:] *= P[2,:] - CURRENT/(A0*phy_const.e*P[1,:])    # Initial Ve


# We initialize the conservative variables
PrimToCons(P, U)


##########################################################################################
#           Loop with Forward Euler                                                      #
#           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
#                                                                                        #
##########################################################################################

if TIMESCHEME == 'Forward Euler':

    while time < TIMEFINAL:
        # Save results
        if (iter %SAVERATE) ==0:
            SaveResults(P, U, P_Inlet, P_Outlet, CURRENT, V, x_center, time, i_save)
            i_save += 1
            print("Iter = ", iter,"\tTime = ", time/1e-6," \mus \tJ = ",CURRENT," A")

        # Compute the electron temperature from the previous iteration
        compute_Te(P)
            
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        
        # # Compute the convective Delta t
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # # Compute the source in the center of the cell
        Source(P, S)

        # # Update the solution
        U[:,:] = U[:,:] - Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:]

        # # Compute the primitive vars for next step
        ConsToPrim(U, P)

        time += Delta_t
        iter += 1

if TIMESCHEME == 'TVDRK3':

    while time < TIMEFINAL:
        # Save results
        if (iter %SAVERATE) ==0:
            SaveResults(P, U, P_Inlet, P_Outlet, CURRENT, V, x_center, time, i_save)
            i_save += 1
            print("Iter = ", iter,"\tTime = ", time/1e-6," \mus \tJ = ",CURRENT," A\tV = ", V," V")
        
        #################################################
        #           FIRST STEP RK3
        #################################################
        
        # Copy the solution to store it
        U_1[:,:] = U[:,:]
        ConsToPrim(U_1, P_1)

        # Compute the electron temperature from the previous iteration
        compute_Te(P)

        
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        
        # Compute the convective Delta t (Only in the first step)
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:,:] = U[:,:] - Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:]
        
        # Compute the primitive vars for next step
        ConsToPrim(U, P)
        
        # # Compute RLC Circuit
        # if Circuit :
        #     dJdt          = (J - J0)/Delta_t
            
        #     RHS_Volt0[0]  = X_Volt0[1]
        #     RHS_Volt0[1]  = -1/(R*C)*X_Volt0[1] - 1./(L*C)*X_Volt0[0] + 1/C*dJdt
        #     X_Volt1 = X_Volt0 + Delta_t*RHS_Volt0
        
        #################################################
        #           SECOND STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the electron temperature from the previous iteration
        compute_Te(P)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:,:] = 0.75*U_1[:,:] + 0.25*U[:,:] + 0.25*(- Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:])
        
        # Compute the primitive vars for next step
        ConsToPrim(U, P)
        
        # # Compute RLC Circuit
        # if Circuit :
        #     dJdt          = (J - J0)/Delta_t
        #     RHS_Volt1[0]  = X_Volt1[1]
        #     RHS_Volt1[1]  = -1/(R*C)*X_Volt1[1] - 1./(L*C)*X_Volt1[0] + 1/C*dJdt
        #     X_Volt2       = 0.75*X_Volt0 + 0.25*X_Volt1 + 0.25*Delta_t*RHS_Volt1
        
        #################################################
        #           THIRD STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the electron temperature from the previous iteration
        compute_Te(P)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        
        # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:,:] = 1./3.*U_1[:,:] + 2./3.*U[:,:] + 2./3.*(-Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:])
        
        # Compute the primitive vars for next step
        ConsToPrim(U, P)
        
        # # Compute RLC Circuit
        # if Circuit:
        #     dJdt          = (J - J0)/Delta_t
        #     RHS_Volt2[0]  = X_Volt2[1]
        #     RHS_Volt2[1]  = -1/(R*C)*X_Volt2[1] - 1./(L*C)*X_Volt2[0] + 1/C*dJdt
        #     X_Volt3       = 1./3.*X_Volt0 + 2./3.*X_Volt2 + 2./3.*Delta_t*RHS_Volt2
            
        #     # Reinitialize for the Circuit
        #     J0         = J
        #     X_Volt0[:] = X_Volt3[:]
            
        #     # Change the Voltage
        #     V = V0 - X_Volt0[0]

        time += Delta_t
        iter += 1
