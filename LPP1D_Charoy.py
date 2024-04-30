import numpy as np
import scipy.constants as phy_const
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import configparser
import sys

#########################################################
# We solve for a system of equations written as
# dU/dt + dF/dx = S
# with a Finite Volume Scheme.
#
# The conservative variables are
# U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te],
# F = [rhog*Vg, rhoUi, rhoUi*Ui + ne*e*Te, 5/2 ne*e*Te*Ue].
#
# We use the following primitive variables
# P = [ng, ni, ui,  Te, ve]              TODO: maybe add , E
#
# At the boundaries we impose
# Inlet:
#       ng = mdot/(M*A0*VG)*M
#       ui = -u_bohm
# Outlet:
#       Te = Te_Cath
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
alpha_B  = float(physicalParameters['Anomalous transport alpha_B']) # Anomalous transport
mdot     = float(physicalParameters['Mass flow'])                    # Mass flow rate of propellant
Te_Cath  = float(physicalParameters['Temperature Cathode'])          # Electron temperature at the cathode
Rext     = float(physicalParameters['Ballast resistor'])             # Resistor of the ballast
V        = float(physicalParameters['Voltage'])                      # Potential difference
Circuit  = bool(config.getboolean('Physical Parameters', 'Circuit', fallback=False)) # RLC Circuit

# Magnetic field configuration
MagneticFieldConfig = config['Magnetic field configuration']

if MagneticFieldConfig['Type'] == 'Default':
    print(MagneticFieldConfig['Type'] + ' Magnetic Field')
    
    Bmax     = float(MagneticFieldConfig['Max B-field'])                  # Max Mag field
    LB       = float(MagneticFieldConfig['Length B-field'])               # Length for magnetic field
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
B0       = Bmax*np.exp(-((x_center - L0)/LB)**2.)               # Magnetic field

# Allocation of vectors
P        = np.ones((5, NBPOINTS))                               # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
U        = np.ones((4, NBPOINTS))                               # Conservative vars U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te]
S        = np.ones((4, NBPOINTS))                               # Source Term
F_cell   = np.ones((4, NBPOINTS + 2))                           # Flux at the cell center. We include the Flux of the Ghost cells
F_interf = np.ones((4, NBPOINTS + 1))                           # Flux at the interface
U_Inlet  = np.ones((4, 1))                                      # Ghost cell on the left
P_Inlet  = np.ones((5, 1))                                      # Ghost cell on the left
U_Outlet = np.ones((4, 1))                                      # Ghost cell on the right
P_Outlet = np.ones((5, 1))                                      # Ghost cell on the right
if TIMESCHEME == 'TVDRK3':
    P_1        = np.ones((5, NBPOINTS))                         # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
    U_1        = np.ones((4, NBPOINTS))                         # Conservative vars U = [rhog, rhoi, rhoUi,
    
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

def PrimToCons(P, U):
    U[0,:] = P[0,:]*M                                           # rhog
    U[1,:] = P[1,:]*M                                           # rhoi
    U[2,:] = P[2,:]*P[1,:]*M                                    # rhoiUi
    U[3,:] = 3./2.*P[1,:]*phy_const.e*P[3,:]                    # 3/2*ni*e*Te
    
def ConsToPrim(U, P):
    P[0,:] = U[0,:]/M                                           # ng
    P[1,:] = U[1,:]/M                                           # ni
    P[2,:] = U[2,:]/U[1,:]                                      # Ui = rhoUi/rhoi
    P[3,:] = 2./3.*U[3,:]/(phy_const.e*P[1,:])                  # Te
    P[4,:] = P[2,:] - J/(A0*phy_const.e*P[1,:])                 # ve

def InviscidFlux(P, F):
    F[0,:] = P[0,:]*VG*M                                        # rho_g*v_g
    F[1,:] = P[1,:]*P[2,:]*M                                    # rho_i*v_i
    F[2,:] = M*P[1,:]*P[2,:]*P[2,:] + P[1,:]*phy_const.e*P[3,:] # M*n_i*v_i**2 + p_e
    F[3,:] = 5./2.*P[1,:]*phy_const.e*P[3,:]*P[4,:]             # 5/2n_i*e*T_e*v_e
    
def Source(P, S):
    def gradient(y, d):
        dp_dz = np.empty_like(y)
        dp_dz[1:-1] = (y[2:] - y[:-2]) / (2 * d)
        dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
        dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

        return dp_dz

    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0,:]
    ni = P[1,:]
    ui = P[2,:]
    Te = P[3,:]
    ve = P[4,:]
    
    energy  = 3./2.*ni*phy_const.e*Te       # Electron internal energy
    #Gamma_E = 3./2.*ni*phy_const.e*Te*ve    # Flux of internal energy
    wce     = phy_const.e*B0/m              # electron cyclotron frequency
    
    #############################
    #       Compute the rates   #
    #############################
    Eion    = 12.1  # Ionization energy
    gamma_i = 3     # Excitation coefficient
    Estar   = 50    # Crossover energy

    Kiz = 1.8e-13*(((1.5*Te)/Eion)**0.25)*np.exp(- 4*Eion/(3*Te))  # Ion - neutral  collision rate          MARTIN: Change 
    Kel = 2.5e-13                                                  # Electron - neutral  collision rate     MARTIN: put to zero 

    sigma = 2.*Te/Estar  # SEE yield
    sigma[sigma > 0.986] = 0.986

    nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate MARTIN:nu_iw is zero
    #Limit the collisions to inside the thruster
    index_L0         = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.
    
    nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate


    # TODO: Put decreasing wall collisions (Not needed for the moment)
    #    if decreasing_nu_iw:
    #        index_L1 = np.argmax(z > L1)
    #        index_L0 = np.argmax(z > L0)
    #        index_ind = index_L1 - index_L0 + 1
    #
    #        nu_iw[index_L0: index_L1] = nu_iw[index_L0] * np.arange(index_ind, 1, -1) / index_ind
    #        nu_iw[index_L1:] = 0.0


    ##################################################
    #       Compute the electron properties          #
    ##################################################
    phi_W = Te*np.log(np.sqrt(M/(2*np.pi*m))*(1 - sigma))       # Wall potential
    Ew    = 2*Te + (1 - sigma)*phi_W                            # Energy lost at the wall

    c_s    = np.sqrt(phy_const.e*Te/M)                          # Sound velocity
    nu_m   = ng*Kel + alpha_B*wce + nu_ew                       # Electron momentum - transfer collision frequency
    mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))    # Effective mobility

    #div_u   = gradient(ve, d=Delta_x)               # To be used with 3./2. in line 160 and + phy_const.e*ni*Te*div_u  in line 231
    div_p   = gradient(phy_const.e*ni*Te, d=Delta_x) # To be used with 5./2 and + div_p*ve in line 231

    S[0,:] = (-ng[:]*ni[:]*Kiz[:] + nu_iw[:]*ni[:])*M                                                      # Gas Density MARTIN: -ng[:]*ni[:]*Kiz[:] will be a constant
    S[1,:] = (ng[:]*ni[:]*Kiz[:] - nu_iw[:]*ni[:])*M                                                       # Ion Density
    S[2,:] = (ng[:]*ni[:]*Kiz[:]*VG - (phy_const.e/(mu_eff[:]*M))*ni[:]*ve[:] - nu_iw[:]*ni[:]*ui[:])*M    # Momentum
    S[3,:] = -ng[:]*ni[:]*Kiz[:]*Eion*gamma_i*phy_const.e - nu_ew[:]*ni[:]*Ew*phy_const.e + 1./mu_eff[:]*(ni[:]*ve[:])**2./ni[:]*phy_const.e + div_p*ve #+ phy_const.e*ni*Te*div_u  #- gradI_term*ni*Te*grdI          # Energy
    
# Compute the Current
def compute_I(P): # MARTIN Check that this is consistent with the source term
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
    wce     = phy_const.e*B0/m              # electron cyclotron frequency

    
    #############################
    #       Compute the rates   #
    #############################
    Eion    = 12.1  # Ionization energy
    gamma_i = 3     # Excitation coefficient
    Estar   = 50    # Crossover energy

    Kiz = 1.8e-13*(((1.5*Te)/Eion)**0.25)*np.exp(- 4*Eion/(3*Te))  # Ion - neutral  collision rate          TODO: Replace by better
    Kel = 2.5e-13                                                  # Electron - neutral  collision rate     TODO: Replace by good one

    sigma = 2.*Te/Estar                                            # SEE yield
    sigma[sigma > 0.986] = 0.986

    nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)       # Ion - wall collision rate d
    #Limit the collisions to inside the thruster
    index_L0         = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.
    
    nu_ew = nu_iw/(1 - sigma)                                      # Electron - wall collision rate


    nu_m   = ng*Kel + alpha_B*wce + nu_ew                          # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e/(m*nu_m))*(1./(1 + (wce/nu_m)**2))       # Effective mobility
    dp_dz  = gradient(ni*Te, d = Delta_x)
    
    # top = V + np.sum( (Gamma_i / (mu_eff * ni)) + dp_dz / ni)*dz
    top = V + trapz((Gamma_i / (mu_eff * ni)) + dp_dz / ni, d=Delta_x)
    # bottom = q * A0 * Rext + np.sum( 1. / (mu_eff * ni))*dz
    bottom = phy_const.e * A0 * Rext + trapz(1. / (mu_eff * ni), d=Delta_x)
    I0 = top / bottom  # Discharge current density
    
    #print("I0 = ", I0*phy_const.e*A0)
    return I0*phy_const.e*A0

    
def SetInlet(P_In, U_ghost, P_ghost):
    
    U_Bohm  = np.sqrt(phy_const.e*P_In[3]/M)
    
    U_ghost[0] = mdot/(M*A0*VG)*M
    U_ghost[1] = P_In[1]*M
    U_ghost[2] = -2.*P_In[1]*U_Bohm*M - P_In[1]*P_In[2]*M
    U_ghost[3] = 3./2.*P_In[1]*phy_const.e*P_In[3]
    
    P_ghost[0] = U_ghost[0]/M                                     # ng
    P_ghost[1] = U_ghost[1]/M                                     # ni
    P_ghost[2] = U_ghost[2]/U_ghost[1]                            # Ui
    P_ghost[3] = 2./3.*U_ghost[3]/(phy_const.e*P_ghost[1])        # Te
    P_ghost[4] = P_ghost[2] - J/(A0*phy_const.e*P_ghost[1])       # ve
    
def SetOutlet(P_In, U_ghost, P_ghost):

    U_ghost[0] = P_In[0]*M
    U_ghost[1] = P_In[1]*M
    U_ghost[2] = P_In[1]*P_In[2]*M
    U_ghost[3] = 3./2.*P_In[1]*phy_const.e*Te_Cath
    
    P_ghost[0] = U_ghost[0]/M                                     # ng
    P_ghost[1] = U_ghost[1]/M                                     # ni
    P_ghost[2] = U_ghost[2]/U_ghost[1]                            # Ui
    P_ghost[3] = 2./3.*U_ghost[3]/(phy_const.e*P_ghost[1])        # Te
    P_ghost[4] = P_ghost[2] - J/(A0*phy_const.e*P_ghost[1])       # ve

    
##########################################################
#           Functions defining our numerics              #
##########################################################

# TODO: These are vector. Better allocate them
def computeMaxEigenVal_e(P):

    U_Bohm   = np.sqrt(phy_const.e*P[3,:]/M)
    
    return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[4,:])*2, abs(U_Bohm + P[4,:])*2)]
    
def computeMaxEigenVal_i(P):

    U_Bohm   = np.sqrt(phy_const.e*P[3,:]/M)

    return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]

def NumericalFlux(P, U, F_cell, F_interf):

    # Compute the max eigenvalue
    lambda_max_i_R  = computeMaxEigenVal_i(P[:,1:NBPOINTS+2])
    lambda_max_i_L  = computeMaxEigenVal_i(P[:,0:NBPOINTS+1])
    lambda_max_i_12 = np.array([max(l1, l2) for l1, l2 in zip(lambda_max_i_L, lambda_max_i_R)])
    
    lambda_max_e_R  = computeMaxEigenVal_e(P[:,1:NBPOINTS+2])
    lambda_max_e_L  = computeMaxEigenVal_e(P[:,0:NBPOINTS+1])
    lambda_max_e_12 = np.array([max(l1, l2) for l1, l2 in zip(lambda_max_e_L, lambda_max_e_R)])
    
    # Compute the flux at the interface
    F_interf[0,:] = 0.5*(F_cell[0,0:NBPOINTS+1] + F_cell[0,1:NBPOINTS+2]) - 0.5*VG*(U[0,1:NBPOINTS+2] - U[0,0:NBPOINTS+1])
    F_interf[1,:] = 0.5*(F_cell[1,0:NBPOINTS+1] + F_cell[1,1:NBPOINTS+2]) - 0.5*lambda_max_i_12*(U[1,1:NBPOINTS+2] - U[1,0:NBPOINTS+1])
    F_interf[2,:] = 0.5*(F_cell[2,0:NBPOINTS+1] + F_cell[2,1:NBPOINTS+2]) - 0.5*lambda_max_i_12*(U[2,1:NBPOINTS+2] - U[2,0:NBPOINTS+1])
    F_interf[3,:] = 0.5*(F_cell[3,0:NBPOINTS+1] + F_cell[3,1:NBPOINTS+2]) - 0.5*lambda_max_e_12*(U[3,1:NBPOINTS+2] - U[3,0:NBPOINTS+1])

def ComputeDelta_t(P):
    # Compute the max eigenvalue
    lambda_max_i_R  = computeMaxEigenVal_i(P[:,1:NBPOINTS+2])
    lambda_max_i_L  = computeMaxEigenVal_i(P[:,0:NBPOINTS+1])
    lambda_max_i_12 = np.array([max(l1, l2) for l1, l2 in zip(lambda_max_i_L, lambda_max_i_R)])
    
    lambda_max_e_R  = computeMaxEigenVal_e(P[:,1:NBPOINTS+2])
    lambda_max_e_L  = computeMaxEigenVal_e(P[:,0:NBPOINTS+1])
    lambda_max_e_12 = np.array([max(l1, l2) for l1, l2 in zip(lambda_max_e_L, lambda_max_e_R)])

    Delta_t = CFL*Delta_x/(max(max(lambda_max_e_12), max(lambda_max_i_12)))
    return Delta_t
    
    
    
##########################################################################################
#                                                                                        #
#                               SAVE RESULTS                                             #
#                                                                                        #
##########################################################################################

i_save = 0

def SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save):
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
    pickle.dump([time, P, U, P_Inlet, P_Outlet, J, V, B0, x_center], open(filenameTemp, 'wb')) # TODO: Save the current and the electric field


 
##########################################################################################################
#           Initial field                                                                                #
#           P := Primitive vars [0: ng, 1: ni, 2: ui, 3: Te, 4: ve]                                      #
#           U := Conservative vars [0: rhog, 1: rhoi, 2: rhoiui, 3: 3./2.ni*e*Te]                        #
#                                                                                                        #
##########################################################################################################

NG0 = 1e19 # MARTIN : Use same initial field than Thomas
NI0 = 1e17
TE0 = 5.


time = 0.
iter = 0
J    = 0.                                       # Initial Current

# We initialize the primitive variables
P[0,:] *= mdot / (M * A0 * VG)                  # Initial propellant density ng
P[1,:] *= NI0                                   # Initial ni
P[2,:] *= 0.                                    # Initial vi
P[3,:] *= TE0                                   # Initial Te
P[4,:] *= P[2,:] - J/(A0*phy_const.e*P[1,:])    # Initial Ve

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
            SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save)
            i_save += 1
            print("Iter = ", iter,"\tTime = ", time/1e-6," \mus \tJ = ",J," A")
            
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        
        # Compute the convective Delta t
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:,:] = U[:,:] - Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:]

        # Compute the current
        J = compute_I(P)

        # Compute the primitive vars for next step
        ConsToPrim(U, P)

        time += Delta_t
        iter += 1

if TIMESCHEME == 'TVDRK3':

    while time < TIMEFINAL:
        # Save results
        if (iter %SAVERATE) ==0:
            SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save)
            i_save += 1
            print("Iter = ", iter,"\tTime = ", time/1e-6," \mus \tJ = ",J," A\tV = ", V," V")
        
        #################################################
        #           FIRST STEP RK3
        #################################################
        
        # Copy the solution to store it
        U_1[:,:] = U[:,:]
        ConsToPrim(U_1, P_1)
        
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
        
        # Compute the current
        J = compute_I(P)
        
        # Compute the primitive vars for next step
        ConsToPrim(U, P)
        
        # Compute RLC Circuit
        if Circuit :
            dJdt          = (J - J0)/Delta_t
            
            RHS_Volt0[0]  = X_Volt0[1]
            RHS_Volt0[1]  = -1/(R*C)*X_Volt0[1] - 1./(L*C)*X_Volt0[0] + 1/C*dJdt
            X_Volt1 = X_Volt0 + Delta_t*RHS_Volt0
        
        #################################################
        #           SECOND STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:,:] = 0.75*U_1[:,:] + 0.25*U[:,:] + 0.25*(- Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:])
        
        # Compute the current
        J = compute_I(P)
        
        # Compute the primitive vars for next step
        ConsToPrim(U, P)
        
        # Compute RLC Circuit
        if Circuit :
            dJdt          = (J - J0)/Delta_t
            RHS_Volt1[0]  = X_Volt1[1]
            RHS_Volt1[1]  = -1/(R*C)*X_Volt1[1] - 1./(L*C)*X_Volt1[0] + 1/C*dJdt
            X_Volt2       = 0.75*X_Volt0 + 0.25*X_Volt1 + 0.25*Delta_t*RHS_Volt1
        
        #################################################
        #           THIRD STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        
        # Compute the Numerical at the interfaces
        NumericalFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), np.concatenate([U_Inlet, U, U_Outlet], axis=1), F_cell, F_interf)
        
        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:,:] = 1./3.*U_1[:,:] + 2./3.*U[:,:] + 2./3.*(-Delta_t/Delta_x*(F_interf[:,1:NBPOINTS+1] - F_interf[:,0:NBPOINTS]) + Delta_t*S[:,:])
        
        # Compute the current
        J = compute_I(P)
        
        # Compute the primitive vars for next step
        ConsToPrim(U, P)
        
        # Compute RLC Circuit
        if Circuit:
            dJdt          = (J - J0)/Delta_t
            RHS_Volt2[0]  = X_Volt2[1]
            RHS_Volt2[1]  = -1/(R*C)*X_Volt2[1] - 1./(L*C)*X_Volt2[0] + 1/C*dJdt
            X_Volt3       = 1./3.*X_Volt0 + 2./3.*X_Volt2 + 2./3.*Delta_t*RHS_Volt2
            
            # Reinitialize for the Circuit
            J0         = J
            X_Volt0[:] = X_Volt3[:]
            
            # Change the Voltage
            V = V0 - X_Volt0[0]

        time += Delta_t
        iter += 1
