import numpy as np
import math
import scipy.constants as phy_const
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import configparser
import sys
from numba import njit
import time as ttime
import glob
import scipy.interpolate as interpolate

from modules.simu_params import SimuParameters

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
#       ng = MDOT/(M*A0*VG)*M
#       ui = -u_bohm
# Outlet:
#       Te = Te_Cath
#
# The user can change the PHYSICAL PARAMETERS
# or the NUMERICAL PARAMETERS
#
# TODO: Test with Thomas' benchmark
#
# This script FLHET_compiled.py has a few functions compiled with numba.
# It is approximately 2.3 times faster than its not compiled counterpart:
# the LPP1D.py script.
##########################################################


tttime_start = ttime.time()

configfile = sys.argv[1]

msp = SimuParameters(configfile)

##### Renames th variables for more clarity #####
Resultsdir = msp.Results
MDOT    = msp.MDOT
Mi      = msp.Mi
A0      = msp.A0
VG      = msp.VG
NI0     = msp.NI0
TE0     = msp.TE0
ESTAR   = msp.ESTAR
wall_inter_type     = msp.wall_inter_type
R1      = msp.R1
R2      = msp.R2
LX  = msp.LX
LTHR    = msp.LTHR
KEL     = msp.KEL
TIMESCHEME  = msp.TIMESCHEME
TIMEFINAL   = msp.TIMEFINAL
SAVERATE    = msp.SAVERATE
boolIonColl = msp.boolIonColl
boolSizImposed  = msp.boolSizImposed
Eion        = msp.Eion
gamma_i     = msp.gamma_i
Te_inj        = msp.Te_inj
Rext        = msp.Rext
Te_Cath     = msp.Te_Cath
boolPressureDiv     = msp.boolPressureDiv # is dp/dx * u_e  accounted in the energy equation
CFL         = msp.CFL
HEATFLUX    = msp.HEATFLUX
IMPlICIT    = msp.IMPlICIT
boolCircuit = msp.Circuit
V           = msp.V0

MakeTheResultsDir(Resultsdir)

msp.save_config_file("Configuration.cfg")

# delete current data in location:
list_of_res_files = glob.glob(Resultsdir+"/Data/Macroscopic*.pkl")
for filenametemp in list_of_res_files:
    os.remove(filenametemp)
if len(list_of_res_files) > 0:
    print("Warning: all Macroscopic*.pkl files in the "+Resultsdir+" location were deleted to welcome new data.")

Delta_t = 1.0  # Initialization of Delta_t (do not change)

##########################################################
#           Allocation of large vectors                  #
##########################################################

x_mesh, x_center, Delta_x, x_center_extended, Delta_x_extended = msp.return_tiled_domain()
NBPOINTS = np.shape(x_center)[0]


def compute_B_array():

    BMAX = msp.BMAX
    B0     = msp.B0
    LTHR   = msp.LTHR
    LB1    = msp.LB1
    LB2     = msp.LB2

    if msp.BTYPE == 'CharoyBenchmark':

        a1 = (BMAX - B0)/(1 - math.exp(-LTHR**2/(2*LB1**2)))
        a2 = (BMAX - fBLX)/(1 - math.exp(-(fLX - LTHR)**2/(2*LB2**2)))
        b1 = BMAX - a1
        b2 = BMAX - a2
        Barr1 = a1*np.exp(-(fx_center - LTHR)**2/(2*LB1**2)) + b1
        Barr2 = a2*np.exp(-(fx_center - LTHR)**2/(2*LB2**2)) + b2    # Magnetic field outside the thruster

        Barr = np.where(fx_center <= LTHR, Barr1, Barr2)
    
    else:
        Barr    = BMAX * np.exp(-(((fx_center - LTHR) / LB1) ** 2.0))  # Magnetic field within the thruster
        Barr    = np.where(fx_center < LTHR, Barr, BMAX * np.exp(-(((fx_center - LTHR) / LB2) ** 2.0)))  # Magnetic field outside the thruster

    return Barr

Barr = compute_B_array()

alpha_B1, alpha_B2  = msp.extract_anom_coeffs()

def compute_alphaB_array():

    alpha_B = (np.ones(NBPOINTS) * alpha_B1)  # Anomalous transport coefficient inside the thruster
    alpha_B = np.where(x_center < msp.LTHR, alpha_B, alpha_B2)  # Anomalous transport coefficient in the plume
    alpha_B_smooth = np.copy(alpha_B)

    # smooth between alpha_B1 and alpha_B2
    nsmooth_o2 = msp.NBPOINTS_INIT//20
    for index in range(nsmooth_o2, NBPOINTS - (nsmooth_o2+1)):
        alpha_B_smooth[index] = np.mean(alpha_B[ index-nsmooth_o2 : index+(nsmooth_o2+1) ])
    
    return alpha_B_smooth    

alpha_B = compute_alphaB_array(x_center, alpha_B1, alpha_B2, LTHR, msp.NBPOINTS_INIT)

##### Save Unvariant data #####
ResultsData = fResults + "/Data"
pickle.dump(
    [fBarr, fx_mesh, fx_center, falpha_B], open(ResultsData + "/MacroscopicUnvariants.pkl", "wb")
)

##### Allocation of vectors #####
P = np.ones((5, NBPOINTS))  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
U = np.ones((4, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te]
S = np.ones((4, NBPOINTS))  # Source Term
Efield  = np.zeros(NBPOINTS)
F_cell = np.ones((4, NBPOINTS + 2))  # Flux at the cell center. We include the Flux of the Ghost cells
F_interf = np.ones((4, NBPOINTS + 1))  # Flux at the interface
U_LeftGhost = np.ones((4, 1))  # Ghost cell on the left
P_LeftGhost = np.ones((5, 1))  # Ghost cell on the left
U_RightGhost = np.ones((4, 1))  # Ghost cell on the right
P_RightGhost = np.ones((5, 1))  # Ghost cell on the right

if START_FROM_INPUT:

    list_of_pkls = glob.glob(INPUT_DIR+"/MacroscopicVars*.pkl")
    assert(len(list_of_pkls) == 1)
    INPUT_FILE  = list_of_pkls[0]
    print("Simulation starts from the profiles stored in ", INPUT_FILE)

    with open(INPUT_FILE, 'rb') as f: # !!! Caution !!!: outdated, because currently data are not saved like this. May malfunction for input file recently added.
        [t_INIT, P_INIT, U_INIT, P_LeftGhost_INIT, P_RightGhost_INIT, J_INIT, V_INIT, B_INIT, x_center_INIT] = pickle.load(f)

    NBPOINTS_initialField = P_INIT.shape[1]
    #Delta_x_initialField  = LX/NBPOINTS_initialField
    x_mesh_initialField   = np.zeros(NBPOINTS_initialField+1, dtype=float) # Mesh in the interface
    x_mesh_initialField[1:-1]   = 0.5*(x_center_INIT[:-1] + x_center_INIT[1:])
    x_mesh_initialField[-1]     = LX
    #x_center_initialField = np.linspace(Delta_x_initialField, LX - Delta_x_initialField, NBPOINTS_initialField)     # Mesh in the center of cell

        # intégration de la fonction INITIAL field en cours.
    P0_INTERP = interpolate.interp1d(x_center_INIT, P_INIT[0,:], fill_value=(P_INIT[0,0], P_INIT[0,-1]), bounds_error=False)
    P1_INTERP = interpolate.interp1d(x_center_INIT, P_INIT[1,:], fill_value=(P_INIT[1,0], P_INIT[1,-1]), bounds_error=False)
    P2_INTERP = interpolate.interp1d(x_center_INIT, P_INIT[2,:], fill_value=(P_INIT[2,0], P_INIT[2,-1]), bounds_error=False)
    P3_INTERP = interpolate.interp1d(x_center_INIT, P_INIT[3,:], fill_value=(P_INIT[3,0], P_INIT[3,-1]), bounds_error=False)
    P4_INTERP = interpolate.interp1d(x_center_INIT, P_INIT[4,:], fill_value=(P_INIT[4,0], P_INIT[4,-1]), bounds_error=False)

    # We initialize the primitive variables
    P[0,:] = P0_INTERP(x_center)                           # Initial propellant density ng TODO
    P[1,:] = P1_INTERP(x_center)                           # Initial ni
    P[2,:] = P2_INTERP(x_center)                           # Initial vi
    P[3,:] = P3_INTERP(x_center)                           # Initial Te
    P[4,:] = P4_INTERP(x_center)                           # Initial Ve

    Jm1 = J_INIT
    J   = J_INIT
    
    del t_INIT, P_INIT, U_INIT, P_LeftGhost_INIT, P_RightGhost_INIT, J_INIT, V_INIT, B_INIT , x_center_INIT, P0_INTERP, P1_INTERP, P2_INTERP, P3_INTERP, P4_INTERP

else:

    # We initialize the primitive variables
    ng_anode = MDOT / (Mi* A0 * VG)  # Initial propellant density ng at the anode location
    #P[0,:] = InitNeutralDensity(x_center, ng_anode, VG, P, ionization_type, SIZMAX, LSIZ1, LSIZ2) # initialize n_g in the space so that it is cst in time if there is no wall recombination.
    ### Warning, in the code currently, neutrals dyanmic is canceled.
    P[0,:] = ng_anode
    P[1, :] = msp.NI0  # Initial ni
    P[2, :] = 0.0  # Initial vi
    P[3, :] = msp.TE0  # Initial Te

    def SmoothInitialTemperature(bulk_array:np.ndarray, Toutlet:float)->np.ndarray:
        """Return a smoothed version of the array bulkarray. It contains the bulk e-
        initial temperature. It smoothes the possible jump between this bulk
        value and the cathode e- temperature. Otherwise it would introduce a
        harmful discontinuity. 

        Args:
            bulk_array (np.ndarray): array containg the initial bulk T_e.
            Toutlet (float): value of T_e at the outlet. Usually the cathode T_e.
        """
        bulk_copy = np.copy(bulk_array)
        nsmooth = bulk_array.shape[0]//10
        for i in range(nsmooth):
            a = (i+1)/(nsmooth+1)
            bulk_copy[-1-i] = bulk_array[-1-i]*a + Toutlet*(1 - a)
        
        return bulk_copy

    P[3, :] = SmoothInitialTemperature(P[3, :], Te_Cath)
    P[4, :] = 0.0

    Jm1 = 0.
    J = 0.0  # Initial Current

if msp.TIMESCHEME == "TVDRK3":
    P_1 = np.ones(
        (5, NBPOINTS)
    )  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
    U_1 = np.ones((4, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi,

if msp.Circuit:

    R = msp.R
    L = msp.L
    C = msp.C
    V0 = msp.V0
    print(f"~~~~~~~~~~~~~~~~ Circuit: R = {R:.2e} Ohm")
    print(f"~~~~~~~~~~~~~~~~ Circuit: L = {L:.2e} H")
    print(f"~~~~~~~~~~~~~~~~ Circuit: C = {C:.2e} F")

    X_Volt0 = np.zeros(2)  # [DeltaV, dDeltaV/dt]
    X_Volt1 = np.zeros(2)
    X_Volt2 = np.zeros(2)
    X_Volt3 = np.zeros(2)

    RHS_Volt0 = np.zeros(2)
    RHS_Volt1 = np.zeros(2)
    RHS_Volt2 = np.zeros(2)

    A_Volt = np.zeros([2, 2])
    A_Volt[0, 0] = 0.0
    A_Volt[0, 1] = 1.0
    A_Volt[1, 1] = -1 / (L * C)
    A_Volt[1, 0] = -1 / (R * C)

    dJdt = 0.0

i_save = 0
time = 0.0
iter = 0

##### Compute Si_{iz} #####
xm = (fLSIZ1 + fLSIZ2)/2
imposed_Siz = msp.SIZMAX*np.cos(math.pi*(x_center - xm)/(fLSIZ2 - msp.LSIZ1))
imposed_Siz = np.where((x_center < msp.LSIZ1)|(x_center > msp.LSIZ2), 0., imposed_Siz)
del xm


###############################################
#           FUNCTIONS DEFINING OUR MODEL
###############################################

def compute_mu(fP):
    ng = fP[0, :]
    Te = fP[3, :]

    wce = phy_const.e*fBarr/phy_const.m_e

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986

    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/fMi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate            
    elif wall_inter_type == "None":
        nu_iw = np.zeros(fP.shape[1], dtype=float)     # Ion - wall collision rate
        nu_ew = np.zeros(fP.shape[1], dtype=float)     # Electron - wall collision rate

    nu_m = ng*KEL + alpha_B*wce + nu_ew
    mu_eff_arr = (phy_const.e/(phy_const.m_e * nu_m * (1 + (wce/nu_m)**2)))

    return mu_eff_arr


@njit
def PrimToCons(fP, fU, fMi):
    fU[0, :] = fP[0, :] * fMi # rhog
    fU[1, :] = fP[1, :] * fMi # rhoi
    fU[2, :] = fP[2, :] * fP[1, :] * fMi # rhoiUi
    fU[3, :] = 3.0 / 2.0 * fP[1, :] * phy_const.e * fP[3, :]  # 3/2*ni*e*Te


@njit
def ConsToPrim(fU, fP, fMi, fA0, fJ=0.0):
    fP[0, :] = fU[0, :] / fMi # ng
    fP[1, :] = fU[1, :] / fMi # ni
    fP[2, :] = fU[2, :] / fU[1, :]  # Ui = rhoUi/rhoi
    fP[3, :] = 2.0 / 3.0 * fU[3, :] / (phy_const.e * fP[1, :])  # Te
    fP[4, :] = fP[2, :] - fJ / (fA0 * phy_const.e * fP[1, :])  # ve


@njit
def InviscidFlux(fP, fF):
    fF[0, :] = fP[0, :] * VG * Mi # rho_g*v_g
    fF[1, :] = fP[1, :] * fP[2, :] * Mi # rho_i*v_i
    fF[2, :] = (
        Mi* fP[1, :] * fP[2, :] * fP[2, :] + fP[1, :] * phy_const.e * fP[3, :]
    )  # M*n_i*v_i**2 + p_e
    fF[3, :] = 5.0 / 2.0 * fP[1, :] * phy_const.e * fP[3, :] * fP[4, :]  # 5/2n_i*e*T_e*v_e


@njit
def CumTrapz(y, d):
    n = y.shape[0]
    cuminteg = np.zeros(y.shape, dtype=float)
    
    for i in range(1, n):
        cuminteg[i] = cuminteg[i-1] + d * (y[i] + y[i-1]) / 2.0

    return cuminteg


@njit
def InitNeutralDensity(fx_center, fng_cathode, fVG, fP, fisSourceImposed, fSIZMAX, fLSIZ1, fLSIZ2):
    Eion = 12.1
    ni = fP[1,:]
    Te = fP[3,:]
    if fisSourceImposed:
        Siz_arr = compute_imposed_Siz(fx_center, fSIZMAX, fLSIZ1, fLSIZ2)
        dx_temp = (fx_center[-1] - fx_center[0])/(fx_center.shape[0] - 1)
        ng_init = fng_cathode - (1/fVG)*CumTrapz(Siz_arr, dx_temp)

    else:
        Kiz = 1.8e-13* (((1.5*Te)/Eion)**0.25) * np.exp(- 4*Eion/(3*Te))
        ng_init = fng_cathode*np.exp( - (1/fVG) * ni * Kiz * fx_center)

    return ng_init


@njit
def gradient(y, x):
    dp_dz = np.zeros(y.shape)
    dp_dz[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    return dp_dz


@njit
def compute_E(fP):

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0,:]
    ni = fP[1,:]
    Te = fP[3,:]
    ve = fP[4,:]

    me = phy_const.m_e
    wce     = phy_const.e*Barr/me   # electron cyclotron frequency
    
    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/M)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
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
        ng * KEL + alpha_B * wce + nu_ew
        )  # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e / (me* nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
        )  # Effective mobility    dp_dz  = np.gradient(ni*Te, Delta_x)

    dp_dz  = gradient(ni*Te, x_center)
    
    E = - ve / mu_eff - dp_dz / ni  # Discharge electric field
    return E


@njit
def Source(fP, fS):

    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]
    ni = fP[1, :]
    vi = fP[2, :]
    Te = fP[3, :]
    ve = fP[4, :]

    # Gamma_E = 3./2.*ni*phy_const.e*Te*ve    # Flux of internal energy
    me = phy_const.m_e
    wce = phy_const.e * fBarr / me # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    Siz_arr = np.zeros(ng.shape, dtype=float) # the final unit of Siz_arr is m^(-3).s^(-1)
    # Computing ionization source term:
    if not fisSourceImposed:
        Kiz = 1.8e-13* (((1.5*Te)/Eion)**0.25) * np.exp(- 4*Eion/(3*Te))   # Ion - neutral  collision rate          MARTIN: Change
        Siz_arr = ng*ni*Kiz
    else:
        Siz_arr = imposed_Siz

    # If ionization collision are considered in the momentum and energy equations.
    if enableIonColl:
        d_IC = 1.0
    else:
        d_IC = 0.

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[fx_center > LTHR] = 0.0
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
    phi_W = Te * np.log(np.sqrt(Mi/ (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = (
        ng * KEL + alpha_B * wce + nu_ew
        )  # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e / (me* nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
        )  # Effective mobility

    #div_u   = gradient(ve, d=Delta_x)               # To be used with 3./2. in line 160 and + phy_const.e*ni*Te*div_u  in line 231
    
    if boolPressureDiv:
        div_p = gradient(phy_const.e*ni*Te, x_center) # To be used with 5./2 and + div_p*ve in line 231    
    else:
        div_p = np.zeros(Te.shape) #this line to match the old version of the code.

    fS[0, :] = (-d_IC * Siz_arr + nu_iw * ni) * Mi # Gas Density
    fS[1, :] = (Siz_arr - nu_iw * ni) * Mi # Ion Density
    fS[2, :] = (
        d_IC * Siz_arr * VG
        - (phy_const.e / (mu_eff[:] * Mi)) * ni * ve
        - nu_iw * ni * vi
        ) * Mi # Momentum
    fS[3,:] = (
        - d_IC * Siz_arr * Eion * gamma_i * phy_const.e  +  (1.0 - d_IC)*Siz_arr* 1.5 * Te_inj* phy_const.e
        - nu_ew * ni * Ew * phy_const.e
        + (1./mu_eff) * ni * ve**2 * phy_const.e
        + div_p*ve
    )

    #+ phy_const.e*ni*Te*div_u  #- gradI_term*ni*Te*grdI          # Energy in Joule


@njit
def heatFlux(fP, fS):

    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :] # It contains ghost cells
    ni = fP[1, :] # It contains ghost cells
    Te = fP[3, :] # It contains ghost cells

    # fBarr_extended    = np.concatenate([[fBarr[0]], fBarr, [fBarr[-1]]])
    # falpha_B_extended = np.concatenate([[falpha_B[0]], falpha_B, [falpha_B[-1]]])

    me = phy_const.m_e
    wce = phy_const.e * fBarr / me # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
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
    phi_W = Te * np.log(np.sqrt(Mi/ (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = (
        ng * KEL + alpha_B * wce + nu_ew
    )  # 

    kappa = 5./2. * ni * phy_const.e**2 * Te / (phy_const.m_e * nu_m)
    kappa_perp = kappa/(1 + (wce / nu_m) ** 2)

    # kappa_12 = 0.5*(kappa[1:] + kappa[:-1])
    kappa_12 = 0.5*(kappa_perp [1:] + kappa_perp[:-1])
    grad_Te  = (Te[1:] - Te[:-1]) / ( x_center[1:] - x_center[:-1] )

    q_12     =  - 0.5*kappa_12*grad_Te  # 1/2 test just to match P.A. data
    q_source = (q_12[1:] - q_12[:-1])/Delta_x

    #fS[0, :] = (-Siz_arr + nu_iw[:] * ni[:]) * fMi # Gas Density

    fS[3,:] += (
        -q_source
    )
    delta_T_min = CFL * np.min( ( ni[1:-1] * phy_const.e * Delta_x**2 ) / kappa_perp[1:-1] )

    return delta_T_min

    #+ phy_const.e*ni*Te*div_u  #- gradI_term*ni*Te*grdI          # Energy in Joule


@njit
def TDMA(a,b,c,d):      # Thomas algorithm for the implicit solver a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p


@njit
def heatFluxImplicit(fP, fDelta_t):


    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :] # It contains ghost cells
    ni = fP[1, :] # It contains ghost cells
    Te = fP[3, :] # It contains ghost cells

    me = phy_const.m_e
    wce = phy_const.e * fBarr / me # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/Mi)
        # Limit the wall interactions to the inner channel
        nu_iw[x_center > LTHR] = 0.0
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
    phi_W = Te * np.log(np.sqrt(Mi/ (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = (
        ng * KEL + alpha_B * wce + nu_ew
    )  # 

    kappa = 5./2. * ni * phy_const.e**2 * Te / (phy_const.m_e * nu_m)
    kappa_perp = kappa/(1 + (wce / nu_m) ** 2)

    # kappa_12 = 0.5*(kappa[1:] + kappa[:-1])
    kappa_12 = 0.5*(kappa_perp [1:] + kappa_perp[:-1])

    # Simple implicit
    Coefficient = 2. / 3. * fDelta_t/( ni * phy_const.e * Delta_x)

    a_lowerDiag = - Coefficient[1:-1] * kappa_12[:-1] / (x_center[1:-1] - x_center[:-2])
    b_mainDiag  = (
        np.ones_like(Coefficient[1:-1]) + 
        Coefficient[1:-1]* ( kappa_12[:-1] * (x_center[2:] - x_center[1:-1]) + kappa_12[1:] * (x_center[1:-1] - x_center[:-2]) ) / ((x_center[1:-1] - x_center[:-2]) * (x_center[2:] - x_center[1:-1]))
        )
    c_upperDiag = - Coefficient[1:-1] * kappa_12[1:]  / (x_center[2:] - x_center[1:-1])

    d_solutionVector      = np.copy(Te[1:-1])
    d_solutionVector[0]  -= a_lowerDiag[0]*Te[0]      # Adding the boundary conditions
    d_solutionVector[-1] -= c_upperDiag[-1]*Te[-1] 

    # # Crank Nicolson constant dx
    # Coefficient = 0.5*2. / 3. * fDelta_t/( ni * fDelta_x**2)

    # a_lowerDiag = -Coefficient[1:-1]*kappa_12[:-1]
    # b_mainDiag  = np.ones_like(Coefficient[1:-1]) + Coefficient[1:-1]*(kappa_12[:-1] + kappa_12[1:])
    # c_upperDiag = -Coefficient[1:-1]*kappa_12[1:]

    # d_solutionVector      = np.copy(Te[1:-1]) - a_lowerDiag*Te[:-2] - c_upperDiag*Te[2:] + b_mainDiag*Te[1:-1]
    # d_solutionVector[0]  += a_lowerDiag[0]*Te[0]     # Adding the boundary conditions
    # d_solutionVector[-1] += c_upperDiag[-1]*Te[-1] 



    return TDMA(a_lowerDiag[1:], b_mainDiag, c_upperDiag[:-1], d_solutionVector)


# Compute the Current
@njit
def compute_I(fP, fV):

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0, :]
    ni = fP[1, :]
    vi = fP[2, :]
    Te = fP[3, :]
    ve = fP[4, :]

    me = phy_const.m_e
    wce = phy_const.e * Barr / me # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / ESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(R2 - R1))*np.sqrt(phy_const.e*Te/Mi)
        # Limit the collisions to inside the thruster
        index_LTHR = np.argmax(x_center > LTHR)
        nu_iw[index_LTHR:] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate
    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)     # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)     # Electron - wall collision rate

    # Electron momentum - transfer collision frequency
    nu_m = ng * KEL + alpha_B * wce + nu_ew

    mu_eff = (phy_const.e / (me* nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
    )  # Effective mobility

    dp_dz = gradient(ni*Te, x_center)

    value_trapz_1 = (
        np.sum(
            ( (vi / mu_eff + dp_dz / ni)[1:] + (vi / mu_eff + dp_dz / ni)[:-1] ) * (x_center[1:] - x_center[:-1]) / 2.0
            ) 
    )
    top = fV + value_trapz_1

    value_trapz_2 = (
        np.sum(
            ((1.0 / (mu_eff * ni))[1:] + (1.0 / (mu_eff * ni))[:-1]) * (x_center[1:] - x_center[:-1]) / 2.0
            ) 
    )
    value_trapz_2 = np.trapz((1.0 / (mu_eff * ni)) , x_center)
    bottom = phy_const.e * A0 * Rext + value_trapz_2

    J0 = top / bottom  # Discharge current density
    return J0 * phy_const.e * A0


@njit
def SetInlet(fP_LeftColumn, fU_ghost, fP_ghost, fJ=0.0, moment=1):
    #TODO: change the Dirichlet BCs so that a fixed value s is achieved in the frontier x=0. So the ghost value must be s_g = 2*s - s[0], where s[0] is the left value of the bulk array. Currently only v_i is computed this way to achieve the Bohm velocity at the frontier. It is not the case for n_g and T_e.
    fP_LC   = fP_LeftColumn     # renaming for more elegance 

    U_Bohm = np.sqrt(phy_const.e * fP_LC[3] / Mi)

    if not fisSourceImposed:
        if fP_LC[1] * fP_LC[2] < 0.0:
            fU_ghost[0] = (MDOT - Mi* fP_LC[1] * fP_LC[2] * A0) / (A0 * VG)
        else:
            fU_ghost[0] = MDOT / (A0 * VG)

    else:
        fU_ghost[0] = MDOT / (A0 * VG)
    
    fU_ghost[1] = fP_LC[1] * Mi
    fU_ghost[2] = -2.0 * fP_LC[1] * U_Bohm* Mi - fP_LC[1] * fP_LC[2] * Mi     # so that 0.5*(U_ghost[2] + U_LC[2]) = - u_B * U_LC[1] 
    # fU_ghost[3] = 3.0 / 2.0 * fP_LC[1] * phy_const.e * fP_LC[3]
    #fU_ghost[3] = 3.0 * fP_LC[1] * phy_const.e * fTE0 - 3./2. * fP_LC[1] * phy_const.e * fP_LC[3]     # so that the average of the ghost and the cell index 0 is 3/2*rho_i[0]*e*TE0
    fU_ghost[3] = 3.0 / 2.0 * fP_LC[1] * phy_const.e * fP_LC[3] # it is this way to try and reproduce mazallon PPS1350
    # kappa_wall = phy_const.e**2 * fP_LC[1] * fP_LC[3] * 2.38408574e+23 # Test with only anomalous transport
    # Delta_x = 0.000125
    # T_Ghost = fP_LC[1]*U_Bohm*(2*phy_const.e*fP_LC[3])*Delta_x/kappa_wall
    # fU_ghost[3] = 3. / 2. * fP_LC[1] * phy_const.e * T_Ghost
    # if fU_ghost[3] < 0.1:
    #     print("WARNING: Ghost cell too cold")
    #     print( fU_ghost[3])

    fP_ghost[0] = fU_ghost[0] / Mi # ng
    fP_ghost[1] = fU_ghost[1] / Mi # ni
    fP_ghost[2] = fU_ghost[2] / fU_ghost[1]  # Ui
    fP_ghost[3] = 2.0 / 3.0 * fU_ghost[3] / (phy_const.e * fP_ghost[1])  # Te
    fP_ghost[4] = fP_ghost[2] - fJ / (A0 * phy_const.e * fP_ghost[1])  # ve


@njit
def SetOutlet(fP_RightColumn, fU_ghost, fP_ghost, J=0.0):
    #TODO: change the Dirichlet BCs so that a fixed value s is achieved in the frontier x=0. So the ghost value must be s_g = 2*s - s[0], where s[0] is the left value of the bulk array. It is not the case for T_e.
    fP_RC   = fP_RightColumn    # renaming for more elegance

    fU_ghost[0] = fP_RC[0] * Mi
    fU_ghost[1] = fP_RC[1] * Mi
    fU_ghost[2] = fP_RC[1] * fP_RC[2] * Mi
    fU_ghost[3] = 3.0 / 2.0 * fP_RC[1] * phy_const.e * Te_Cath
    # fU_ghost[3] = 3.0 * fP_RC[1] * phy_const.e * Te_Cath - 3./2. * fP_RC[1] * phy_const.e * fP_RC[3] # so that the average temperature between the last cell and the ghost cell is Te_cath

    fP_ghost[0] = fU_ghost[0] / Mi # ng
    fP_ghost[1] = fU_ghost[1] / Mi # ni
    fP_ghost[2] = fU_ghost[2] / fU_ghost[1]  # Ui
    fP_ghost[3] = 2.0 / 3.0 * fU_ghost[3] / (phy_const.e * fP_ghost[1])  # Te
    fP_ghost[4] = fP_ghost[2] - J / (A0 * phy_const.e * fP_ghost[1])  # ve


mu_eff_arr = compute_mu(P)
print(f"Checking mu value [m^2 s^-1 V^-1]. Bmax = {msp.BMAX*10000:.0f}\talpha_B1 = {alpha_B1:.4e}\talpha_B2 = {alpha_B2:.4e}")
#print(mu_eff_arr)
print(f"\tMean value over domain space  {np.mean(mu_eff_arr):.6e}")
print(f"\tStd deviation:                {np.std(mu_eff_arr):.6e}")

# We initialize the conservative variables
PrimToCons(P, U, Mi)

if TIMESCHEME == "Forward Euler":
    ##########################################################################################
    #           Loop with Forward Euler                                                      #
    #           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
    ##########################################################################################
    J = compute_I(P, V)
    while time < TIMEFINAL:

        # Save results
        if (iter % SAVERATE) == 0:
            # Compute the electric field.
            Efield = compute_E(P)                
            # Save the variant data
            filenameTemp = ResultsData + "/MacroscopicVars_" + f"{i_save:06d}" + ".pkl"
            pickle.dump(
                [time, P, U, P_LeftGhost, P_RightGhost, J, Efield, V], open(filenameTemp, "wb")
            )
            i_save += 1
            print(
                "Iter = ",
                iter,
                "\tTime = {:.2f}~µs".format(time / 1e-6),
                "\tI = {:.4f}~A".format(J),
                "\tJ = {:.3e} A/m2".format(J/A0),
            )
        # where I stopped changing the code.
        # Set the boundaries
        SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 1)
        SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, Mi, A0, Te_Cath, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

        # Compute the convective Delta t
        Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), NBPOINTS, Mi, CFL, x_center_extended)
        #print(Delta_t)
        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
            np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
            F_cell,
            F_interf,
            NBPOINTS,
            Mi,
            VG,
        )

        # Compute the source in the center of the cell
        Source(P, S, Barr, boolSizImposed, Eion, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x, boolPressureDiv, gamma_i, Te_inj)
        if HEATFLUX and not IMPlICIT:
            dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x, CFL)
            Delta_t = min(dt_HF, Delta_t)
        elif HEATFLUX and IMPlICIT:
            dt_HF = Delta_t
            Te = heatFluxImplicit(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x_extended, Delta_t)
            print(Te)
        


        # Update the solution
        U[:, :] = (
            U[:, :]
            - Delta_t
            / Delta_x
            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
            + Delta_t * S[:, :]
        )
        # Prevent the energy to be strictly negative
        U[3,:] = np.where(U[3,:] >= 0., U[3,:], 0.)

        # Compute the current
        J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, Mi, A0, J)

        time += Delta_t
        iter += 1
