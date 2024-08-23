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

from modules.simu_params import SimuParameters

##########################################################
#           FUNCTIONS
##########################################################

@njit
def gradient(y, x):
    dp_dz = np.zeros(y.shape)
    dp_dz[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    return dp_dz


@njit
def compute_Rei_empirical(fP, fB, fESTAR, wall_inter_type:str, fR1, fR2, fM, fx_center, fLTHR, fKEL, falpha_B):
    
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0,:]
    ni = fP[1,:]
    Te = fP[3,:]
    ve = fP[4,:]

    me = phy_const.m_e
    wce= phy_const.e*fB/me   # electron cyclotron frequency
    
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

    nu_m = (
        ng * fKEL + falpha_B * wce + nu_ew
        )  # Electron momentum - transfer collision frequency

    vey = (wce/nu_m)*ve

    Rei_emp = - me * ni * vey * nu_m

    return Rei_emp


@njit
def compute_Rei_saturated(fP, fM, fx_center):
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ni = fP[1,:]
    vi = fP[2,:]
    Te = fP[3,:]
    uB = np.sqrt(phy_const.e*Te / fM) # Bohm velocity or sound speed.

    deriv_factor = gradient(vi*ni*Te, fx_center)

    Rei_saturated = (phy_const.e/(16 * np.sqrt(6) * uB)) * np.abs(deriv_factor)

    return Rei_saturated


@njit
def compute_E(fP, fB, fESTAR, wall_inter_type:str, fR1, fR2, fM, fx_center, fLTHR, fKEL, falpha_B):

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP[0,:]
    ni = fP[1,:]
    Te = fP[3,:]
    ve = fP[4,:]

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

    dp_dz  = gradient(ni*Te, fx_center)
    
    E = - ve / mu_eff - dp_dz / ni  # Discharge electric field
    return E


@njit
def cumTrapz(y, dx_arr):
    n = y.shape[0]
    cuminteg = np.zeros(y.shape, dtype=float)
    cuminteg[0] = dx_arr[0] * y[0]
    for i in range(1, n):
        cuminteg[i] = cuminteg[i-1] + dx_arr[i] * y[i]

    return cuminteg


@njit
def compute_phi(fE, fDelta_x, fJ, fV, fRext):
    
    phi = fV - fJ * fRext - cumTrapz(fE, fDelta_x)  # Discharge electrostatic potential
    return phi


@njit
def compute_heat_flux(fP_ext, fBarr, fESTAR, wall_inter_type, fR1, fR2, fMi, fx_center, fLTHR, fKEL, falpha_B):
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = fP_ext[0, :] # It contains ghost cells
    ni = fP_ext[1, :] # It contains ghost cells
    Te = fP_ext[3, :] # It contains ghost cells

    # fBarr_extended    = np.concatenate([[fBarr[0]], fBarr, [fBarr[-1]]])
    # falpha_B_extended = np.concatenate([[falpha_B[0]], falpha_B, [falpha_B[-1]]])

    me = phy_const.m_e
    wce = phy_const.e * fBarr / me # electron cyclotron frequency



    #############################
    #       Compute the rates   #
    #############################
    Eion = 12.1  # Ionization energy
    gamma_i = 3  # Excitation coefficient

    sigma = 2.0 * Te / fESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(fR2 - fR1))*np.sqrt(phy_const.e*Te/fMi)
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
    phi_W = Te * np.log(np.sqrt(fMi/ (2 * np.pi * me)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    nu_m = (
        ng * fKEL + falpha_B * wce + nu_ew
    )  # 

    kappa = 5./2. * ni * phy_const.e**2 * Te / (phy_const.m_e * nu_m)
    kappa_perp = kappa/(1 + (wce / nu_m) ** 2)

    # kappa_12 = 0.5*(kappa[1:] + kappa[:-1])
    kappa_12 = 0.5*(kappa_perp [1:] + kappa_perp[:-1])
    grad_Te  = (Te[1:] - Te[:-1]) / ( fx_center[1:] - fx_center[:-1] )

    q_12     =  - kappa_12*grad_Te

    return 0.5*(q_12[1:] + q_12[:-1])


##########################################################
#           MAIN BLOCK
##########################################################

if __name__ == '__main__':
        
    ##########################################################
    #           POST-PROC PARAMETERS
    ##########################################################
    Results     = sys.argv[1]
    ResultConfig = Results+'/Configuration.cfg'
    msp = SimuParameters(ResultConfig)

    ##########################################################
    #           CONFIGURE PHYSICAL PARAMETERS
    ##########################################################

    MDOT    = msp.MDOT
    Mi      = msp.Mi
    A0      = msp.A0
    VG      = msp.VG
    NI0     = msp.NI0
    TE0     = msp.TE0
    Te_Cath     = msp.Te_Cath    
    ESTAR   = msp.ESTAR
    R1      = msp.R1
    R2      = msp.R2
    LX  = msp.LX
    LTHR    = msp.LTHR
    KEL     = msp.KEL
    Rext        = msp.Rext
    V0           = msp.V0
    boolCircuit = msp.Circuit    
    HEATFLUX    = msp.HEATFLUX
    boolIonColl = msp.boolIonColl
    boolSizImposed  = msp.boolSizImposed
    wall_inter_type     = msp.wall_inter_type

    ##########################################################
    #           NUMERICAL PARAMETERS
    ##########################################################

    Resultsdir  = msp.Results
    TIMESCHEME  = msp.TIMESCHEME
    TIMEFINAL   = msp.TIMEFINAL
    SAVERATE    = msp.SAVERATE
    CFL         = msp.CFL
    IMPlICIT    = msp.IMPlICIT

    START_FROM_INPUT  = msp.START_FROM_INPUT
    if START_FROM_INPUT:
        INPUT_DIR  = msp.INPUT_DIR

    ##########################################################
    #           Collects Large Unvariant Parameters
    ##########################################################
    ResultsData = Results+"/Data"    
    with open(ResultsData + "/MacroscopicUnvariants.pkl", 'rb') as f:
        [B, x_mesh, x_center, alpha_B] = pickle.load(f)
    Delta_x = x_mesh[1:] - x_mesh[:-1]
    x_center_extended = np.insert(x_center, 0, -x_center[0])
    x_center_extended = np.append(x_center_extended, x_center[-1] + Delta_x[-1])
    nu_anom_eff     = alpha_B * (phy_const.e/phy_const.m_e) * B

    ##########################################################
    #           Plot parameters
    ##########################################################

    #plt.style.use('classic')
    #plt.rcParams["font.family"] = 'Times New Roman'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams["font.size"] = 15
    plt.rcParams["lines.linewidth"] = 2
    plt.rc('axes', unicode_minus=False)
    tickfontsize = 11
    axisfontsize = 14

    ########################################################
    #               Make the plots
    ########################################################

    ResultsFigs = Results+"/Figs"
    if not os.path.exists(ResultsFigs):
        os.makedirs(ResultsFigs)

    # open all the files in the directory and sort them to do the video in order
    files       = glob.glob(ResultsData + "/MacroscopicVars_*.pkl")
    filesSorted = sorted(files, key = lambda x: os.path.getmtime(x), reverse=True)
    files.sort(key=os.path.getmtime)


    Current = np.zeros(np.shape(files)[0])
    CurrentDensity = np.zeros(np.shape(files)[0])
    Voltage = np.zeros(np.shape(files)[0])
    time    = np.zeros(np.shape(files)[0])

    for i_save, file in enumerate(files):
        
        with open(file, 'rb') as f:
            [t, P, U, P_LeftGhost, P_RightGhost, J, Efield, V] = pickle.load(f)
        
        # Save the current
        Current[i_save] = J
        CurrentDensity[i_save] = np.mean(P[1,:]*phy_const.e*(P[2,:] - P[4,:]))
        time[i_save]    = t
        Voltage[i_save] = V

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

    ######################################
    #    Plot the effective anomalous freq.
    ######################################
    f, ax = plt.subplots(figsize=(10,5))

    nu_plus = nu_anom_eff[np.argwhere(nu_anom_eff > 0.)]
    x_plus   = x_center[np.argwhere(nu_anom_eff > 0.)]
    nu_minus = nu_anom_eff[np.argwhere(nu_anom_eff < 0.)]
    x_minus   = x_center[np.argwhere(nu_anom_eff < 0.)]
    ax_b = ax.twinx()
    ax.semilogy(x_plus*100, nu_plus, 'g+', label = "$\\nu_{anom}, pos. part$")
    ax.semilogy(x_minus*100, -nu_minus, 'g.', label = "$\\nu_{anom}, neg. part$")        
    ax.set_ylabel('$\\nu_{anom}$ [Hz]', fontsize=axisfontsize)
    ax.set_xlabel('$x$ [cm]', fontsize=axisfontsize)
    ax.yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
    ax.xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
    ax.legend(fontsize=tickfontsize)
    ax_b.plot(x_center*100, B, 'r:')
    ax_b.set_ylabel('$B$ [mT]', color='r', fontsize=axisfontsize)
    plt.savefig(ResultsFigs+"/effective_anom_nu.png", bbox_inches='tight')
    plt.close()

    ########################################
    #        One chart per frmae routine
    ########################################

    for i_save, file in enumerate(files):

        print("Preparing plot for i = ", i_save)

        with open(file, 'rb') as f:
            [t, P, U, P_LeftGhost, P_RightGhost, J, Efield, V] = pickle.load(f)

        phi = compute_phi(Efield, Delta_x, J, V, Rext)
        
        Rei_emp = compute_Rei_empirical(P, B, ESTAR, wall_inter_type, R1, R2, Mi, x_center, LTHR, KEL, alpha_B)
        #print(Rei_emp.shape)

        Rei_sat = compute_Rei_saturated(P, Mi, x_center)
        #print(Rei_sat.shape)

        q_array = compute_heat_flux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),  np.concatenate([[B[0]], B, [B[-1]]]), ESTAR, wall_inter_type, R1, R2, Mi, x_center_extended, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]))
        
        f = plt.figure(figsize = (12,12.5))

        ax1 = plt.subplot2grid((5,2),(0,0))
        ax2 = plt.subplot2grid((5,2),(1,0))
        ax3 = plt.subplot2grid((5,2),(2,0))
        ax4 = plt.subplot2grid((5,2),(3,0))
        ax5 = plt.subplot2grid((5,2),(4,0))
        ax6 = plt.subplot2grid((5,2),(0,1))
        ax7 = plt.subplot2grid((5,2),(1,1))
        ax8 = plt.subplot2grid((5,2),(2,1))
        ax9 = plt.subplot2grid((5,2),(3,1))
        ax10= plt.subplot2grid((5,2),(4,1))

        ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

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
        ax[1].plot(x_center*100, Efield/1000.)
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
        ax[2].set_xticklabels([])
        ax[2].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[2].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])
        ax[2].set_ylim([0., max(1.0, np.max(P[1,:]/1e18))])
        
        ax_b=ax[3].twinx()
        ax[3].plot(x_center*100, Rei_emp, label="$R_{ei}^{emp}$")
        ax[3].plot(x_center*100, Rei_sat, label="$R_{ei}^{sat}$")
        ax[3].set_ylabel('$R_{ei}$ [N m$^{-3}$]', fontsize=axisfontsize)
        ax[3].set_xlabel('$x$ [cm]', fontsize=axisfontsize)        
        ax[3].yaxis.set_tick_params(labelsize=tickfontsize)
        ax[3].xaxis.set_tick_params(labelsize=tickfontsize)
        ax[3].legend(fontsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        ax[4].plot(time*1000., Current, label='Eff. $I_d$')
        ax[4].plot(time*1000., CurrentDensity*A0, label='$I_d = A_0 e (\\Gamma_i - \\Gamma_e)$')
        ax[4].set_ylabel('$I_d$ [A]', fontsize=axisfontsize)
        ax[4].set_xlabel('$t$ [ms]', fontsize=axisfontsize)
        ax[4].plot([time[i_save]*1000., time[i_save]*1000.],
                   [Current[i_save], CurrentDensity[i_save]*A0],
                   'ko', markersize=5)
        ax[4].yaxis.set_tick_params(size=5, width=1.5, labelsize=tickfontsize)
        ax[4].xaxis.set_tick_params(size=5, width=1.5, labelsize=tickfontsize)
        ax[4].legend(fontsize=tickfontsize)
        ax[4].set_ylim([0., max(10.0, 1.1*Current.max())])
        
        ax_b=ax[5].twinx()
        ax[5].plot(x_center*100, P[3,:])
        ax[5].set_ylabel('$T_e$ [eV]', fontsize=axisfontsize)
        ax[5].set_xticklabels([])
        ax[5].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[5].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        ax_b=ax[6].twinx()
        ax[6].plot(x_center*100, P[2,:]/1000, label="$v_i$")
        ax[6].plot(x_center*100, np.sqrt(phy_const.e*P[3,:]/(131.293*phy_const.m_u))/1000.,'g--', label="$v_{Bohm}$")
        ax[6].set_ylabel('$v_i$ [km/s]', fontsize=axisfontsize)
        ax[6].set_xticklabels([])
        ax[6].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[6].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])
        ax[6].legend(fontsize=tickfontsize)

        ax_b = ax[7].twinx()
        ax[7].plot(x_center*100, P[4,:]/1000)
        ax[7].set_ylabel('$v_e$ [km/s]', fontsize=axisfontsize)
        ax[7].set_xticklabels([])
        ax[7].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[7].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
        ax_b = ax[8].twinx()
        ax[8].plot(x_center*100, j_of_x)
        ax[8].set_ylabel('$J_d$ [A.m$^{-2}$]', fontsize=axisfontsize)
        ax[8].set_xticklabels([])
        ax[8].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[8].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[8].set_ylim([0.,max(800., 1.1*j_of_x.max())])
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        ax_b = ax[9].twinx()
        ax[9].plot(x_center*100, q_array)        
        ax[9].set_ylabel('Axial heat flux $q_x$ [W m$^{-2}$]', fontsize=axisfontsize)
        ax[9].set_xlabel('$x$ [cm]', fontsize=axisfontsize)
        ax[9].yaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax[9].xaxis.set_tick_params(which='both', size=5, width=1.5, labelsize=tickfontsize)
        ax_b.plot(x_center*100, B, 'r:')
        ax_b.set_yticklabels([])

        title = 'time = '+ str(round(t/1e-6, 4))+'$\\mu$s'
        f.suptitle(title)

        for axis in ax:
            axis.grid(True)
            
        plt.tight_layout()

        #plt.subplots_adjust(wspace = 0.4, hspace=0.2)
        
        plt.savefig(ResultsFigs+f"/MacroscopicVars_New_{i_save:06d}.png", bbox_inches='tight')
        plt.close()
        

    os.system("ffmpeg -r 10 -i "+ResultsFigs+"/MacroscopicVars_New_%d.png -vcodec mpeg4 -y -vb 20M "+ResultsFigs+"Evolution.mp4")

