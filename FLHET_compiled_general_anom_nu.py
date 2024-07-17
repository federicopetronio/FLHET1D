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

##########################################################
#           CONFIGURE PHYSICAL PARAMETERS
##########################################################

def compute_B_array(fx_center, fBMAX, fB0, fBLX, fLX, fLTHR, fLB1, fLB2):

    a1 = (fBMAX - fB0)/(1 - math.exp(-fLTHR**2/(2*fLB1**2)))
    a2 = (fBMAX - fBLX)/(1 - math.exp(-(fLX - fLTHR)**2/(2*fLB2**2)))
    b1 = fBMAX - a1
    b2 = fBMAX - a2
    Barr1 = a1*np.exp(-(fx_center - fLTHR)**2/(2*fLB1**2)) + b1
    Barr2 = a2*np.exp(-(fx_center - fLTHR)**2/(2*fLB2**2)) + b2    # Magnetic field outside the thruster

    Barr = np.where(fx_center <= fLTHR, Barr1, Barr2)

    return Barr


def compute_alphaB_array(fxcenter, falphaB1, falphaB2, fLTHR, fNBPOINTS_INIT):

    NBPOINTS = fxcenter.shape[0]
    alpha_B = (np.ones(NBPOINTS) * falphaB1)  # Anomalous transport coefficient inside the thruster
    alpha_B = np.where(fxcenter < fLTHR, alpha_B, falphaB2)  # Anomalous transport coefficient in the plume
    alpha_B_smooth = np.copy(alpha_B)

    # smooth between alpha_B1 and alpha_B2
    nsmooth_o2 = fNBPOINTS_INIT//20
    for index in range(nsmooth_o2, NBPOINTS - (nsmooth_o2+1)):
        alpha_B_smooth[index] = np.mean(alpha_B[ index-nsmooth_o2 : index+(nsmooth_o2+1) ])
    
    return alpha_B_smooth


@njit
def compute_imposed_Siz(fx_center, fSIZMAX, fLSIZ1, fLSIZ2):
    xm = (fLSIZ1 + fLSIZ2)/2
    Siz = fSIZMAX*np.cos(math.pi*(fx_center - xm)/(fLSIZ2 - fLSIZ1))
    Siz = np.where((fx_center < fLSIZ1)|(fx_center > fLSIZ2), 0., Siz)

    return Siz

##########################################################
#           Formulas defining our model                  #
##########################################################

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
def InviscidFlux(fP, fF, fVG, fMi):
    fF[0, :] = fP[0, :] * fVG * fMi # rho_g*v_g
    fF[1, :] = fP[1, :] * fP[2, :] * fMi # rho_i*v_i
    fF[2, :] = (
        fMi* fP[1, :] * fP[2, :] * fP[2, :] + fP[1, :] * phy_const.e * fP[3, :]
    )  # M*n_i*v_i**2 + p_e
    fF[3, :] = 5.0 / 2.0 * fP[1, :] * phy_const.e * fP[3, :] * fP[4, :]  # 5/2n_i*e*T_e*v_e


def CompareIonizationTypes(fx_center, fP, fSIZMAX, fLSIZ1, fLSIZ2):
    ng = fP[0, :]
    ni = fP[1, :]
    Te = fP[3, :]

    Eion = 12.1
    Kiz = 1.8e-13* (((1.5*Te)/Eion)**0.25) * np.exp(- 4*Eion/(3*Te))

    xm = (fLSIZ1 + fLSIZ2)/2
    imposedSiz = fSIZMAX*np.cos(math.pi*(fx_center - xm)/(fLSIZ2 - fLSIZ1))
    imposedSiz = np.where((fx_center < fLSIZ1)|(fx_center > fLSIZ2), 0., imposedSiz)

    fig1 = plt.figure(figsize=(12,9))
    plt.plot(fx_center*100, imposedSiz, label='Imposed $S_{iz}$')
    plt.plot(fx_center*100, ng*ni*Kiz, label='$S_{iz} = n_g n_i K_{iz}$')
    plt.xlabel("$x$ [cm]", fontsize=14)
    plt.ylabel("$S_{iz}$ [m$^{-3}$.s$^{-1}$]", fontsize=14)
    plt.legend(fontsize=11)
    plt.show()


@njit
def CumTrapz(y, d):
    n = y.shape[0]
    cuminteg = np.zeros(y.shape, dtype=float)
    
    for i in range(1, n):
        cuminteg[i] = cuminteg[i-1] + d * (y[i] + y[i-1]) / 2.0

    return cuminteg


@njit
def IntegralSiz(fx_center, fSIZMAX, fLSIZ1, fLSIZ2):

    xm = (fLSIZ1 + fLSIZ2)/2
    dlsiz = fLSIZ2 - fLSIZ1
    integ = fSIZMAX*(dlsiz/np.pi)*( np.sin( np.pi*(fx_center - xm)/dlsiz ) + 1.0 )
    integ = np.where(fx_center < fLSIZ1, 0., integ)
    integ = np.where(fx_center > fLSIZ2, 2*fSIZMAX*(dlsiz/np.pi), integ)

    return integ 


@njit
def InitNeutralDensity(fx_center, fng_cathode, fVG, fP, fisSourceImposed, fSIZMAX, fLSIZ1, fLSIZ2):
    Eion = 12.1
    ni = fP[1,:]
    Te = fP[3,:]
    if fisSourceImposed:
        Siz_arr = compute_imposed_Siz(fx_center, fSIZMAX, fLSIZ1, fLSIZ2)
        dx_temp = (fx_center[-1] - fx_center[0])/(fx_center.shape[0] - 1)
        ng_init = fng_cathode - (1/fVG)*CumTrapz(Siz_arr, dx_temp)
        #ng_init = fng_cathode - (1/fVG)*IntegralSiz(fx_center, fSIZMAX, fLSIZ1, fLSIZ2)

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


def compute_mu(fP, fBarr, fESTAR, wall_inter_type:str, fR1, fR2, fMi, fx_center, fLTHR, fKEL, falpha_B):
    
    ng = fP[0, :]
    Te = fP[3, :]

    wce = phy_const.e*fBarr/phy_const.m_e

    sigma = 2.0 * Te / fESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986

    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(fR2 - fR1))*np.sqrt(phy_const.e*Te/fMi)
        # Limit the wall interactions to the inner channel
        nu_iw[fx_center > fLTHR] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate            
    elif wall_inter_type == "None":
        nu_iw = np.zeros(fP.shape[1], dtype=float)     # Ion - wall collision rate
        nu_ew = np.zeros(fP.shape[1], dtype=float)     # Electron - wall collision rate

    nu_m = ng*fKEL + falpha_B*wce + nu_ew
    mu_eff_arr = (phy_const.e/(phy_const.m_e * nu_m * (1 + (wce/nu_m)**2)))

    return mu_eff_arr


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
def Source(fP, fS, fBarr, fisSourceImposed, fenableIonColl, wall_inter_type:str,fx_center, fimposed_Siz, fESTAR, fMi, fR1, fR2, fLTHR, fKEL, falpha_B, fVG, fDelta_x):

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
    Eion = 12.1  # Ionization energy
    gamma_i = 3  # Excitation coefficient

    Siz_arr = np.zeros(ng.shape, dtype=float) # the final unit of Siz_arr is m^(-3).s^(-1)
    # Computing ionization source term:
    if not fisSourceImposed:
        Kiz = 1.8e-13* (((1.5*Te)/Eion)**0.25) * np.exp(- 4*Eion/(3*Te))   # Ion - neutral  collision rate          MARTIN: Change
        Siz_arr = ng*ni*Kiz
    else:
        Siz_arr = fimposed_Siz

    # If ionization collision are considered in the momentum and energy equations.
    if fenableIonColl:
        d_IC = 1.0
    else:
        d_IC = 0.

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
        )  # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e / (me* nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
        )  # Effective mobility

    #div_u   = gradient(ve, d=Delta_x)               # To be used with 3./2. in line 160 and + phy_const.e*ni*Te*div_u  in line 231
    div_p = gradient(phy_const.e*ni*Te, fx_center) # To be used with 5./2 and + div_p*ve in line 231    
    # div_p = np.zeros(Te.shape)

    fS[0, :] = (-d_IC * Siz_arr + nu_iw * ni) * fMi # Gas Density
    fS[1, :] = (Siz_arr - nu_iw * ni) * fMi # Ion Density
    fS[2, :] = (
        d_IC * Siz_arr * fVG
        - (phy_const.e / (mu_eff[:] * fMi)) * ni * ve
        - nu_iw * ni * vi
        ) * fMi # Momentum
    fS[3,:] = (
        - d_IC * Siz_arr * Eion * gamma_i * phy_const.e
        - nu_ew * ni * Ew * phy_const.e
        + (1./mu_eff) * ni * ve**2 * phy_const.e
        + div_p*ve
    )

    #+ phy_const.e*ni*Te*div_u  #- gradI_term*ni*Te*grdI          # Energy in Joule


@njit
def heatFlux(fP, fS, fBarr, wall_inter_type:str,fx_center, fESTAR, fMi, fR1, fR2, fLTHR, fKEL, falpha_B, fDelta_x):

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
    q_source = (q_12[1:] - q_12[:-1])/fDelta_x

    #fS[0, :] = (-Siz_arr + nu_iw[:] * ni[:]) * fMi # Gas Density

    fS[3,:] += (
        -q_source
    )
    delta_T_min = np.min( ( ni[1:-1] * phy_const.e * fDelta_x**2 ) / kappa_perp[1:-1] )

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
def heatFluxImplicit(fP, fBarr, wall_inter_type:str, fx_center, fESTAR, fMi, fR1, fR2, fLTHR, fKEL, falpha_B, fDelta_x, fDelta_t):


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

    # Simple implicit
    Coefficient = 2. / 3. * fDelta_t/( ni * phy_const.e * fDelta_x)

    a_lowerDiag = - Coefficient[1:-1] * kappa_12[:-1] / (fx_center[1:-1] - fx_center[:-2])
    b_mainDiag  = (
        np.ones_like(Coefficient[1:-1]) + 
        Coefficient[1:-1]* ( kappa_12[:-1] * (fx_center[2:] - fx_center[1:-1]) + kappa_12[1:] * (fx_center[1:-1] - fx_center[:-2]) ) / ((fx_center[1:-1] - fx_center[:-2]) * (fx_center[2:] - fx_center[1:-1]))
        )
    c_upperDiag = - Coefficient[1:-1] * kappa_12[1:]  / (fx_center[2:] - fx_center[1:-1])

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
def compute_I(fP, fV, fBarr, wall_inter_type:str,fx_center, fESTAR, fMi, fR1, fR2, fLTHR, fKEL, falpha_B, fDelta_x, fA0, fRext):

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
    wce = phy_const.e * fBarr / me # electron cyclotron frequency

    #############################
    #       Compute the rates   #
    #############################

    sigma = 2.0 * Te / fESTAR  # SEE yield
    sigma[sigma > 0.986] = 0.986
    if wall_inter_type == "Default":
        # nu_iw value before Martin changed the code for Charoy's test cases.    
        nu_iw = (4./3.)*(1./(fR2 - fR1))*np.sqrt(phy_const.e*Te/fMi)
        # Limit the collisions to inside the thruster
        index_LTHR = np.argmax(fx_center > fLTHR)
        nu_iw[index_LTHR:] = 0.0
        nu_ew = nu_iw / (1.0 - sigma)  # Electron - wall collision rate
    elif wall_inter_type == "None":
        nu_iw = np.zeros(Te.shape, dtype=float)     # Ion - wall collision rate
        nu_ew = np.zeros(Te.shape, dtype=float)     # Electron - wall collision rate

    nu_ew = nu_iw / (1 - sigma)  # Electron - wall collision rate

    # Electron momentum - transfer collision frequency
    nu_m = ng * fKEL + falpha_B * wce + nu_ew

    mu_eff = (phy_const.e / (me* nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
    )  # Effective mobility

    dp_dz = gradient(ni*Te, fx_center)

    value_trapz_1 = (
        np.sum(
            ( (vi / mu_eff + dp_dz / ni)[1:] + (vi / mu_eff + dp_dz / ni)[:-1] ) * (fx_center[1:] - fx_center[:-1]) / 2.0
            ) 
    )
    top = fV + value_trapz_1

    value_trapz_2 = (
        np.sum(
            ((1.0 / (mu_eff * ni))[1:] + (1.0 / (mu_eff * ni))[:-1]) * (fx_center[1:] - fx_center[:-1]) / 2.0
            ) 
    )
    value_trapz_2 = np.trapz((1.0 / (mu_eff * ni)) , fx_center)
    bottom = phy_const.e * fA0 * fRext + value_trapz_2

    I0 = top / bottom  # Discharge current density
    return I0 * phy_const.e * fA0


@njit
def SetInlet(fP_LeftColumn, fU_ghost, fP_ghost, fMi, fisSourceImposed, fMDOT, fA0, fVG, fTe_cath, fJ=0.0, moment=1):
    #TODO: change the Dirichlet BCs so that a fixed value s is achieved in the frontier x=0. So the ghost value must be s_g = 2*s - s[0], where s[0] is the left value of the bulk array. Currently only v_i is computed this way to achieve the Bohm velocity at the frontier. It is not the case for n_g and T_e.
    fP_LC   = fP_LeftColumn     # renaming for more elegance 

    U_Bohm = np.sqrt(phy_const.e * fP_LC[3] / fMi)

    if not fisSourceImposed:
        if fP_LC[1] * fP_LC[2] < 0.0:
            fU_ghost[0] = (fMDOT - fMi* fP_LC[1] * fP_LC[2] * fA0) / (fA0 * fVG)
        else:
            fU_ghost[0] = fMDOT / (fA0 * fVG)

    else:
        fU_ghost[0] = fMDOT / (fA0 * fVG)
    
    fU_ghost[1] = fP_LC[1] * fMi
    fU_ghost[2] = -2.0 * fP_LC[1] * U_Bohm* fMi - fP_LC[1] * fP_LC[2] * fMi     # so that 0.5*(U_ghost[2] + U_LC[2]) = - u_B * U_LC[1] 
    # fU_ghost[3] = 3.0 / 2.0 * fP_LC[1] * phy_const.e * fP_LC[3]
    fU_ghost[3] = 3.0 / 2.0 * fP_LC[1] * phy_const.e * 10.0 # Test. TODO: change so that in average, T_e = 10.0 eV. Here not the case. 
    # kappa_wall = phy_const.e**2 * fP_LC[1] * fP_LC[3] * 2.38408574e+23 # Test with only anomalous transport
    # Delta_x = 0.000125
    # T_Ghost = fP_LC[1]*U_Bohm*(2*phy_const.e*fP_LC[3])*Delta_x/kappa_wall
    # fU_ghost[3] = 3. / 2. * fP_LC[1] * phy_const.e * T_Ghost
    # if fU_ghost[3] < 0.1:
    #     print("WARNING: Ghost cell too cold")
    #     print( fU_ghost[3])

    fP_ghost[0] = fU_ghost[0] / fMi # ng
    fP_ghost[1] = fU_ghost[1] / fMi # ni
    fP_ghost[2] = fU_ghost[2] / fU_ghost[1]  # Ui
    fP_ghost[3] = 2.0 / 3.0 * fU_ghost[3] / (phy_const.e * fP_ghost[1])  # Te
    fP_ghost[4] = fP_ghost[2] - fJ / (fA0 * phy_const.e * fP_ghost[1])  # ve


@njit
def SetOutlet(fP_RightColumn, fU_ghost, fP_ghost, fMi, fA0, fTe_Cath, J=0.0):
    #TODO: change the Dirichlet BCs so that a fixed value s is achieved in the frontier x=0. So the ghost value must be s_g = 2*s - s[0], where s[0] is the left value of the bulk array. It is not the case for T_e.
    fP_RC   = fP_RightColumn    # renaming for more elegance

    fU_ghost[0] = fP_RC[0] * fMi
    fU_ghost[1] = fP_RC[1] * fMi
    fU_ghost[2] = fP_RC[1] * fP_RC[2] * fMi
    fU_ghost[3] = 3.0 / 2.0 * fP_RC[1] * phy_const.e * fTe_Cath

    fP_ghost[0] = fU_ghost[0] / fMi # ng
    fP_ghost[1] = fU_ghost[1] / fMi # ni
    fP_ghost[2] = fU_ghost[2] / fU_ghost[1]  # Ui
    fP_ghost[3] = 2.0 / 3.0 * fU_ghost[3] / (phy_const.e * fP_ghost[1])  # Te
    fP_ghost[4] = fP_ghost[2] - J / (fA0 * phy_const.e * fP_ghost[1])  # ve


##########################################################
#           Functions defining our numerics              #
##########################################################

# TODO: These are vector. Better allocate them
@njit
def computeMaxEigenVal_e(fP, fMi):

    U_Bohm = np.sqrt(phy_const.e * fP[3, :] / fMi)

    return np.maximum(np.abs(U_Bohm - fP[4, :]) * 2, np.abs(U_Bohm + fP[4, :]) * 2)


@njit
def computeMaxEigenVal_i(fP, fMi):

    U_Bohm = np.sqrt(phy_const.e * fP[3, :] / fMi)

    # return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]
    return np.maximum(np.abs(U_Bohm - fP[2, :]), np.abs(U_Bohm + fP[2, :]))


@njit
def NumericalFlux(fP, fU, fF_cell, fF_interf, fNBPOINTS, fMi, fVG):

    # Compute the max eigenvalue
    lambda_max_i_R = computeMaxEigenVal_i(fP[:, 1 : fNBPOINTS + 2], fMi)
    lambda_max_i_L = computeMaxEigenVal_i(fP[:, 0 : fNBPOINTS + 1], fMi)
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)

    lambda_max_e_R = computeMaxEigenVal_e(fP[:, 1 : fNBPOINTS + 2], fMi)
    lambda_max_e_L = computeMaxEigenVal_e(fP[:, 0 : fNBPOINTS + 1], fMi)
    lambda_max_e_12 = np.maximum(lambda_max_e_L, lambda_max_e_R)

    # Compute the flux at the interface
    fF_interf[0, :] = 0.5 * (
        fF_cell[0, 0 : fNBPOINTS + 1] + fF_cell[0, 1 : fNBPOINTS + 2]
    ) - 0.5 * fVG * (fU[0, 1 : fNBPOINTS + 2] - fU[0, 0 : fNBPOINTS + 1])

    fF_interf[1, :] = 0.5 * (
        fF_cell[1, 0 : fNBPOINTS + 1] + fF_cell[1, 1 : fNBPOINTS + 2]
    ) - 0.5 * lambda_max_i_12 * (fU[1, 1 : fNBPOINTS + 2] - fU[1, 0 : fNBPOINTS + 1])
    
    fF_interf[2, :] = 0.5 * (
        fF_cell[2, 0 : fNBPOINTS + 1] + fF_cell[2, 1 : fNBPOINTS + 2]
    ) - 0.5 * lambda_max_i_12 * (fU[2, 1 : fNBPOINTS + 2] - fU[2, 0 : fNBPOINTS + 1])
    
    fF_interf[3, :] = 0.5 * (
        fF_cell[3, 0 : fNBPOINTS + 1] + fF_cell[3, 1 : fNBPOINTS + 2]
    ) - 0.5 * lambda_max_e_12 * (fU[3, 1 : fNBPOINTS + 2] - fU[3, 0 : fNBPOINTS + 1])


@njit
def ComputeDelta_t(fP, fNBPOINTS, fMi, fCFL, fDelta_x):

    # Compute the max eigenvalue
    lambda_max_i_R = computeMaxEigenVal_i(fP[:, 1 : fNBPOINTS + 2], fMi)
    lambda_max_i_L = computeMaxEigenVal_i(fP[:, 0 : fNBPOINTS + 1], fMi)
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)

    lambda_max_e_R = computeMaxEigenVal_e(fP[:, 1 : fNBPOINTS + 2], fMi)
    lambda_max_e_L = computeMaxEigenVal_e(fP[:, 0 : fNBPOINTS + 1], fMi)
    lambda_max_e_12 = np.maximum(lambda_max_e_L, lambda_max_e_R)

    # Delta_t = fCFL * fDelta_x / (max(max(lambda_max_e_12), max(lambda_max_i_12)))
    Delta_t = fCFL * min(fDelta_x / np.maximum(lambda_max_e_R[:-1],  lambda_max_e_R[:-1]))
    return Delta_t


##########################################################################################
#                                                                                        #
#                               SAVE RESULTS                                             #
#                                                                                        #
##########################################################################################

def MakeTheResultsDir(fResults):
    if not os.path.exists(fResults):
        os.makedirs(fResults)
    ResultsFigs = fResults + "/Figs"
    if not os.path.exists(ResultsFigs):
        os.makedirs(ResultsFigs)
    ResultsData = fResults + "/Data"
    if not os.path.exists(ResultsData):
        os.makedirs(ResultsData)


def SaveVariantData(fResults, fP, fU, fP_LeftGhost, fP_RightGhost, fJ, fEfield,ftime, fi_save):

    ResultsData = fResults + "/Data"

    # Save the data
    filenameTemp = ResultsData + "/MacroscopicVars_" + f"{fi_save:06d}" + ".pkl"
    pickle.dump(
        [ftime, fP, fU, fP_LeftGhost, fP_RightGhost, fJ, fEfield], open(filenameTemp, "wb")
    )  # TODO: Save the current and the electric field


def SaveUnvariantData(fResults, fBarr, fx_mesh, fx_center, falpha_B):

    ResultsData = fResults + "/Data"

    # Save the data
    filenameTemp = ResultsData + "/MacroscopicUnvariants.pkl"
    pickle.dump(
        [fBarr, fx_mesh, fx_center, falpha_B], open(filenameTemp, "wb")
    )  # TODO: Save the current and the electric field


##########################################################################################################
#           Initial field                                                                                #
#           P := Primitive vars [0: ng, 1: ni, 2: ui, 3: Te, 4: ve]                                      #
#           U := Conservative vars [0: rhog, 1: rhoi, 2: rhoiui, 3: 3./2.ni*e*Te]                        #
#                                                                                                        #
##########################################################################################################

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


def main(fconfigfile):

    tttime_start = ttime.time()

    msp = SimuParameters(fconfigfile)

    ### Renames th variables for more clarity

    Resultsdir  = msp.Results
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
    Rext        = msp.Rext
    Te_Cath     = msp.Te_Cath
    CFL         = msp.CFL
    HEATFLUX    = msp.HEATFLUX
    IMPlICIT    = msp.IMPlICIT
    boolCircuit = msp.Circuit
    V           = msp.V
    START_FROM_INPUT  = msp.START_FROM_INPUT
    if START_FROM_INPUT:
        INPUT_DIR  = msp.INPUT_DIR

    MakeTheResultsDir(Resultsdir)

    msp.save_config_file("Configuration.cfg")

    # delete current data in location:
    list_of_res_files = glob.glob(msp.Results+"/Data/Macroscopic*.pkl")
    for filenametemp in list_of_res_files:
        os.remove(filenametemp)
    if len(list_of_res_files) > 0:
        print("Warning: all Macroscopic*.pkl files in the "+Resultsdir+" location were deleted to welcome new data.")


    ##########################################################
    #           Allocation of large vectors                  #
    ##########################################################

    Delta_t = 1.0  # Initialization of Delta_t (do not change)
    # Delta_x = LX / NBPOINTS

    x_mesh, x_center, Delta_x, x_center_extended, Delta_x_extended = msp.return_tiled_domain()
    NBPOINTS = np.shape(x_center)[0]

    Barr = compute_B_array(x_center, msp.BMAX, msp.B0, msp.BLX, msp.LX, LTHR, msp.LB1, msp.LB2)

    if msp.boolInputAnomNu:
        df_nu_anom  = msp.extract_anom_nu()
        nu_anom    = interpolate.interp1d(df_nu_anom['x[cm]']/100, df_nu_anom['nu_anom[Hz]'])(x_center)
        w_ce    = (phy_const.e / phy_const.m_e) * Barr
        alpha_B = nu_anom / w_ce

        """
        PIC_alphaB_plus = alpha_B[np.argwhere(alpha_B > 0.)]
        PIC_x_plus   = x_center[np.argwhere(alpha_B > 0.)]

        PIC_alphB_minus = alpha_B[np.argwhere(alpha_B < 0.)]
        PIC_x_minus   = x_center[np.argwhere(alpha_B < 0.)]

        f, ax = plt.subplots(figsize=(12,7))

        #ax.semilogy([LTHR*100, LTHR*100], [1e-5, 100.], 'k--', label = "channel exit")
        ax.semilogy(PIC_x_plus, PIC_alphaB_plus, 'g+', label = "PIC $\\alpha_B$ PIC, pos. part")
        ax.semilogy(PIC_x_minus, -PIC_alphB_minus, 'g.', label = "PIC $\\alpha_B$ PIC, neg. part")
        #ax.set_ylim([1e-3, 20.])
        ax.set_xlabel("$x$ [cm]")
        ax.set_ylabel("Effective $\\alpha_B$ [rad$^{-1}$]")

        ax.grid()
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        plt.show()

        f, ax = plt.subplots(figsize=(12,7))

        PIC_nu_plus = (alpha_B*w_ce)[np.argwhere(alpha_B > 0.)]
        PIC_x_plus   = x_center[np.argwhere(alpha_B > 0.)]

        PIC_nu_minus = (alpha_B*w_ce)[np.argwhere(alpha_B < 0.)]
        PIC_x_minus   = x_center[np.argwhere(alpha_B < 0.)]        
        #ax.semilogy([LTHR*100, LTHR*100], [1e-5, 100.], 'k--', label = "channel exit")
        ax.semilogy(PIC_x_plus, PIC_nu_plus, 'g+', label = "PIC $\\nu_{anom}$ PIC, pos. part")
        ax.semilogy(PIC_x_minus, -PIC_nu_minus, 'g.', label = "PIC $\\nu_{anom}$ PIC, neg. part")
        #ax.set_ylim([1e-3, 20.])
        ax.set_xlabel("$x$ [cm]")
        ax.set_ylabel("Effective $\\nu_{anom}$ [Hz]")

        ax.grid()
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        plt.show()
        """

    else:
        alpha_B1, alpha_B2  = msp.extract_anomalous()
        alpha_B = compute_alphaB_array(x_center, alpha_B1, alpha_B2, LTHR, msp.NBPOINTS_INIT)

    # Save the Unvariants data as pkl:
    SaveUnvariantData(Resultsdir, Barr, x_mesh, x_center, alpha_B)

    # Allocation of vectors
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

        with open(INPUT_FILE, 'rb') as f:
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
        V0 = msp.V
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

    imposed_Siz = compute_imposed_Siz(x_center, msp.SIZMAX, msp.LSIZ1, msp.LSIZ2)

    # compute mu to check whether or not similitude B scaling works.
    if not msp.boolInputAnomNu:
        mu_eff_arr = compute_mu(P, Barr, ESTAR, wall_inter_type, R1, R2, Mi, x_center, LTHR, KEL, alpha_B)
        print(f"Checking mu value [m^2 s^-1 V^-1]. Bmax = {msp.BMAX*10000:.0f}\talpha_B1 = {alpha_B1:.4e}\talpha_B2 = {alpha_B2:.4e}")
        #print(mu_eff_arr)
        print(f"\tMean value over domain space  {np.mean(mu_eff_arr):.6e}")
        print(f"\tStd deviation:                {np.std(mu_eff_arr):.6e}")

    # We initialize the conservative variables
    PrimToCons(P, U, Mi)


    ##########################################################################################
    #           Loop with Forward Euler                                                      #
    #           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
    ##########################################################################################

    if TIMESCHEME == "Forward Euler":
        J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)
        while time < TIMEFINAL:

            # Save results
            if (iter % SAVERATE) == 0:
                # Compute the electric field.
                Efield = compute_E(P, Barr, ESTAR, wall_inter_type, R1, R2, Mi, x_center, LTHR, KEL, alpha_B)                
                SaveVariantData(Resultsdir, P, U, P_LeftGhost, P_RightGhost, J, Efield, time, i_save)
                i_save += 1
                print(
                    "Iter = ",
                    iter,
                    "\tTime = {:.2f}~µs".format(time / 1e-6),
                    "\tI = {:.4f}~A".format(J),
                    "\tJ = {:.3e} A/m2".format(J/A0),
                )
                # Another way of processing J_d which is equivalent to J_d = I / A0
                #j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
                #mean_j = (1/LX) * np.sum(j_of_x * Delta_x)
                #print("J processed another way = {:.3e} A/m2".format(mean_j)) 
            # Set the boundaries
            SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 1)
            SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, Mi, A0, Te_Cath, J)

            # Compute the Fluxes in the center of the cell
            InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

            # Compute the convective Delta t
            Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), NBPOINTS, Mi, CFL, Delta_x)
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
            Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)
            if HEATFLUX and not IMPlICIT:
                dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)
                Delta_t = min(dt_HF, Delta_t)
            if HEATFLUX and IMPlICIT:
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

    if TIMESCHEME == "TVDRK3":

        while time < TIMEFINAL:
            
            # Save results
            if (iter % SAVERATE) == 0:
                # Compute the electric field.
                Efield = compute_E(P, Barr, ESTAR, wall_inter_type, R1, R2, Mi, x_center, LTHR, KEL, alpha_B)
                SaveVariantData(Resultsdir, P, U, P_LeftGhost, P_RightGhost, J, Efield, time, i_save)
                i_save += 1
                print(
                    "Iter = ",
                    iter,
                    "\tTime = {:.2f}~µs".format(time / 1e-6),
                    "\tI = {:.4f}~A".format(J),
                    "\tJ = {:.3e} A/m2".format(J/A0),
                )
                # Another way of processing J_d which is equivalent to J_d = I / A0
                #j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
                #mean_j = (1/LX) * np.sum(j_of_x * Delta_x)
                #print("J processed another way = {:.3e} A/m2".format(mean_j))
                #CompareIonizationTypes(x_center, P)

            #################################################
            #           FIRST STEP RK3
            #################################################

            # Copy the solution to store it
            U_1[:, :] = U[:, :]
            ConsToPrim(U_1, P_1, Mi, A0, J)
            J_1 = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)
            ConsToPrim(U_1, P_1, Mi, A0, J_1)

            # Set the boundaries
            SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J)
            SetOutlet(P[:, -1], U_RightGhost, P_RightGhost,Mi, A0, Te_Cath, J)
            # Compute the Fluxes in the center of the cell
            InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)
            # Compute the convective Delta t (Only in the first step)
            Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), NBPOINTS, Mi, CFL, Delta_x)
            if iter == 0:
                Delta_t = Delta_t/3
            # Compute the Numerical at the interfaces
            NumericalFlux(
                np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
                np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
                F_cell,
                F_interf,
                NBPOINTS,
                Mi,
                VG
            )

            # Compute the source in the center of the cell
            Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)

            if HEATFLUX and not IMPlICIT:
                dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)
                Delta_t = min(dt_HF, Delta_t)
            # First half step of strang-splitting
            if HEATFLUX and IMPlICIT:
                dt_HF = Delta_t
                P[3, :] = heatFluxImplicit(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x_extended, Delta_t)
                U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]

            
            if iter == 0:
                Delta_t = Delta_t/3


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

            # Compute RLC boolCircuit
            if boolCircuit:
                dJdt = (J - Jm1) / Delta_t

                RHS_Volt0[0] = X_Volt0[1]
                RHS_Volt0[1] = (
                    -1 / (R * C) * X_Volt0[1] - 1.0 / (L * C) * X_Volt0[0] + 1 / C * dJdt
                )
                X_Volt1 = X_Volt0 + Delta_t * RHS_Volt0

            #################################################
            #           SECOND STEP RK3
            #################################################
            # Set the boundaries
            SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 2)
            SetOutlet(P[:, -1], U_RightGhost, P_RightGhost,Mi, A0, Te_Cath, J)

            # Compute the Fluxes in the center of the cell
            InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

            # Compute the Numerical at the interfaces
            NumericalFlux(
                np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
                np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
                F_cell,
                F_interf,
                NBPOINTS,
                Mi,
                VG
            )

            # Compute the source in the center of the cell
            Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)
            if HEATFLUX and not IMPlICIT:
                dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)
               

            # Update the solution
            U[:, :] = (
                0.75 * U_1[:, :]
                + 0.25 * U[:, :]
                + 0.25
                * (
                    -Delta_t
                    / Delta_x
                    * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
                    + Delta_t * S[:, :]
                )
            )

            # Prevent the energy to be strictly negative
            U[3,:] = np.where(U[3,:] >= 0., U[3,:], 0.)

            # Compute the current
            J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)

            # Compute the primitive vars for next step
            ConsToPrim(U, P, Mi, A0, J)

            # Compute RLC Circuit
            if boolCircuit:
                dJdt = (J - Jm1) / Delta_t
                RHS_Volt1[0] = X_Volt1[1]
                RHS_Volt1[1] = (
                    -1 / (R * C) * X_Volt1[1] - 1.0 / (L * C) * X_Volt1[0] + 1 / C * dJdt
                )
                X_Volt2 = 0.75 * X_Volt0 + 0.25 * X_Volt1 + 0.25 * Delta_t * RHS_Volt1

            #################################################
            #           THIRD STEP RK3
            #################################################
            # Set the boundaries
            SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 3)
            SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, Mi, A0, Te_Cath, J)

            # Compute the Fluxes in the center of the cell
            InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

            # Compute the Numerical at the interfaces
            NumericalFlux(
                np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
                np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
                F_cell,
                F_interf,
                NBPOINTS,
                Mi,
                VG
            )
            # Compute the source in the center of the cell
            Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)
            if HEATFLUX and not IMPlICIT:
                dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)

            # Update the solution
            U[:, :] = (
                1.0 / 3.0 * U_1[:, :]
                + 2.0 / 3.0 * U[:, :]
                + 2.0
                / 3.0
                * (
                    -Delta_t
                    / Delta_x
                    * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
                    + Delta_t * S[:, :]
                )
            )
            # Second half step of strang-splitting
            if HEATFLUX and IMPlICIT:
                SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 3)
                SetOutlet(P[:, -1], U_RightGhost, P_RightGhost,Mi, A0, Te_Cath, J)
                P[3, :] = heatFluxImplicit(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x_extended, 0.5*Delta_t)
                U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]

            # Prevent the energy to be strictly negative
            U[3,:] = np.where(U[3,:] >= 0., U[3,:], 0.)

            # Compute the current
            J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)

            # Compute the primitive vars for next step
            ConsToPrim(U, P, Mi, A0, J)

            # Compute RLC Circuit
            if boolCircuit:
                dJdt = (J - Jm1) / Delta_t
                RHS_Volt2[0] = X_Volt2[1]
                RHS_Volt2[1] = (
                    -1 / (R * C) * X_Volt2[1] - 1.0 / (L * C) * X_Volt2[0] + 1 / C * dJdt
                )
                X_Volt3 = (
                    1.0 / 3.0 * X_Volt0
                    + 2.0 / 3.0 * X_Volt2
                    + 2.0 / 3.0 * Delta_t * RHS_Volt2
                )

                # Reinitialize for the Circuit
                Jm1 = J
                X_Volt0[:] = X_Volt3[:]

                # Change the Voltage
                V = V0 - X_Volt0[0]
            time += Delta_t
            if (iter %SAVERATE) ==0:
                filename = Resultsdir + "/time_vec_njit.dat"
                ttime_intermediate = ttime.time()
                a_str = " ".join(map(str, [iter, ttime_intermediate - tttime_start]))
                if iter == 0 and os.path.exists(filename):
                    os.remove(filename)
                    print("File removed:" + filename)
                if os.path.exists(filename):
                    with open(filename, 'a') as file:
                        file.write(a_str)
                        file.write("\n")  # Add a newline at the end (optional)
                else:
                    with open(filename, 'w') as file:
                        file.write(a_str)
                        file.write("\n")  # Add a newline at the end (optional)
            iter += 1
    # Saves the last frame
    SaveVariantData(Resultsdir, P, U, P_LeftGhost, P_RightGhost, J, Efield, time, i_save)
    i_save += 1
    print(
        "Iter = ",
        iter,
        "\tTime = {:.2f}~µs".format(time / 1e-6),
        "\tI = {:.4f}~A".format(J),
        "\tJ = {:.3e} A/m2".format(J/A0),
    )
    # Another way of processing J_d which is equivalent to J_d = I / A0
    #j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
    #mean_j = (1/LX) * np.sum(j_of_x * Delta_x)
    #print("J processed another way = {:.3e} A/m2".format(mean_j))


    ttime_end = ttime.time()
    print("Exec time = {:.2f} s".format(ttime_end - tttime_start))


def main_alphaB_param_study(fconfigFile, falpha_B1_arr, falpha_B2_arr):

    tttime_start = ttime.time()

    msp = SimuParameters(fconfigFile)

    ### Renames th variables for more clarity

    Resultsdir  = msp.Results
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
    LTHR    = msp.LTHR
    KEL     = msp.KEL
    TIMESCHEME  = msp.TIMESCHEME
    TIMEFINAL   = msp.TIMEFINAL
    SAVERATE    = msp.SAVERATE
    boolIonColl = msp.boolIonColl
    boolSizImposed  = msp.boolSizImposed
    Rext        = msp.Rext
    Te_Cath     = msp.Te_Cath
    CFL         = msp.CFL
    HEATFLUX    = msp.HEATFLUX
    IMPlICIT    = msp.IMPlICIT
    boolCircuit = msp.Circuit
    V           = msp.V


    if not os.path.exists(Resultsdir):
        os.makedirs(Resultsdir)
    msp.save_config_file("Configuration.cfg")

    ##########################################################
    #           Allocation of large vectors                  #
    ##########################################################

    Delta_t = 1.0  # Initialization of Delta_t (do not change)
    # Delta_x = LX / NBPOINTS

    x_mesh, x_center, Delta_x, x_center_extended, Delta_x_extended = msp.return_tiled_domain()
    NBPOINTS = np.shape(x_center)[0]

    Barr = compute_B_array(x_center, msp.BMAX, msp.B0, msp.BLX, msp.LX, LTHR, msp.LB1, msp.LB2)

    imposed_Siz     = compute_imposed_Siz(x_center, msp.SIZMAX, msp.LSIZ1, msp.LSIZ2)

    nalphaB1 = falpha_B1_arr.shape[0]
    nalphaB2 = falpha_B2_arr.shape[0]
    alpha_B1_mesh, alpha_B2_mesh = np.meshgrid(falpha_B1_arr, falpha_B2_arr)

    np.savetxt(Resultsdir + '/alphaB1_mesh.csv', alpha_B1_mesh, delimiter='\t')
    np.savetxt(Resultsdir + '/alphaB2_mesh.csv', alpha_B2_mesh, delimiter='\t')

    means_j = np.zeros(alpha_B1_mesh.shape, dtype=float)

    for j in range(nalphaB1):
        for i in range(nalphaB2):
            alpha_B1 = falpha_B1_arr[j]
            alpha_B2 = falpha_B2_arr[i]

            print(f"i={i:3d} aB2={alpha_B2:.4e} ;\tj={j:3d} aB1={alpha_B1:.4e} ;\tRunning...", end='\r')
            subttime_start = ttime.time()

            alpha_B = compute_alphaB_array(x_center, alpha_B1, alpha_B2, LTHR, msp.NBPOINTS_INIT)

            # Allocation of vectors
            P = np.ones((5, NBPOINTS))  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
            U = np.ones((4, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te]
            S = np.ones((4, NBPOINTS))  # Source Term
            F_cell = np.ones((4, NBPOINTS + 2))  # Flux at the cell center. We include the Flux of the Ghost cells
            F_interf = np.ones((4, NBPOINTS + 1))  # Flux at the interface
            U_LeftGhost = np.ones((4, 1))  # Ghost cell on the left
            P_LeftGhost = np.ones((5, 1))  # Ghost cell on the left
            U_RightGhost = np.ones((4, 1))  # Ghost cell on the right
            P_RightGhost = np.ones((5, 1))  # Ghost cell on the right

            if TIMESCHEME == "TVDRK3":
                P_1 = np.ones(
                    (5, NBPOINTS)
                )  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
                U_1 = np.ones((4, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi,

            if boolCircuit:
                R = msp.R
                L = msp.L
                C = msp.C
                V0 = msp.V
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
                Jm1 = 0.0

            time = 0.0
            iter = 0
            J = 0.0  # Initial Current

            # We initialize the primitive variables
            ng_anode = MDOT / (Mi* A0 * VG)  # Initial propellant density ng at the anode location
            P[1, :] *= NI0  # Initial ni
            P[2, :] *= 0.0  # Initial vi
            P[3, :] *= TE0  # Initial Te
            # P[3, :]  = SmoothInitialTemperature(P[3, :], Te_Cath)
            P[4, :] *= P[2, :] - J / (A0 * phy_const.e * P[1, :])  # Initial Ve
            #P[0,:] = InitNeutralDensity(x_center, ng_anode, VG, P, IonizationConfig['Type'], SIZMAX, LSIZ1, LSIZ2) # initialize n_g in the space so that it is cst in time if there is no wall recombination.
            ### Warning, in the code currently, neutrals dyanmic is canceled.
            P[0,:] = ng_anode

            # We initialize the conservative variables
            PrimToCons(P, U, Mi)


            ##########################################################################################
            #           Loop with Forward Euler                                                      #
            #           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
            ##########################################################################################

            if TIMESCHEME == "Forward Euler":
                J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)
                while time < TIMEFINAL:

                    # Set the boundaries
                    SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 1)
                    SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, Mi, A0, Te_Cath, J)

                    # Compute the Fluxes in the center of the cell
                    InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

                    # Compute the convective Delta t
                    Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), NBPOINTS, Mi, CFL, Delta_x)
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
                    Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)
                    if HEATFLUX and not IMPlICIT:
                        dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)
                        Delta_t = min(dt_HF, Delta_t)
                    if HEATFLUX and IMPlICIT:
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

            if TIMESCHEME == "TVDRK3":

                while time < TIMEFINAL:

                    #################################################
                    #           FIRST STEP RK3
                    #################################################

                    # Copy the solution to store it
                    U_1[:, :] = U[:, :]
                    ConsToPrim(U_1, P_1, Mi, A0, J)
                    J_1 = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)
                    ConsToPrim(U_1, P_1, Mi, A0, J_1)

                    # Set the boundaries
                    SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J)
                    SetOutlet(P[:, -1], U_RightGhost, P_RightGhost,Mi, A0, Te_Cath, J)
                    # Compute the Fluxes in the center of the cell
                    InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)
                    # Compute the convective Delta t (Only in the first step)
                    Delta_t = ComputeDelta_t(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), NBPOINTS, Mi, CFL, Delta_x)
                    if iter == 0:
                        Delta_t = Delta_t/3
                    # Compute the Numerical at the interfaces
                    NumericalFlux(
                        np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
                        np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
                        F_cell,
                        F_interf,
                        NBPOINTS,
                        Mi,
                        VG
                    )

                    # Compute the source in the center of the cell
                    Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)

                    if HEATFLUX and not IMPlICIT:
                        dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)
                        Delta_t = min(dt_HF, Delta_t)
                    # First half step of strang-splitting
                    if HEATFLUX and IMPlICIT:
                        dt_HF = Delta_t
                        P[3, :] = heatFluxImplicit(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x_extended, Delta_t)
                        U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]

                    
                    if iter == 0:
                        Delta_t = Delta_t/3


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

                    # Compute RLC Circuit
                    if boolCircuit:
                        dJdt = (J - Jm1) / Delta_t

                        RHS_Volt0[0] = X_Volt0[1]
                        RHS_Volt0[1] = (
                            -1 / (R * C) * X_Volt0[1] - 1.0 / (L * C) * X_Volt0[0] + 1 / C * dJdt
                        )
                        X_Volt1 = X_Volt0 + Delta_t * RHS_Volt0

                    #################################################
                    #           SECOND STEP RK3
                    #################################################
                    # Set the boundaries
                    SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 2)
                    SetOutlet(P[:, -1], U_RightGhost, P_RightGhost,Mi, A0, Te_Cath, J)

                    # Compute the Fluxes in the center of the cell
                    InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

                    # Compute the Numerical at the interfaces
                    NumericalFlux(
                        np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
                        np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
                        F_cell,
                        F_interf,
                        NBPOINTS,
                        Mi,
                        VG
                    )

                    # Compute the source in the center of the cell
                    Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)
                    if HEATFLUX and not IMPlICIT:
                        dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)
                    

                    # Update the solution
                    U[:, :] = (
                        0.75 * U_1[:, :]
                        + 0.25 * U[:, :]
                        + 0.25
                        * (
                            -Delta_t
                            / Delta_x
                            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
                            + Delta_t * S[:, :]
                        )
                    )

                    # Prevent the energy to be strictly negative
                    U[3,:] = np.where(U[3,:] >= 0., U[3,:], 0.)

                    # Compute the current
                    J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)

                    # Compute the primitive vars for next step
                    ConsToPrim(U, P, Mi, A0, J)

                    # Compute RLC Circuit
                    if boolCircuit:
                        dJdt = (J - Jm1) / Delta_t
                        RHS_Volt1[0] = X_Volt1[1]
                        RHS_Volt1[1] = (
                            -1 / (R * C) * X_Volt1[1] - 1.0 / (L * C) * X_Volt1[0] + 1 / C * dJdt
                        )
                        X_Volt2 = 0.75 * X_Volt0 + 0.25 * X_Volt1 + 0.25 * Delta_t * RHS_Volt1

                    #################################################
                    #           THIRD STEP RK3
                    #################################################
                    # Set the boundaries
                    SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 3)
                    SetOutlet(P[:, -1], U_RightGhost, P_RightGhost, Mi, A0, Te_Cath, J)

                    # Compute the Fluxes in the center of the cell
                    InviscidFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), F_cell, VG, Mi)

                    # Compute the Numerical at the interfaces
                    NumericalFlux(
                        np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1),
                        np.concatenate([U_LeftGhost, U, U_RightGhost], axis=1),
                        F_cell,
                        F_interf,
                        NBPOINTS,
                        Mi,
                        VG
                    )
                    # Compute the source in the center of the cell
                    Source(P, S, Barr, boolSizImposed, boolIonColl, wall_inter_type, x_center, imposed_Siz, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, VG, Delta_x)
                    if HEATFLUX and not IMPlICIT:
                        dt_HF = heatFlux(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), S,  np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x)

                    # Update the solution
                    U[:, :] = (
                        1.0 / 3.0 * U_1[:, :]
                        + 2.0 / 3.0 * U[:, :]
                        + 2.0
                        / 3.0
                        * (
                            -Delta_t
                            / Delta_x
                            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
                            + Delta_t * S[:, :]
                        )
                    )
                    # Second half step of strang-splitting
                    if HEATFLUX and IMPlICIT:
                        SetInlet(P[:, 0], U_LeftGhost, P_LeftGhost, Mi, boolSizImposed, MDOT, A0, VG, J, 3)
                        SetOutlet(P[:, -1], U_RightGhost, P_RightGhost,Mi, A0, Te_Cath, J)
                        P[3, :] = heatFluxImplicit(np.concatenate([P_LeftGhost, P, P_RightGhost], axis=1), np.concatenate([[Barr[0]], Barr, [Barr[-1]]]), wall_inter_type, x_center_extended, ESTAR, Mi, R1, R2, LTHR, KEL, np.concatenate([[alpha_B[0]], alpha_B, [alpha_B[-1]]]), Delta_x_extended, 0.5*Delta_t)
                        U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]

                    # Prevent the energy to be strictly negative
                    U[3,:] = np.where(U[3,:] >= 0., U[3,:], 0.)

                    # Compute the current
                    J = compute_I(P, V, Barr, wall_inter_type, x_center, ESTAR, Mi, R1, R2, LTHR, KEL, alpha_B, Delta_x, A0, Rext)

                    # Compute the primitive vars for next step
                    ConsToPrim(U, P, Mi, A0, J)

                    # Compute RLC Circuit
                    if boolCircuit:
                        dJdt = (J - Jm1) / Delta_t
                        RHS_Volt2[0] = X_Volt2[1]
                        RHS_Volt2[1] = (
                            -1 / (R * C) * X_Volt2[1] - 1.0 / (L * C) * X_Volt2[0] + 1 / C * dJdt
                        )
                        X_Volt3 = (
                            1.0 / 3.0 * X_Volt0
                            + 2.0 / 3.0 * X_Volt2
                            + 2.0 / 3.0 * Delta_t * RHS_Volt2
                        )

                        # Reinitialize for the Circuit
                        Jm1 = J
                        X_Volt0[:] = X_Volt3[:]

                        # Change the Voltage
                        V = V0 - X_Volt0[0]
                    time += Delta_t

                    iter += 1

            j_of_x = P[1,:]*phy_const.e*(P[2,:] - P[4,:])
            means_j[i, j] = (1/msp.LX) * np.sum(j_of_x * Delta_x)
            subttime_end = ttime.time()
            subttime_delta = int(subttime_end - subttime_start)
            print(f"i={i:3d} aB2={alpha_B2:.4e} ;\tj={j:3d} aB1={alpha_B1:.4e} ;\tJ={means_j[i, j]:.3e} A.m^{-2}\tt_comput={subttime_delta} s")

    np.savetxt(Resultsdir + '/mean_j_array.csv', means_j, delimiter='\t')

    ttime_end = ttime.time()
    print("Exec time = {:.2f} s".format(ttime_end - tttime_start))


if __name__ == '__main__':

    main(sys.argv[1])

    """    nalpha = 20
    alpha_B1_arr = np.linspace(-4, -1, nalpha) # range of anomalous coeffs. in the channel
    alpha_B2_arr = np.linspace(-4, -1, nalpha) # range of anomalous coeffs. in the channel
    alpha_B1_arr = 10**alpha_B1_arr
    alpha_B2_arr = 10**alpha_B2_arr
    
    #alpha_B1_arr = np.array([3.7927e-3])
    #alpha_B2_arr = np.array([7.8476e-3, 1.1288e-2])
    #main_alphaB_param_study('config_alphaB_prm_study.ini', alpha_B1_arr, alpha_B2_arr)
    """