import numpy as np
import scipy.constants as phy_const
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import configparser
import sys
import scipy as sc
from numba import njit
import time as ttime

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

tttime_start = ttime.time()

configFile = sys.argv[1]
config = configparser.ConfigParser()
config.read(configFile)

physicalParameters = config["Physical Parameters"]

Gas_type = physicalParameters["Gas"]
VG = float(physicalParameters["Gas velocity"])  # Gas velocity
m = phy_const.m_e  # Electron mass
R1 = float(physicalParameters["Inner radius"])  # Inner radius of the thruster
R2 = float(physicalParameters["Outer radius"])  # Outer radius of the thruster
A0 = np.pi * (R2**2 - R1**2)  # Area of the thruster
LENGTH = float(physicalParameters["Length of axis"])  # length of Axis of the simulation
L0 = float(
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
    physicalParameters["Temperature Cathode"]
)  # Electron temperature at the cathode
Rext = float(physicalParameters["Ballast resistor"])  # Resistor of the ballast
V = float(physicalParameters["Voltage"])  # Potential difference
Circuit = bool(
    config.getboolean("Physical Parameters", "Circuit", fallback=False)
)  # RLC Circuit
Estar = float(physicalParameters["Crossover energy"])  # Crossover energy

# Ion mass
if Gas_type == 'Xenon' :
    M = 131.293 * phy_const.m_u
elif Gas_type == 'Krypton' :
    M = 83.798 * phy_const.m_u

# Magnetic field configuration
MagneticFieldConfig = config["Magnetic field configuration"]

if MagneticFieldConfig["Type"] == "Default":
    print(MagneticFieldConfig["Type"] + " Magnetic Field")

    Bmax = float(MagneticFieldConfig["Max B-field"])  # Max Mag field
    LB1 = float(MagneticFieldConfig["Length B-field 1"])  # Length for magnetic field
    LB2 = float(MagneticFieldConfig["Length B-field 2"])  # Length for magnetic field
    saveBField = bool(MagneticFieldConfig["Save B-field"])

##########################################################
#           NUMERICAL PARAMETERS
##########################################################
NumericsConfig = config["Numerical Parameters"]

NBPOINTS = int(NumericsConfig["Number of points"])  # Number of cells
SAVERATE = int(NumericsConfig["Save rate"])  # Rate at which we store the data
CFL = float(NumericsConfig["CFL"])  # Nondimensional size of the time step
TIMEFINAL = float(NumericsConfig["Final time"])  # Last time of simulation
Results = NumericsConfig["Result dir"]  # Name of result directory
TIMESCHEME = NumericsConfig["Time integration"]  # Time integration scheme
AlphaModel = NumericsConfig['Anomalous transport model']        # Model chosen for the anomalous transport

Safrandata = config["Safran data"]

Curr_ref  = float(Safrandata['Mean current'])
Thrust_ref = float(Safrandata['Thrust'])
ISP_ref  = int(Safrandata['ISP'])
Eta_ref = int(Safrandata['Efficiency'])

if not os.path.exists(Results):
    os.makedirs(Results)
with open(Results + "/Configuration.cfg", "w") as configfile:
    config.write(configfile)

##########################################################
#           Allocation of large vectors                  #
##########################################################

Delta_t = 1.0  # Initialization of Delta_t (do not change)
Delta_x = LENGTH / NBPOINTS

x_mesh = np.linspace(0, LENGTH, NBPOINTS + 1)  # Mesh in the interface
x_center = np.linspace(Delta_x, LENGTH - Delta_x, NBPOINTS)  # Mesh in the center of cell
B0 = Bmax * np.exp(-(((x_center - L0) / LB1) ** 2.0))  # Magnetic field within the thruster
B0 = np.where(x_center < L0, B0, Bmax * np.exp(-(((x_center - L0) / LB2) ** 2.0)))  # Magnetic field outside the thruster



# Allocation of vectors
P = np.ones((5, NBPOINTS))  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
U = np.ones((4, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi, 3/2 ne*e*Te]
S = np.ones((4, NBPOINTS))  # Source Term
F_cell = np.ones((4, NBPOINTS + 2))  # Flux at the cell center. We include the Flux of the Ghost cells
F_interf = np.ones((4, NBPOINTS + 1))  # Flux at the interface
U_Inlet = np.ones((4, 1))  # Ghost cell on the left
P_Inlet = np.ones((5, 1))  # Ghost cell on the left
U_Outlet = np.ones((4, 1))  # Ghost cell on the right
P_Outlet = np.ones((5, 1))  # Ghost cell on the right
if TIMESCHEME == "TVDRK3":
    P_1 = np.ones(
        (5, NBPOINTS)
    )  # Primitive vars P = [ng, ni, ui,  Te, ve] TODO: maybe add , E
    U_1 = np.ones((4, NBPOINTS))  # Conservative vars U = [rhog, rhoi, rhoUi,

if Circuit:
    R = float(physicalParameters["R"])
    L = float(physicalParameters["L"])
    C = float(physicalParameters["C"])
    V0 = V
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
    J0 = 0.0

##########################################################
#     Definition of the anomalous transport parameter    #
##########################################################
if AlphaModel == 'Step' :
    alpha_B  = np.ones(NBPOINTS)*alpha_B1                            # Anomalous transport coefficient inside the thruster
    alpha_B  = np.where(x_center < L0, alpha_B, alpha_B2)         # Anomalous transport coefficient in the plume
elif AlphaModel == 'Linear' :
    alpha_B = np.ones(NBPOINTS)
    for i in range(len(x_center)) :
        alpha_B[i] = (((alpha_B2-alpha_B1)/LENGTH)*x_center[i]+alpha_B1)
elif AlphaModel == 'Capelli' :                     # We treat other cases lowers, as they deal with nu_an instead of a vector alpha_B
    pass
elif AlphaModel == 'Cafleur' :
    pass
elif AlphaModel == 'Chodura' :
    pass
elif AlphaModel == 'Data-driven' :
     pass
##########################################################
#           Formulas defining our model                  #
##########################################################

#For Kel
if Gas_type =="Xenon":
    coll = """  0.000000e+0	3.447540e-19
    1.000000e-4	3.447540e-19
    1.000000e-3	3.310000e-19
    3.000000e-3	3.000000e-19
    5.000000e-3	2.790000e-19
    7.000000e-3	2.620000e-19
    1.000000e-2	2.420000e-19
    1.500000e-2	2.160000e-19
    2.000000e-2	1.950000e-19
    3.514000e-2	1.493300e-19
    7.152000e-2	9.140250e-20
    1.091700e-1	6.042430e-20
    1.481500e-1	4.028030e-20
    1.885000e-1	2.715650e-20
    2.302700e-1	1.834800e-20
    2.735000e-1	1.198630e-20
    3.182600e-1	7.393130e-21
    3.645800e-1	4.207040e-21
    4.125400e-1	2.231110e-21
    4.621800e-1	1.329620e-21
    5.135600e-1	1.009430e-21
    5.667500e-1	1.136880e-21
    6.218100e-1	1.517440e-21
    6.788000e-1	2.097760e-21
    7.378000e-1	2.824930e-21
    7.988700e-1	3.731510e-21
    8.620900e-1	4.698550e-21
    9.275200e-1	5.869940e-21
    9.952600e-1	7.292590e-21
    1.065380e+0	8.902680e-21
    1.137960e+0	1.079130e-20
    1.213090e+0	1.288960e-20
    1.290870e+0	1.468070e-20
    1.371370e+0	1.666340e-20
    1.454710e+0	1.876390e-20
    1.540970e+0	2.101710e-20
    1.630270e+0	2.348180e-20
    1.722700e+0	2.610590e-20
    1.818380e+0	2.872720e-20
    1.917430e+0	3.155480e-20
    2.019950e+0	3.450950e-20
    2.126080e+0	3.726160e-20
    2.235940e+0	4.018350e-20
    2.349650e+0	4.328440e-20
    2.467370e+0	4.657400e-20
    2.589220e+0	5.022760e-20
    2.715350e+0	5.417950e-20
    2.845920e+0	5.838640e-20
    2.981070e+0	6.286320e-20
    3.120980e+0	6.735680e-20
    3.265800e+0	7.207120e-20
    3.415700e+0	7.686130e-20
    3.570880e+0	8.185310e-20
    3.731510e+0	8.679890e-20
    3.897790e+0	9.191850e-20
    4.069910e+0	9.716020e-20
    4.248070e+0	1.024490e-19
    4.432500e+0	1.079660e-19
    4.623410e+0	1.136740e-19
    4.821030e+0	1.196760e-19
    5.025600e+0	1.262940e-19
    5.237350e+0	1.329340e-19
    5.456540e+0	1.384150e-19
    5.683440e+0	1.440870e-19
    5.918310e+0	1.499580e-19
    6.161430e+0	1.552470e-19
    6.413100e+0	1.602740e-19
    6.673610e+0	1.654880e-19
    6.943280e+0	1.708730e-19
    7.222430e+0	1.751390e-19
    7.511380e+0	1.791160e-19
    7.810490e+0	1.821270e-19
    8.120110e+0	1.843770e-19
    8.772370e+0	1.863430e-19
    9.471290e+0	1.840690e-19
    9.839270e+0	1.819100e-19
    1.022018e+1	1.789030e-19
    1.061449e+1	1.753150e-19
    1.102264e+1	1.717060e-19
    1.144515e+1	1.664210e-19
    1.188250e+1	1.613140e-19
    1.233521e+1	1.557630e-19
    1.280384e+1	1.502080e-19
    1.328894e+1	1.443340e-19
    1.379108e+1	1.383530e-19
    1.431087e+1	1.323970e-19
    1.484893e+1	1.265560e-19
    1.540590e+1	1.203260e-19
    1.598244e+1	1.141790e-19
    1.657924e+1	1.081260e-19
    1.719701e+1	1.023990e-19
    1.783649e+1	9.698660e-20
    1.849845e+1	9.186940e-20
    1.918366e+1	8.703110e-20
    1.989296e+1	8.245560e-20
    2.062719e+1	7.757970e-20
    2.138721e+1	7.291040e-20
    2.217395e+1	6.852900e-20
    2.298833e+1	6.441710e-20
    2.383133e+1	6.055760e-20
    2.470396e+1	5.693440e-20
    2.560725e+1	5.353260e-20
    2.654229e+1	5.033820e-20
    2.751018e+1	4.733830e-20
    2.851209e+1	4.452050e-20
    2.954921e+1	4.187360e-20
    3.062278e+1	3.936340e-20
    3.173407e+1	3.699010e-20
    3.288442e+1	3.476210e-20
    3.407519e+1	3.267050e-20
    3.530781e+1	3.070670e-20
    3.658374e+1	2.886270e-20
    3.790451e+1	2.713100e-20
    3.927170e+1	2.550460e-20
    4.068694e+1	2.401650e-20
    4.215191e+1	2.265660e-20
    4.366836e+1	2.137470e-20
    4.523810e+1	2.016620e-20
    4.686301e+1	1.902690e-20
    4.854502e+1	1.795280e-20
    5.028614e+1	1.700460e-20
    5.208844e+1	1.642730e-20
    5.395409e+1	1.587000e-20
    5.588529e+1	1.533190e-20
    5.788437e+1	1.481240e-20
    5.995369e+1	1.431080e-20
    6.431306e+1	1.404620e-20
    6.898420e+1	1.379440e-20
    7.398942e+1	1.354730e-20
    7.935261e+1	1.340050e-20
    8.812509e+1	1.321740e-20
    9.785531e+1	1.303700e-20
    1.013293e+2	1.284520e-20
    1.049254e+2	1.244530e-20
    1.086478e+2	1.205800e-20
    1.125011e+2	1.168280e-20
    1.164898e+2	1.131950e-20
    1.206186e+2	1.096750e-20
    1.248925e+2	1.062660e-20
    1.293167e+2	1.029630e-20
    1.338963e+2	9.976410e-21
    1.386368e+2	9.666530e-21
    1.435440e+2	9.366350e-21
    1.486236e+2	9.075560e-21
    1.538817e+2	8.906350e-21
    1.593245e+2	8.780500e-21
    1.649587e+2	8.656450e-21
    1.707908e+2	8.534190e-21
    1.768279e+2	8.413670e-21
    1.830772e+2	8.294880e-21
    1.895461e+2	8.177790e-21
    1.962423e+2	8.062370e-21
    2.031738e+2	7.930980e-21
    2.103489e+2	7.780940e-21
    2.177762e+2	7.633760e-21
    2.254644e+2	7.489380e-21
    2.334229e+2	7.347760e-21
    2.416610e+2	7.208830e-21
    2.501886e+2	7.072550e-21
    2.590160e+2	6.938870e-21
    2.681535e+2	6.807730e-21
    2.776121e+2	6.679080e-21
    2.874031e+2	6.552890e-21
    2.975383e+2	6.429090e-21
    3.080295e+2	6.300940e-21
    3.188895e+2	6.173310e-21
    3.301311e+2	6.048290e-21
    3.417678e+2	5.925800e-21
    3.538134e+2	5.805820e-21
    3.662823e+2	5.688270e-21
    3.791894e+2	5.573110e-21
    3.925501e+2	5.460290e-21
    4.063803e+2	5.330630e-21
    4.206965e+2	5.181950e-21
    4.355158e+2	5.037420e-21
    4.508559e+2	4.896940e-21
    4.667351e+2	4.760390e-21
    4.831724e+2	4.627650e-21
    5.001872e+2	4.499410e-21
    5.178000e+2	4.445230e-21
    5.360318e+2	4.391710e-21
    5.549043e+2	4.338840e-21
    5.744399e+2	4.286600e-21
    5.946621e+2	4.235000e-21
    6.155950e+2	4.184020e-21
    6.372635e+2	4.133660e-21
    6.596934e+2	4.083910e-21
    6.829117e+2	4.034760e-21
    7.069458e+2	3.968270e-21
    7.318245e+2	3.859100e-21
    7.575776e+2	3.752940e-21
    7.842356e+2	3.649700e-21
    8.118305e+2	3.549310e-21
    8.403951e+2	3.451680e-21
    8.699636e+2	3.356740e-21
    9.005711e+2	3.264420e-21
    9.322543e+2	3.174630e-21
    9.650509e+2	3.087330e-21"""
elif Gas_type=="Krypton":
     coll = """   0.000000e+0	3.447540e-19
     1.000000e-4	3.447540e-19
     1.000000e-3	3.310000e-19
     3.000000e-3	3.000000e-19
     5.000000e-3	2.790000e-19
     7.000000e-3	2.620000e-19
     1.000000e-2	2.420000e-19
     1.500000e-2	2.160000e-19
     2.000000e-2	1.950000e-19
     3.514000e-2	1.493300e-19
     7.152000e-2	9.140250e-20
     1.091700e-1	6.042430e-20
     1.481500e-1	4.028030e-20
     1.885000e-1	2.715650e-20
     2.302700e-1	1.834800e-20
     2.735000e-1	1.198630e-20
     3.182600e-1	7.393130e-21
     3.645800e-1	4.207040e-21
     4.125400e-1	2.231110e-21
     4.621800e-1	1.329620e-21
     5.135600e-1	1.009430e-21
     5.667500e-1	1.136880e-21
     6.218100e-1	1.517440e-21
     6.788000e-1	2.097760e-21
     7.378000e-1	2.824930e-21
     7.988700e-1	3.731510e-21
     8.620900e-1	4.698550e-21
     9.275200e-1	5.869940e-21
     9.952600e-1	7.292590e-21
     1.065380e+0	8.902680e-21
     1.137960e+0	1.079130e-20
     1.213090e+0	1.288960e-20
     1.290870e+0	1.468070e-20
     1.371370e+0	1.666340e-20
     1.454710e+0	1.876390e-20
     1.540970e+0	2.101710e-20
     1.630270e+0	2.348180e-20
     1.722700e+0	2.610590e-20
     1.818380e+0	2.872720e-20
     1.917430e+0	3.155480e-20
     2.019950e+0	3.450950e-20
     2.126080e+0	3.726160e-20
     2.235940e+0	4.018350e-20
     2.349650e+0	4.328440e-20
     2.467370e+0	4.657400e-20
     2.589220e+0	5.022760e-20
     2.715350e+0	5.417950e-20
     2.845920e+0	5.838640e-20
     2.981070e+0	6.286320e-20
     3.120980e+0	6.735680e-20
     3.265800e+0	7.207120e-20
     3.415700e+0	7.686130e-20
     3.570880e+0	8.185310e-20
     3.731510e+0	8.679890e-20
     3.897790e+0	9.191850e-20
     4.069910e+0	9.716020e-20
     4.248070e+0	1.024490e-19
     4.432500e+0	1.079660e-19
     4.623410e+0	1.136740e-19
     4.821030e+0	1.196760e-19
     5.025600e+0	1.262940e-19
     5.237350e+0	1.329340e-19
     5.456540e+0	1.384150e-19
     5.683440e+0	1.440870e-19
     5.918310e+0	1.499580e-19
     6.161430e+0	1.552470e-19
     6.413100e+0	1.602740e-19
     6.673610e+0	1.654880e-19
     6.943280e+0	1.708730e-19
     7.222430e+0	1.751390e-19
     7.511380e+0	1.791160e-19
     7.810490e+0	1.821270e-19
     8.120110e+0	1.843770e-19
     8.772370e+0	1.863430e-19
     9.471290e+0	1.840690e-19
     9.839270e+0	1.819100e-19
     1.022018e+1	1.789030e-19
     1.061449e+1	1.753150e-19
     1.102264e+1	1.717060e-19
     1.144515e+1	1.664210e-19
     1.188250e+1	1.613140e-19
     1.233521e+1	1.557630e-19
     1.280384e+1	1.502080e-19
     1.328894e+1	1.443340e-19
     1.379108e+1	1.383530e-19
     1.431087e+1	1.323970e-19
     1.484893e+1	1.265560e-19
     1.540590e+1	1.203260e-19
     1.598244e+1	1.141790e-19
     1.657924e+1	1.081260e-19
     1.719701e+1	1.023990e-19
     1.783649e+1	9.698660e-20
     1.849845e+1	9.186940e-20
     1.918366e+1	8.703110e-20
     1.989296e+1	8.245560e-20
     2.062719e+1	7.757970e-20
     2.138721e+1	7.291040e-20
     2.217395e+1	6.852900e-20
     2.298833e+1	6.441710e-20
     2.383133e+1	6.055760e-20
     2.470396e+1	5.693440e-20
     2.560725e+1	5.353260e-20
     2.654229e+1	5.033820e-20
     2.751018e+1	4.733830e-20
     2.851209e+1	4.452050e-20
     2.954921e+1	4.187360e-20
     3.062278e+1	3.936340e-20
     3.173407e+1	3.699010e-20
     3.288442e+1	3.476210e-20
     3.407519e+1	3.267050e-20
     3.530781e+1	3.070670e-20
     3.658374e+1	2.886270e-20
     3.790451e+1	2.713100e-20
     3.927170e+1	2.550460e-20
     4.068694e+1	2.401650e-20
     4.215191e+1	2.265660e-20
     4.366836e+1	2.137470e-20
     4.523810e+1	2.016620e-20
     4.686301e+1	1.902690e-20
     4.854502e+1	1.795280e-20
     5.028614e+1	1.700460e-20
     5.208844e+1	1.642730e-20
     5.395409e+1	1.587000e-20
     5.588529e+1	1.533190e-20
     5.788437e+1	1.481240e-20
     5.995369e+1	1.431080e-20
     6.431306e+1	1.404620e-20
     6.898420e+1	1.379440e-20
     7.398942e+1	1.354730e-20
     7.935261e+1	1.340050e-20
     8.812509e+1	1.321740e-20
     9.785531e+1	1.303700e-20
     1.013293e+2	1.284520e-20
     1.049254e+2	1.244530e-20
     1.086478e+2	1.205800e-20
     1.125011e+2	1.168280e-20
     1.164898e+2	1.131950e-20
     1.206186e+2	1.096750e-20
     1.248925e+2	1.062660e-20
     1.293167e+2	1.029630e-20
     1.338963e+2	9.976410e-21
     1.386368e+2	9.666530e-21
     1.435440e+2	9.366350e-21
     1.486236e+2	9.075560e-21
     1.538817e+2	8.906350e-21
     1.593245e+2	8.780500e-21
     1.649587e+2	8.656450e-21
     1.707908e+2	8.534190e-21
     1.768279e+2	8.413670e-21
     1.830772e+2	8.294880e-21
     1.895461e+2	8.177790e-21
     1.962423e+2	8.062370e-21
     2.031738e+2	7.930980e-21
     2.103489e+2	7.780940e-21
     2.177762e+2	7.633760e-21
     2.254644e+2	7.489380e-21
     2.334229e+2	7.347760e-21
     2.416610e+2	7.208830e-21
     2.501886e+2	7.072550e-21
     2.590160e+2	6.938870e-21
     2.681535e+2	6.807730e-21
     2.776121e+2	6.679080e-21
     2.874031e+2	6.552890e-21
     2.975383e+2	6.429090e-21
     3.080295e+2	6.300940e-21
     3.188895e+2	6.173310e-21
     3.301311e+2	6.048290e-21
     3.417678e+2	5.925800e-21
     3.538134e+2	5.805820e-21
     3.662823e+2	5.688270e-21
     3.791894e+2	5.573110e-21
     3.925501e+2	5.460290e-21
     4.063803e+2	5.330630e-21
     4.206965e+2	5.181950e-21
     4.355158e+2	5.037420e-21
     4.508559e+2	4.896940e-21
     4.667351e+2	4.760390e-21
     4.831724e+2	4.627650e-21
     5.001872e+2	4.499410e-21
     5.178000e+2	4.445230e-21
     5.360318e+2	4.391710e-21
     5.549043e+2	4.338840e-21
     5.744399e+2	4.286600e-21
     5.946621e+2	4.235000e-21
     6.155950e+2	4.184020e-21
     6.372635e+2	4.133660e-21
     6.596934e+2	4.083910e-21
     6.829117e+2	4.034760e-21
     7.069458e+2	3.968270e-21
     7.318245e+2	3.859100e-21
     7.575776e+2	3.752940e-21
     7.842356e+2	3.649700e-21
     8.118305e+2	3.549310e-21
     8.403951e+2	3.451680e-21
     8.699636e+2	3.356740e-21
     9.005711e+2	3.264420e-21
     9.322543e+2	3.174630e-21
     9.650509e+2	3.087330e-21 """
    
coll = np.array([i.split() for i in coll.split("\n")], dtype=float)                             
xarray = np.logspace(-4, 3, 1000)
coll_interp = np.interp(xarray, coll[:, 0], coll[:, 1])


def simpson(f,vec):
    simps = []
    for i in range (len(vec)):
        inter=( ((f[i]-f[i-1])/(vec[i]-vec[i-1]))*(vec[i]-vec[i-1])/2 + f[i-1])
        simps.append( ((vec[i+1]-vec[i])/6) * (f[i] + 4*inter + f[i+1]) )
    return simps

def max_distrib(temperature, energies):
    f_d = 2 * (energies/np.pi/temperature)**(.5)/temperature * np.exp(-energies/temperature)
    return(f_d)

def integrate_fct(E_vec, sigma_iz, f_distrib):
    f_distrib = f_distrib / phy_const.elementary_charge
    E_vec = E_vec * phy_const.elementary_charge
    integrando = sigma_iz * E_vec**.5 * f_distrib
    Kz = (2 / phy_const.electron_mass)**0.5 * sc.integrate.simpson(integrando,x=E_vec)
    return Kz

Tvec = []
int_result = []
for ind in np.linspace(0.001,50, 1000):
    distr = max_distrib(ind, xarray)
    int_result.append(integrate_fct(xarray[1:], coll_interp[1:], distr[1:]))
    Tvec.append(ind)
Tvec = np.asarray(Tvec)
int_result = np.asarray(int_result)

@njit
def Kel(T_input, Tvec, int_result):
    Kel_fct = np.zeros(len(T_input))
    for ind_t, T_ind in enumerate(T_input) :
        i = 0
        while T_ind > Tvec[i] :
            i=i+1
        Kel_fct[ind_t] = ((int_result[i]-int_result[i-1])/(Tvec[i]-Tvec[i-1]))*(T_ind-Tvec[i-1]) + int_result[i-1]
    return Kel_fct
    
#Other
@njit
def gradient(y, d):
    dp_dz=np.empty_like(y)
    dp_dz[1:-1] =(y[2:] -y[:-2]) /(2*d)
    dp_dz[0] =2*dp_dz[1] -dp_dz[2]
    dp_dz[-1] =2*dp_dz[-2] -dp_dz[-3]
    return dp_dz
    
@njit
def PrimToCons(P, U):
    U[0, :] = P[0, :] * M  # rhog
    U[1, :] = P[1, :] * M  # rhoi
    U[2, :] = P[2, :] * P[1, :] * M  # rhoiUi
    U[3, :] = 3.0 / 2.0 * P[1, :] * phy_const.e * P[3, :]  # 3/2*ni*e*Te


@njit
def ConsToPrim(U, P, J=0.0):
    P[0, :] = U[0, :] / M  # ng
    P[1, :] = U[1, :] / M  # ni
    P[2, :] = U[2, :] / U[1, :]  # Ui = rhoUi/rhoi
    P[3, :] = 2.0 / 3.0 * U[3, :] / (phy_const.e * P[1, :])  # Te
    P[4, :] = P[2, :] - J / (A0 * phy_const.e * P[1, :])  # ve


@njit
def InviscidFlux(P, F):
    F[0, :] = P[0, :] * VG * M  # rho_g*v_g
    F[1, :] = P[1, :] * P[2, :] * M  # rho_i*v_i
    F[2, :] = (
        M * P[1, :] * P[2, :] * P[2, :] + P[1, :] * phy_const.e * P[3, :]
    )  # M*n_i*v_i**2 + p_e
    F[3, :] = 5.0 / 2.0 * P[1, :] * phy_const.e * P[3, :] * P[4, :]  # 5/2n_i*e*T_e*v_e


@njit
def Source(P, S):
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0, :]
    ni = P[1, :]
    ui = P[2, :]
    Te = P[3, :]
    ve = P[4, :]

    energy = 3.0 / 2.0 * ni * phy_const.e * Te  # Electron internal energy
    # Gamma_E = 3./2.*ni*phy_const.e*Te*ve    # Flux of internal energy
    wce = phy_const.e * B0 / m  # electron cyclotron frequency
    wce0 = np.amax(wce)
    #############################
    #       Compute the rates   #
    #############################
    if Gas_type == 'Xenon' :
        Eion = 12.1  # Ionization energy
        gamma_i = 3  # Excitation coefficient
        Kiz = (
            1.8e-13 * (((1.5 * Te) / Eion) ** 0.25) * np.exp(-4 * Eion / (3 * Te))
        )  # Ion - neutral  collision rate          TODO: Replace by better
    elif Gas_type == 'Krypton' :
        Eion = 14
        gamma_i = 3  # Excitation coefficient
        Kiz = (
            1.6e-13 * (((1.5 * Te) / Eion) ** 0.25) * np.exp(-4 * Eion / (3 * Te))
        )  # Ion - neutral  collision rate          TODO: Replace by better
    sigma = 2.0 * Te / Estar  # SEE yield
    sigma[sigma > 0.986] = 0.986
    nu_iw = (
        (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    )  # Ion - wall collision rate
    # Limit the collisions to inside the thruster
    index_L0 = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.0


    nu_ew = nu_iw / (1 - sigma)  # Electron - wall collision rate

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
    
    epsi0 = phy_const.epsilon_0
    phi_W = Te * np.log(np.sqrt(M / (2 * np.pi * m)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    c_s = np.sqrt(phy_const.e * Te / M)  # Sound velocity
    v_de = V/(LENGTH*B0) # Drift velocity 
    w_pi = np.sqrt((ni*phy_const.e**2)/(M*epsi0)) # Ion plasma frquency 
    
    if AlphaModel == 'Capelli' : # We implement the nu_an corresponding to the model chosen
        pass
    elif AlphaModel == 'Lafleur' :
        pass
    elif AlphaModel == 'Chodura' :
        nu_an = alpha_B1*w_pi*(1-np.exp((-v_de)/(alpha_B2*c_s)))
    elif AlphaModel == 'Data-driven' :
        nu_an = alpha_B1 * wce * (vi/(alpha_B2*c_s + v_de))
    else : # For the other models, we use the alpha_B vector as defined earlier
        nu_an = ng * Kel(Te, Tvec, int_result) + alpha_B*wce
        # print(Kel(Te, Tvec, int_result))
        # plt.figure()
        # plt.plot(Te, Kel(Te, Tvec, int_result), ls='', marker = "v")
        # plt.show()
        
    nu_m = (
        nu_an + nu_ew
    )  # Electron momentum - transfer collision frequency
    
    mu_eff = (phy_const.e / (m * nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
    )  # Effective mobility

    div_p=gradient(phy_const.e*ni*Te, d=Delta_x) # To be used with 5./2 and + div_p*ve belo

    S[0, :] = (-ng[:] * ni[:] * Kiz[:] + nu_iw[:] * ni[:]) * M  # Gas Density
    S[1, :] = (ng[:] * ni[:] * Kiz[:] - nu_iw[:] * ni[:]) * M  # Ion Density
    S[2, :] = (
        ng[:] * ni[:] * Kiz[:] * VG
        - (phy_const.e / (mu_eff[:] * M)) * ni[:] * ve[:]
        - nu_iw[:] * ni[:] * ui[:]
    ) * M  # Momentum
    S[3, :] = (
        -ng[:] * ni[:] * Kiz[:] * Eion * gamma_i * phy_const.e
        - nu_ew[:] * ni[:] * Ew * phy_const.e
        + 1.0 / mu_eff[:] * (ni[:] * ve[:]) ** 2.0 / ni[:] * phy_const.e
+div_p*ve
) # + phy_const.e*ni*Te*div_u #- gradI_term*ni*Te*grdI # Energy


# Compute the Current
@njit
def compute_I(P, V):
    # def trapz(y, d):
    #     return np.sum( (y[1:] + y[:-1]) )*d/2.0
    # def gradient(y, d):
    #     dp_dz = np.empty_like(y)
    #     dp_dz[1:-1] = (y[2:] - y[:-2]) / (2 * d)
    #     dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    #     dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    #     return dp_dz

    # TODO: This is already computed! Maybe move to the source
    #############################################################
    #       We give a name to the vars to make it more readable
    #############################################################
    ng = P[0, :]
    ni = P[1, :]
    ui = P[2, :]
    Te = P[3, :]
    ve = P[4, :]
    Gamma_i = ni * ui
    wce = phy_const.e * B0 / m  # electron cyclotron frequency
    wce0 = np.amax(wce)
    #############################
    #       Compute the rates   #
    #############################
    if Gas_type == 'Xenon' :
        Eion = 12.1  # Ionization energy
        gamma_i = 3  # Excitation coefficient
        Kiz = (
            1.8e-13 * (((1.5 * Te) / Eion) ** 0.25) * np.exp(-4 * Eion / (3 * Te))
        )  # Ion - neutral  collision rate          TODO: Replace by better
        # print("Te = ", Te, "Eion", Eion)
    elif Gas_type == 'Krypton' :
        Eion = 14
        gamma_i = 3  # Excitation coefficient
        Kiz = (
            1.6e-13 * (((1.5 * Te) / Eion) ** 0.25) * np.exp(-4 * Eion / (3 * Te))
        )  # Ion - neutral  collision rate          TODO: Replace by better
    sigma = 2.0 * Te / Estar  # SEE yield
    sigma[sigma > 0.986] = 0.986

    nu_iw = (
        (4.0 / 3.0) * (1.0 / (R2 - R1)) * np.sqrt(phy_const.e * Te / M)
    )  # Ion - wall collision rate d
    # Limit the collisions to inside the thruster
    index_L0 = np.argmax(x_center > L0)
    nu_iw[index_L0:] = 0.0


    nu_ew = nu_iw / (1 - sigma)  # Electron - wall collision rate

    # We implement the anomalous transfer mobility following the model chosen
   
    epsi0 = phy_const.epsilon_0
    phi_W = Te * np.log(np.sqrt(M / (2 * np.pi * m)) * (1 - sigma))  # Wall potential
    Ew = 2 * Te + (1 - sigma) * phi_W  # Energy lost at the wall

    c_s = np.sqrt(phy_const.e * Te / M)  # Sound velocity
    v_de = V/(LENGTH*B0) # Drift velocity 
    w_pi = np.sqrt((ni*phy_const.e**2)/(M*epsi0)) # Ion plasma frquency 
    
    if AlphaModel == 'Capelli' : # We implement the nu_an corresponding to the model chosen
        pass
    elif AlphaModel == 'Cafleur' :
        pass
    elif AlphaModel == '*Chodura' :
        nu_an = alpha_B1*w_pi*(1-np.exp((-v_de)/(alpha_B2*c_s)))
    elif AlphaModel == 'Data-driven' :
        nu_an = alpha_B1 * wce * (vi/(alpha_B2*c_s + v_de))
    else : # For the other models, we use the alpha_B vector as defined earlier
            nu_an = ng * Kel(Te, Tvec, int_result) + alpha_B*wce

    
    nu_m = (
         nu_an + nu_ew
    )  # Electron momentum - transfer collision frequency

    mu_eff = (phy_const.e / (m * nu_m)) * (
        1.0 / (1 + (wce / nu_m) ** 2)
    )  # Effective mobility

    dp_dz = np.empty_like(ni * Te)

    dp_dz[1:-1] = ((ni * Te)[2:] - (ni * Te)[:-2]) / (2 * Delta_x)
    dp_dz[0] = 2 * dp_dz[1] - dp_dz[2]
    dp_dz[-1] = 2 * dp_dz[-2] - dp_dz[-3]

    value_trapz_1 = (
        np.sum(
            (
                ((Gamma_i / (mu_eff * ni)) + dp_dz / ni)[1:]
                + ((Gamma_i / (mu_eff * ni)) + dp_dz / ni)[:-1]
            )
        )
        * Delta_x
        / 2.0
    )
    top = V + value_trapz_1

    value_trapz_2 = (
        np.sum(((1.0 / (mu_eff * ni))[1:] + (1.0 / (mu_eff * ni))[:-1])) * Delta_x / 2.0
    )
    bottom = phy_const.e * A0 * Rext + value_trapz_2

    I0 = top / bottom  # Discharge current density
    return I0 * phy_const.e * A0


@njit
def SetInlet(P_In, U_ghost, P_ghost, J=0.0, moment=1):

    U_Bohm = np.sqrt(phy_const.e * P_In[3] / M)

    if P_In[1] * P_In[2] < 0.0:
        U_ghost[0] = (mdot - M * P_In[1] * P_In[2] * A0) / (A0 * VG)
    else:
        U_ghost[0] = mdot / (A0 * VG)
    U_ghost[1] = P_In[1] * M
    U_ghost[2] = -2.0 * P_In[1] * U_Bohm * M - P_In[1] * P_In[2] * M
    U_ghost[3] = 3.0 / 2.0 * P_In[1] * phy_const.e * P_In[3]

    P_ghost[0] = U_ghost[0] / M  # ng
    P_ghost[1] = U_ghost[1] / M  # ni
    P_ghost[2] = U_ghost[2] / U_ghost[1]  # Ui
    P_ghost[3] = 2.0 / 3.0 * U_ghost[3] / (phy_const.e * P_ghost[1])  # Te
    P_ghost[4] = P_ghost[2] - J / (A0 * phy_const.e * P_ghost[1])  # ve


@njit
def SetOutlet(P_In, U_ghost, P_ghost, J=0.0):

    U_ghost[0] = P_In[0] * M
    U_ghost[1] = P_In[1] * M
    U_ghost[2] = P_In[1] * P_In[2] * M
    U_ghost[3] = 3.0 / 2.0 * P_In[1] * phy_const.e * Te_Cath

    P_ghost[0] = U_ghost[0] / M  # ng
    P_ghost[1] = U_ghost[1] / M  # ni
    P_ghost[2] = U_ghost[2] / U_ghost[1]  # Ui
    P_ghost[3] = 2.0 / 3.0 * U_ghost[3] / (phy_const.e * P_ghost[1])  # Te
    P_ghost[4] = P_ghost[2] - J / (A0 * phy_const.e * P_ghost[1])  # ve


##########################################################
#           Functions defining our numerics              #
##########################################################


# TODO: These are vector. Better allocate them
@njit
def computeMaxEigenVal_e(P):

    U_Bohm = np.sqrt(phy_const.e * P[3, :] / M)

    return np.maximum(np.abs(U_Bohm - P[4, :]) * 2, np.abs(U_Bohm + P[4, :]) * 2)


@njit
def computeMaxEigenVal_i(P):

    U_Bohm = np.sqrt(phy_const.e * P[3, :] / M)

    # return [max(l1, l2) for l1, l2 in zip(abs(U_Bohm - P[2,:]), abs(U_Bohm + P[2,:]))]
    return np.maximum(np.abs(U_Bohm - P[2, :]), np.abs(U_Bohm + P[2, :]))


@njit
def NumericalFlux(P, U, F_cell, F_interf):

    # Compute the max eigenvalue
    lambda_max_i_R = computeMaxEigenVal_i(P[:, 1 : NBPOINTS + 2])
    lambda_max_i_L = computeMaxEigenVal_i(P[:, 0 : NBPOINTS + 1])
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)

    lambda_max_e_R = computeMaxEigenVal_e(P[:, 1 : NBPOINTS + 2])
    lambda_max_e_L = computeMaxEigenVal_e(P[:, 0 : NBPOINTS + 1])
    lambda_max_e_12 = np.maximum(lambda_max_e_L, lambda_max_e_R)

    # Compute the flux at the interface
    F_interf[0, :] = 0.5 * (
        F_cell[0, 0 : NBPOINTS + 1] + F_cell[0, 1 : NBPOINTS + 2]
    ) - 0.5 * VG * (U[0, 1 : NBPOINTS + 2] - U[0, 0 : NBPOINTS + 1])
    F_interf[1, :] = 0.5 * (
        F_cell[1, 0 : NBPOINTS + 1] + F_cell[1, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i_12 * (U[1, 1 : NBPOINTS + 2] - U[1, 0 : NBPOINTS + 1])
    F_interf[2, :] = 0.5 * (
        F_cell[2, 0 : NBPOINTS + 1] + F_cell[2, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_i_12 * (U[2, 1 : NBPOINTS + 2] - U[2, 0 : NBPOINTS + 1])
    F_interf[3, :] = 0.5 * (
        F_cell[3, 0 : NBPOINTS + 1] + F_cell[3, 1 : NBPOINTS + 2]
    ) - 0.5 * lambda_max_e_12 * (U[3, 1 : NBPOINTS + 2] - U[3, 0 : NBPOINTS + 1])


@njit
def ComputeDelta_t(P):
    # Compute the max eigenvalue
    lambda_max_i_R = computeMaxEigenVal_i(P[:, 1 : NBPOINTS + 2])
    lambda_max_i_L = computeMaxEigenVal_i(P[:, 0 : NBPOINTS + 1])
    lambda_max_i_12 = np.maximum(lambda_max_i_L, lambda_max_i_R)

    lambda_max_e_R = computeMaxEigenVal_e(P[:, 1 : NBPOINTS + 2])
    lambda_max_e_L = computeMaxEigenVal_e(P[:, 0 : NBPOINTS + 1])
    lambda_max_e_12 = np.maximum(lambda_max_e_L, lambda_max_e_R)

    Delta_t = CFL * Delta_x / (max(max(lambda_max_e_12), max(lambda_max_i_12)))
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
    ResultsFigs = Results + "/Figs"
    if not os.path.exists(ResultsFigs):
        os.makedirs(ResultsFigs)
    ResultsData = Results + "/Data"
    if not os.path.exists(ResultsData):
        os.makedirs(ResultsData)

    # Save the data
    filenameTemp = ResultsData + "/MacroscopicVars_" + f"{i_save:08d}" + ".pkl"
    pickle.dump(
        [time, P, U, P_Inlet, P_Outlet, J, V, B0, x_center], open(filenameTemp, "wb")
    )  # TODO: Save the current and the electric field


##########################################################################################################
#           Initial field                                                                                #
#           P := Primitive vars [0: ng, 1: ni, 2: ui, 3: Te, 4: ve]                                      #
#           U := Conservative vars [0: rhog, 1: rhoi, 2: rhoiui, 3: 3./2.ni*e*Te]                        #
#                                                                                                        #
##########################################################################################################

NG0 = 5e18
NI0 = 5e17
TE0 = 5.0


time = 0.0
iter = 0
J = 0.0  # Initial Current

# We initialize the primitive variables
P[0, :] *= mdot / (M * A0 * VG)  # Initial propellant density ng
P[1, :] *= NI0  # Initial ni
P[2, :] *= 0.0  # Initial vi
P[3, :] *= TE0  # Initial Te
P[4, :] *= P[2, :] - J / (A0 * phy_const.e * P[1, :])  # Initial Ve

# We initialize the conservative variables
PrimToCons(P, U)


##########################################################################################
#           Loop with Forward Euler                                                      #
#           U^{n+1}_j = U^{n}_j - Dt/Dx(F^n_{j+1/2} - F^n_{j-1/2}) + Dt S^n_j            #
#                                                                                        #
##########################################################################################

if TIMESCHEME == "Forward Euler":
    J = compute_I(P, V)
    while time < TIMEFINAL:
        # Save results
        if (iter % SAVERATE) == 0:
            SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save)
            i_save += 1
            print(
                "Iter = ",
                iter,
                "\tTime = {:.2f}~µs".format(time / 1e-6),
                "\tJ = {:.4f}~A".format(J),
            )

        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J, 1)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the convective Delta t
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:, :] = (
            U[:, :]
            - Delta_t
            / Delta_x
            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
            + Delta_t * S[:, :]
        )

        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        time += Delta_t
        iter += 1

if TIMESCHEME == "TVDRK3":

    while time < TIMEFINAL:
        # Save results
        if (iter % SAVERATE) == 0:
            SaveResults(P, U, P_Inlet, P_Outlet, J, V, x_center, time, i_save)
            i_save += 1
            print(
                "Iter = {}".format(iter),
                "\t Time = {:.4f} µs".format(time * 1e6),
                "\t J = {:.4f} A".format(J),
                "\t V = {:.4f} V".format(V),
            )
            if iter == 5:
                sys.exit(1)
        #################################################
        #           FIRST STEP RK3
        #################################################

        # Copy the solution to store it
        U_1[:, :] = U[:, :]
        ConsToPrim(U_1, P_1, J)
        J_1 = compute_I(P, V)
        ConsToPrim(U_1, P_1, J_1)

        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)
        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)
        # Compute the convective Delta t (Only in the first step)
        Delta_t = ComputeDelta_t(np.concatenate([P_Inlet, P, P_Outlet], axis=1))

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

        # Update the solution
        U[:, :] = (
            U[:, :]
            - Delta_t
            / Delta_x
            * (F_interf[:, 1 : NBPOINTS + 1] - F_interf[:, 0:NBPOINTS])
            + Delta_t * S[:, :]
        )

        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC Circuit
        if Circuit:
            dJdt = (J - J0) / Delta_t

            RHS_Volt0[0] = X_Volt0[1]
            RHS_Volt0[1] = (
                -1 / (R * C) * X_Volt0[1] - 1.0 / (L * C) * X_Volt0[0] + 1 / C * dJdt
            )
            X_Volt1 = X_Volt0 + Delta_t * RHS_Volt0

        #################################################
        #           SECOND STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J, 2)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )

        # Compute the source in the center of the cell
        Source(P, S)

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

        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC Circuit
        if Circuit:
            dJdt = (J - J0) / Delta_t
            RHS_Volt1[0] = X_Volt1[1]
            RHS_Volt1[1] = (
                -1 / (R * C) * X_Volt1[1] - 1.0 / (L * C) * X_Volt1[0] + 1 / C * dJdt
            )
            X_Volt2 = 0.75 * X_Volt0 + 0.25 * X_Volt1 + 0.25 * Delta_t * RHS_Volt1

        #################################################
        #           THIRD STEP RK3
        #################################################
        # Set the boundaries
        SetInlet(P[:, 0], U_Inlet, P_Inlet, J, 3)
        SetOutlet(P[:, -1], U_Outlet, P_Outlet, J)

        # Compute the Fluxes in the center of the cell
        InviscidFlux(np.concatenate([P_Inlet, P, P_Outlet], axis=1), F_cell)

        # Compute the Numerical at the interfaces
        NumericalFlux(
            np.concatenate([P_Inlet, P, P_Outlet], axis=1),
            np.concatenate([U_Inlet, U, U_Outlet], axis=1),
            F_cell,
            F_interf,
        )
        # Compute the source in the center of the cell
        Source(P, S)

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
        # Compute the current
        J = compute_I(P, V)

        # Compute the primitive vars for next step
        ConsToPrim(U, P, J)

        # Compute RLC Circuit
        if Circuit:
            dJdt = (J - J0) / Delta_t
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
            J0 = J
            X_Volt0[:] = X_Volt3[:]

            # Change the Voltage
            V = V0 - X_Volt0[0]
        time += Delta_t
        if (iter %SAVERATE) ==0:
            filename = Results + "/time_vec_njit.dat"
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

ttime_end = ttime.time()
print("Exec time = {:.2f} s".format(ttime_end - tttime_start))
