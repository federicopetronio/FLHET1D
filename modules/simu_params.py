import numpy as np
import configparser
from scipy import constants as phy_const
import math
import os
import pandas as pd


class SimuParameters():
    """this class "pack" the plasma parameters to be used for normalisation"""

    def __init__(self,
                fconfigfile
                ):

        self.configinipath = fconfigfile

        config = configparser.ConfigParser()
        config.read(fconfigfile)

        physicalParameters = config["Physical Parameters"]

        self.VG = float(physicalParameters["Gas velocity"])  # Gas velocity
        self.Mi = float(physicalParameters["Ion Mass"]) * phy_const.m_u  # Ion Mass
        self.R1 = float(physicalParameters["Inner radius"])  # Inner radius of the thruster
        self.R2 = float(physicalParameters["Outer radius"])  # Outer radius of the thruster
        self.A0 = np.pi * (self.R2**2 - self.R1**2)  # Area of the thruster
        self.LX = float(physicalParameters["Length of axis"])  # length of Axis of the simulation
        self.LTHR = float(
            physicalParameters["Length of thruster"]
        )  # length of thruster (position of B_max)

        self.MDOT = float(physicalParameters["Mass flow"])  # Mass flow rate of propellant
        self.Te_Cath = float(
            physicalParameters["e- Temperature Cathode"]
        )  # Electron temperature at the cathode
        self.TE0 = float(physicalParameters["Initial e- temperature"]) # Initial electron temperature at the cathode.
        self.NI0 = float(physicalParameters["Initial plasma density"]) # Initial plasma density.
        #NG0 = float(physicalParameters["Initial neutrals density"]) # Initial neutrals density. No need for this parameter it is processed to have be coehrent with MDOT, AO and VG.
        self.Rext = float(physicalParameters["Ballast resistor"])  # Resistor of the ballast
        self.V = float(physicalParameters["Voltage"])  # Potential difference
        
        self.Circuit = bool(
            config.getboolean("Physical Parameters", "Circuit", fallback=False)
        )  # RLC Circuit
        if self.Circuit:
            self.R = float(physicalParameters["R"])
            self.L = float(physicalParameters["L"])
            self.C = float(physicalParameters["C"])

        self.HEATFLUX = bool(
            config.getboolean("Physical Parameters", "Electron heat flux", fallback=False)
        )

        # Anomalous Transport
        anomParams  = config["Anomalous Transport"]
        self.boolInputAnomNu     = bool(
            config.getboolean("Anomalous Transport", "Input anomalous nu", fallback=False)
        )

        # Magnetic field configuration
        MagneticFieldConfig = config["Magnetic field configuration"]


        print(MagneticFieldConfig["Type"] + " Magnetic Field")
        self.BTYPE  = MagneticFieldConfig["Type"]
        self.BMAX = float(MagneticFieldConfig["Max B-field"])  # Max Mag field
        self.B0 = float(MagneticFieldConfig["B-field at 0"])  # Mag field at x=0
        self.BLX = float(MagneticFieldConfig["B-field at LX"])  # Mag field at x=LX
        self.LB1 = float(MagneticFieldConfig["Length B-field 1"])  # Length for magnetic field
        self.LB2 = float(MagneticFieldConfig["Length B-field 2"])  # Length for magnetic field


        # Ionization source term configuration
        IonizationConfig = config["Ionization configuration"]
        self.boolSizImposed = bool(
            config.getboolean("Ionization configuration", "source is imposed", fallback=False)
        )
        if self.boolSizImposed:
            print("The ionization source term is imposed as specified in T.Charoy's thesis, section 2.2.2.")
        self.SIZMAX  = float(IonizationConfig["Maximum S_iz value"])  # Max Mag field
        self.LSIZ1   = float(IonizationConfig["Position of 1st S_iz zero"])  # Mag field at x=0
        self.LSIZ2   = float(IonizationConfig["Position of 2nd S_iz zero"])  # Mag field at x=LX
        assert(self.LSIZ2 >= self.LSIZ1)
        try:
            self.Eion   = float(IonizationConfig["ionization energy"])
            self.gamma_i= float(IonizationConfig["coefficient gamma_i"])
            self.Einj= float(IonizationConfig["injected e- temperature"])
        except KeyError:
            print("\tUserWarning: the config file " + fconfigfile + " does not specify all three of the parameters 'ionization energy', 'coefficient gamma_i' and 'injection e- temperature'. It may be a config file suited for an older version of the code. Default values for these values are used.")
            self.Eion   = 12.1
            self.gamma_i= 3.0
            self.Einj   = 10.

        # Collisions parameters
        CollisionsConfig = config["Collisions"]
        self.boolIonColl = bool(
            config.getboolean("Collisions", "enable ionization coll.", fallback=False)
        )    
        self.KEL = float(CollisionsConfig["Elastic collisions reaction rate"])

        # Wall interactions
        WallInteractionConfig = config["Wall interactions"]
        self.wall_inter_type = WallInteractionConfig["Type"]
        self.ESTAR = float(WallInteractionConfig["Crossover energy"])  # Crossover energy
        assert((self.wall_inter_type == "Default")|(self.wall_inter_type == "None"))

        ##########################################################
        #           NUMERICAL PARAMETERS
        ##########################################################
        NumericsConfig = config["Numerical Parameteres"]

        self.NBPOINTS_INIT = int(NumericsConfig["Number of points"])  # Number of cells
        try:
            self.SAVERATE = int(NumericsConfig["Save rate"])  # Rate at which we store the data
        except KeyError:
            self.SAVERATE = -1
        self.CFL = float(NumericsConfig["CFL"])  # Nondimensional size of the time step
        self.TIMEFINAL = float(NumericsConfig["Final time"])  # Last time of simulation
        self.Results = NumericsConfig["Result dir"]  # Name of result directory
        self.TIMESCHEME = NumericsConfig["Time integration"]  # Time integration scheme
        self.IMPlICIT   = bool(
            config.getboolean("Numerical Parameteres", "Implicit heat flux", fallback=False) ) # Time integration scheme for heat flux equation
        self.MESHREFINEMENT =  bool(
            config.getboolean("Numerical Parameteres", "Mesh refinement", fallback=False) )
        if self.MESHREFINEMENT:
            self.MESHLEVELS = int(NumericsConfig["Mesh levels"])
            self.REFINEMENTLENGTH = float(NumericsConfig["Refinement length"])
        else:
            self.MESHLEVELS = 0
            self.REFINEMENTLENGTH = 0.
        self.START_FROM_INPUT = bool(
            config.getboolean("Numerical Parameteres", "Start from input profiles", fallback=False) )
        if self.START_FROM_INPUT:
            self.INPUT_DIR = NumericsConfig["Input profiles directory"]


    def save_config_file(self, filename):

        config = configparser.ConfigParser()
        config.read(self.configinipath)

        if not os.path.exists(self.Results):
            os.makedirs(self.Results)
        
        with open(self.Results + "/" + filename, "w") as configfile:
            config.write(configfile)


    def extract_anom_coeffs(self):

        # Extracting the anomalous transport coefficients

        config = configparser.ConfigParser()
        config.read(self.configinipath)

        anomParams = config["Anomalous Transport"]
        alpha_B1 = float(
            anomParams["Anomalous coefficient alpha_B1"]
        )  # Anomalous transport
        alpha_B2 = float(
            anomParams["Anomalous coefficient alpha_B2"]
        )  # Anomalous transport


        return alpha_B1, alpha_B2
    

    def extract_anom_nu(self):

        # Extracting the anomalous transport coefficients

        config = configparser.ConfigParser()
        config.read(self.configinipath)

        anomParams  = config["Anomalous Transport"]
        inputfil    = anomParams["Anomalous nu file"]

        return pd.read_csv(inputfil, sep='\t', header=0)
    

    def return_tiled_domain(self):

        x_mesh = np.linspace(0, self.LX, self.NBPOINTS_INIT + 1)  # Mesh in the interface
        if self.MESHREFINEMENT and self.MESHLEVELS > 1:
            # get the point below the refinement length
            Dx_notRefined = x_mesh[1]
            #First level
            i_refinement = int(np.floor(self.REFINEMENTLENGTH/Dx_notRefined))
            mesh_level_im1 = np.linspace(0, x_mesh[i_refinement], i_refinement*2 + 1)
            mesh_refinedim1 = np.concatenate((mesh_level_im1[:-1], x_mesh[i_refinement:]))
            # plt.plot(x_mesh, np.zeros_like(x_mesh), linestyle='None', marker='o', markersize=2)
            # plt.plot(mesh_refinedim1, np.ones_like(mesh_refinedim1), linestyle='None', marker='o', markersize=2)

            #Secondand rest of levels level
            for i_level in range(2, self.MESHLEVELS + 1):
                i_refinement_level   = int((np.shape(mesh_level_im1)[0] - 1)/2)
                mesh_level_i         = np.linspace(0, mesh_level_im1[i_refinement_level], i_refinement_level*2 + 1)
                mesh_refined_level_i = np.concatenate((mesh_level_i[:-1], mesh_refinedim1[i_refinement_level:]))

                mesh_level_im1      = mesh_level_i
                mesh_refinedim1     = mesh_refined_level_i
                # plt.plot(mesh_refined_level_i, np.ones_like(mesh_refined_level_i)*i_level, linestyle='None', marker='o', markersize=2)
            x_mesh = np.copy(mesh_refinedim1)

        x_center = (x_mesh[1:] + x_mesh[:-1])/2
        Delta_x  =  x_mesh[1:] - x_mesh[:-1]
        # We create an array that also include the position of the ghost cells
        x_center_extended = np.insert(x_center, 0, -x_center[0])
        x_center_extended = np.append(x_center_extended, x_center[-1] + Delta_x[-1])
        Delta_x_extended  = np.insert(Delta_x, 0, Delta_x[0])
        Delta_x_extended  = np.append(Delta_x_extended, Delta_x[-1])

        return x_mesh, x_center, Delta_x, x_center_extended, Delta_x_extended

