import numpy as np
import configparser
from scipy import constants as phy_const
import math
import os


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

        # Magnetic field configuration
        MagneticFieldConfig = config["Magnetic field configuration"]

        if MagneticFieldConfig["Type"] == "Default":
            print(MagneticFieldConfig["Type"] + " Magnetic Field")

            self.BMAX = float(MagneticFieldConfig["Max B-field"])  # Max Mag field
            self.B0 = float(MagneticFieldConfig["B-field at 0"])  # Mag field at x=0
            self.BLX = float(MagneticFieldConfig["B-field at LX"])  # Mag field at x=LX
            self.LB1 = float(MagneticFieldConfig["Length B-field 1"])  # Length for magnetic field
            self.LB2 = float(MagneticFieldConfig["Length B-field 2"])  # Length for magnetic field


        # Ionization source term configuration
        IonizationConfig = config["Ionization configuration"]
        self.ionization_type = IonizationConfig["Type"]
        if self.ionization_type == "SourceIsImposed":
            print("The ionization source term is imposed as specified in T.Charoy's thesis, section 2.2.2.")
        self.SIZMAX  = float(IonizationConfig["Maximum S_iz value"])  # Max Mag field
        self.LSIZ1   = float(IonizationConfig["Position of 1st S_iz zero"])  # Mag field at x=0
        self.LSIZ2   = float(IonizationConfig["Position of 2nd S_iz zero"])  # Mag field at x=LX
        assert(self.LSIZ2 >= self.LSIZ1)

        # Collisions parameters
        CollisionsConfig = config["Collisions"]
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
        self.SAVERATE = int(NumericsConfig["Save rate"])  # Rate at which we store the data
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

        self.x_mesh, self.x_center, self.Delta_x, self.x_center_extd, self.Delta_x_extd = self.return_tiled_domain()

        self.NBPOINTS = np.shape(self.x_center)[0]

        self.Barr = self.return_imposed_B()


    def save_config_file(self, filename):

        config = configparser.ConfigParser()
        config.read(self.configinipath)

        if not os.path.exists(self.Results):
            os.makedirs(self.Results)
        
        with open(self.Results + "/" + filename, "w") as configfile:
            config.write(configfile)


    def extract_anomalous(self):

        # Extracting the anomalous transport coefficients

        config = configparser.ConfigParser()
        config.read(self.configinipath)

        physicalParameters = config["Physical Parameters"]
        alpha_B1 = float(
            physicalParameters["Anomalous transport alpha_B1"]
        )  # Anomalous transport
        alpha_B2 = float(
            physicalParameters["Anomalous transport alpha_B2"]
        )  # Anomalous transport


        return alpha_B1, alpha_B2
    

    def return_tiled_domain(self):

        x_mesh = np.linspace(0, self.LX, self.NBPOINTS_INIT + 1)  # Mesh in the interface
        if self.MESHREFINEMENT:
            # get the point below the refinement length
            Dx_notRefined = x_mesh[1]
            #First level
            i_refinement = int(np.floor(self.REFINEMENTLENGTH/Dx_notRefined))
            mesh_level_im1 = np.linspace(0, x_mesh[i_refinement], i_refinement*2 + 1)
            mesh_refinedim1 = np.concatenate((mesh_level_im1[:-1], x_mesh[i_refinement:]))
            # plt.plot(x_mesh, np.zeros_like(x_mesh), linestyle='None', marker='o', markersize=2)
            # plt.plot(mesh_refinedim1, np.ones_like(mesh_refinedim1), linestyle='None', marker='o', markersize=2)

            #Secondand rest of levels level
            if self.MESHLEVELS > 1:
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


    def return_imposed_B(self):

        fBMAX = self.BMAX
        fB0 = self.B0
        fBLX = self.BLX
        fLTHR = self.LTHR
        fLX = self.LX
        fLB1 = self.LB1
        fLB2 = self.LB2
        fx_center = self.x_center


        a1 = (fBMAX - fB0)/(1 - math.exp(-fLTHR**2/(2*fLB1**2)))
        a2 = (fBMAX - fBLX)/(1 - math.exp(-(fLX - fLTHR)**2/(2*fLB2**2)))
        b1 = fBMAX - a1
        b2 = fBMAX - a2
        Barr1 = a1*np.exp(-(fx_center - fLTHR)**2/(2*fLB1**2)) + b1
        Barr2 = a2*np.exp(-(fx_center - fLTHR)**2/(2*fLB2**2)) + b2    # Magnetic field outside the thruster

        Barr = np.where(fx_center <= fLTHR, Barr1, Barr2)

        return Barr


    def return_imposed_Siz(self):

        fLSIZ1 = self.LSIZ1
        fLSIZ2 = self.LSIZ2
        fSIZMAX= self.SIZMAX
        fx_center = self.x_center

        xm = (fLSIZ1 + fLSIZ2)/2
        Siz = fSIZMAX*np.cos(math.pi*(fx_center - xm)/(fLSIZ2 - fLSIZ1))
        Siz = np.where((fx_center < fLSIZ1)|(fx_center > fLSIZ2), 0., Siz)

        return Siz
