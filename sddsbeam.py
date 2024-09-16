import numpy as np
from SDDSFile import SDDSFile, SDDS_Types
import munch

class sddsbeam(munch.Munch):

    m_e = 9.1093837015e-31
    speed_of_light = 299792458.0
    e = 1.602176634e-19
    elementary_charge = 1.602176634e-19
    pi = 3.141592653589793
    electron_mass = 9.1093837015e-31
    particle_mass = m_e
    E0 = particle_mass * speed_of_light**2
    E0_eV = E0 / elementary_charge
    q_over_c = elementary_charge / speed_of_light

    def __init__(self):
        self.beam = munch.Munch()

    @property
    def cpx(self):
        return self.beam['px'] / self.q_over_c
    @property
    def cpy(self):
        return self.beam['py'] / self.q_over_c
    @property
    def cpz(self):
        return self.beam['pz'] / self.q_over_c

    @property
    def cp(self):
        return np.sqrt(self.cpx**2 + self.cpy**2 + self.cpz**2)

    @property
    def gamma(self):
        return np.sqrt(1+(self.cp/self.E0_eV)**2)

    @property
    def vz(self):
        velocity_conversion = 1 / (self.m_e * self.gamma)
        return velocity_conversion * self.beam['pz']

    @property
    def Bz(self):
        return self.vz / self.speed_of_light

    @property
    def BetaGamma(self):
        return self.cp/self.E0_eV

    def read_SDDS_file(self, fileName, charge=None, ascii=False, page=-1):
        self.sddsindex = 1
        elegantObject = SDDSFile(index=(self.sddsindex), ascii=ascii)
        elegantObject.read_file(fileName, page=page)
        elegantData = elegantObject.data
        for k, v in elegantData.items():
            # case handling for multiple ELEGANT runs per file
            # only extract the first run (in ELEGANT this is the fiducial run)
            if isinstance(v, np.ndarray):
                if v.ndim > 1:
                    self.beam[k] = v[0]
                else:
                    # print(k)
                    self.beam[k] = v
            else:
                self.beam[k] = v
        self.filename = fileName
        self['code'] = "SDDS"
        cp = (self.beam['p']) * self.E0_eV
        cpz = cp / np.sqrt(self.beam['xp']**2 + self.beam['yp']**2 + 1)
        cpx = self.beam['xp'] * cpz
        cpy = self.beam['yp'] * cpz
        self.beam['px'] = cpx * self.q_over_c
        self.beam['py'] = cpy * self.q_over_c
        self.beam['pz'] = cpz * self.q_over_c
        # self.beam['t'] = self.beam['t']
        self.beam['z'] = (-1*self.Bz * self.speed_of_light) * (self.beam.t-np.mean(self.beam.t)) #np.full(len(self.t), 0)
        if 'Charge' in elegantData and len(elegantData['Charge']) > 0:
            self.beam['total_charge'] = elegantData['Charge'][0]
            self.beam['charge'] = np.full(len(self.beam['z']), self.beam['total_charge']/len(self.beam['x']))
        elif charge is None:
            self.beam['total_charge'] = 0
            self.beam['charge'] = np.full(len(self.beam['z']), self.beam['total_charge']/len(self.beam['x']))
        else:
            self.beam['total_charge'] = charge
            self.beam['charge'] = np.full(len(self.beam['z']), self.beam['total_charge']/len(self.beam['x']))
        self.beam['nmacro'] = np.full(len(self.beam['z']), 1)
        # self.beam['charge'] = []

    def write_SDDS_file(self, filename, ascii=False, xyzoffset=[0,0,0]):
        """Save an SDDS file using the SDDS class."""
        xoffset = xyzoffset[0]
        yoffset = xyzoffset[1]
        zoffset = xyzoffset[2] # Don't think I need this because we are using t anyway...
        self.sddsindex += 1
        x = SDDSFile(index=(self.sddsindex), ascii=ascii)

        Cnames = ["x", "xp", "y", "yp", "t", "p", "particleID"]
        Ctypes = [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_LONG]
        Csymbols = ["", "x'","","y'","","",""]
        Cunits = ["m","","m","","s","m$be$nc",""]
        # Ccolumns = [np.array(self.beam.x) - float(xoffset), self.beam.xp, np.array(self.beam.y) - float(yoffset), self.beam.yp, self.beam.t , self.cp/self.E0_eV, self.beam.particleID]
        # modified Ccolumns here to take momentum directly from the p variable (as used in my code!)
        Ccolumns = [np.array(self.beam.x) - float(xoffset), self.beam.xp, np.array(self.beam.y) - float(yoffset), self.beam.yp, self.beam.t , self.beam.p, self.beam.particleID]
        x.add_columns(Cnames, Ccolumns, Ctypes, Cunits, Csymbols)

        Pnames = ["pCentral", "Charge", "Particles"]
        Ptypes = [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE]
        Psymbols = ["p$bcen$n", "", ""]
        Punits = ["m$be$nc", "C", ""]
        parameterData = [np.mean(self.BetaGamma), abs(self.beam['total_charge']), len(self.beam.x)]
        x.add_parameters(Pnames, parameterData, Ptypes, Punits, Psymbols)

        x.write_file(filename)

    def setbeam_charge(self, charge):
        self.beam['total_charge'] = charge
