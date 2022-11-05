from math import sqrt, log, floor
import re
import numpy as np
from functools import reduce

def modu(A, B):
    A = float(A)
    B = float(B)
    m = sqrt((A - B) * (A - B))
    return m

def mass_dif_allowed(ppm, measured_mass):
    theoretical_mass = (((ppm / (10 ^ -6))) / measured_mass)
    return theoretical_mass

def ppm_calc(measured_mass, theoretical_mass):
    if (type(measured_mass) != float) or (type(theoretical_mass) != float):
        return None

    if float(theoretical_mass) != 0:
        ppm = abs(((1 * (10 ^ 6)) * (float(measured_mass) - float(theoretical_mass))) / float(theoretical_mass))
        mass_diff = modu(measured_mass, theoretical_mass)
        return [ppm, mass_diff]
    else:
        return None

def split_formula(formula):
    cReg = re.compile("[Cc][0-9]{0,3}")
    hReg = re.compile("[Hh][0-9]{0,3}")
    oReg = re.compile("[Oo][0-9]{0,3}")
    nReg = re.compile("[Nn][0-9]{0,3}")

    elements = {}

    count = 0
    for i in cReg.findall(formula):
        if len(i) == 1:
            count += 1
        else:
            count += int(i[1:])
    elements["C"] = count

    count = 0
    for i in hReg.findall(formula):
        if len(i) == 1:
            count += 1
        else:
            count += int(i[1:])
    elements["H"] = count

    count = 0
    for i in nReg.findall(formula):
        if len(i) == 1:
            count += 1
        else:
            count += int(i[1:])
    elements["N"] = count

    count = 0
    for i in oReg.findall(formula):
        if len(i) == 1:
            count += 1
        else:
            count += int(i[1:])
    elements["O"] = count



    return elements

def calc_MW(formula):
    form_dict = split_formula(formula)
    no_of_carbon = (int(form_dict['Carbon']) * c)
    no_of_hydrogen = (int(form_dict['Hydrogen']) * h)
    no_of_oxygen = (int(form_dict['Oxygen']) * o)
    no_nitrogen = (int(form_dict['Nitrogen']) * n)

    MWt = (no_of_carbon  + no_of_hydrogen + no_of_oxygen  + no_nitrogen )

    return MWt

def precision_digits(f, width):
    """Return number of digits after decimal point to print f in width chars.
    """
    precision = log(abs(f), 10)
    if precision < 0:
        precision = 0
    precision = width - int(floor(precision))
    precision -= 3 if f < 0 else 2  # sign and decimal point
    if precision < 1:
        precision = 1
    return precision

def gcd(numbers):
    """Return greatest common divisor of integer numbers.

    Using Euclid's algorithm.

    Examples
    --------
    >>> gcd([4])
    4
    >>> gcd([3, 6])
    3
    >>> gcd([6, 7])
    1

    """

    def _gcd(a, b):
        """Return greatest common divisor of two integer numbers."""
        while b:
            a, b = b, a % b
        return a

    return reduce(_gcd, numbers)

class Spectrum(object):
    """Mass distribution of compound.
    """

    def __init__(self, data):
        self._data = data
        self.masses = data[0]
        self.abundances = data[1]
        self.nominals = data[2]
        self.isotopes = data[3]
        self.normalised = self.abundances / self.abundances[np.argmax(self.abundances)]


    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, value):
        if isinstance(value, np.ndarray):
            self._masses = value
        else:
            return None

    @property
    def abundances(self):
        return self._abundances

    @abundances.setter
    def abundances(self, value):
        if isinstance(value, np.ndarray):
            self._abundances = value
        else:
            return None

    @property
    def nominals(self):
        return self._nominals

    @nominals.setter
    def nominals(self, value):
        if isinstance(value, np.ndarray):
            self._nominals = value
        else:
            return None

    @property
    def normalised(self):
        return self._normalised

    @normalised.setter
    def normalised(self, value):
        if isinstance(value, np.ndarray):
            self._normalised = value
        else:
            return None

    @property
    def range(self):
        """Return smallest and largest isotopic masses."""
        return {"min": self.masses[np.argmin(self.masses)], "max": self.masses[np.argmax(self.masses)]}

    @property
    def peak(self):
        """Return most abundant mass and fraction."""
        return self.isotopes[np.argmax(self.normalised)]

    @property
    def mean(self):
        """Return mean of all masses in spectrum."""
        return np.mean(self.masses)


