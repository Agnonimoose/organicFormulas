from baseBlocks import *
from chemFuncs import *
import re, itertools
import numpy as np

# Common chemical groups
GROUPS = {
    'Abu': 'C4H7NO',
    'Acet': 'C2H3O',
    'Acm': 'C3H6NO',
    'Adao': 'C10H15O',
    'Aib': 'C4H7NO',
    'Ala': 'C3H5NO',
    'Arg': 'C6H12N4O',
    'Argp': 'C6H11N4O',
    'Asn': 'C4H6N2O2',
    'Asnp': 'C4H5N2O2',
    'Asp': 'C4H5NO3',
    'Aspp': 'C4H4NO3',
    'Asu': 'C8H13NO3',
    'Asup': 'C8H12NO3',
    'Boc': 'C5H9O2',
    'Bom': 'C8H9O',
    'Bpy': 'C10H8N2',  # Bipyridine
    'Brz': 'C8H6BrO2',
    'Bu': 'C4H9',
    'Bum': 'C5H11O',
    'Bz': 'C7H5O',
    'Bzl': 'C7H7',
    'Bzlo': 'C7H7O',
    'Cha': 'C9H15NO',
    'Chxo': 'C6H11O',
    'Cit': 'C6H11N3O2',
    'Citp': 'C6H10N3O2',
    'Clz': 'C8H6ClO2',
    'Cp': 'C5H5',
    'Cy': 'C6H11',
    'Cys': 'C3H5NOS',
    'Cysp': 'C3H4NOS',
    'Dde': 'C10H13O2',
    'Dnp': 'C6H3N2O4',
    'Et': 'C2H5',
    'Fmoc': 'C15H11O2',
    'For': 'CHO',
    'Gln': 'C5H8N2O2',
    'Glnp': 'C5H7N2O2',
    'Glp': 'C5H5NO2',
    'Glu': 'C5H7NO3',
    'Glup': 'C5H6NO3',
    'Gly': 'C2H3NO',
    'Hci': 'C7H13N3O2',
    'Hcip': 'C7H12N3O2',
    'His': 'C6H7N3O',
    'Hisp': 'C6H6N3O',
    'Hser': 'C4H7NO2',
    'Hserp': 'C4H6NO2',
    'Hx': 'C6H11',
    'Hyp': 'C5H7NO2',
    'Hypp': 'C5H6NO2',
    'Ile': 'C6H11NO',
    'Ivdde': 'C14H21O2',
    'Leu': 'C6H11NO',
    'Lys': 'C6H12N2O',
    'Lysp': 'C6H11N2O',
    'Mbh': 'C15H15O2',
    'Me': 'CH3',
    'Mebzl': 'C8H9',
    'Meobzl': 'C8H9O',
    'Met': 'C5H9NOS',
    'Mmt': 'C20H17O',
    'Mtc': 'C14H19O3S',
    'Mtr': 'C10H13O3S',
    'Mts': 'C9H11O2S',
    'Mtt': 'C20H17',
    'Nle': 'C6H11NO',
    'Npys': 'C5H3N2O2S',
    'Nva': 'C5H9NO',
    'Odmab': 'C20H26NO3',
    'Orn': 'C5H10N2O',
    'Ornp': 'C5H9N2O',
    'Pbf': 'C13H17O3S',
    'Pen': 'C5H9NOS',
    'Penp': 'C5H8NOS',
    'Ph': 'C6H5',
    'Phe': 'C9H9NO',
    'Phepcl': 'C9H8ClNO',
    'Phg': 'C8H7NO',
    'Pmc': 'C14H19O3S',
    'Ppa': 'C8H7O2',
    'Pro': 'C5H7NO',
    'Prop': 'C3H7',
    'Py': 'C5H5N',
    'Pyr': 'C5H5NO2',
    'Sar': 'C3H5NO',
    'Ser': 'C3H5NO2',
    'Serp': 'C3H4NO2',
    'Sta': 'C8H15NO2',
    'Stap': 'C8H14NO2',
    'Tacm': 'C6H12NO',
    'Tbdms': 'C6H15Si',
    'Tbu': 'C4H9',
    'Tbuo': 'C4H9O',
    'Tbuthio': 'C4H9S',
    'Tfa': 'C2F3O',
    'Thi': 'C7H7NOS',
    'Thr': 'C4H7NO2',
    'Thrp': 'C4H6NO2',
    'Tips': 'C9H21Si',
    'Tms': 'C3H9Si',
    'Tos': 'C7H7O2S',
    'Trp': 'C11H10N2O',
    'Trpp': 'C11H9N2O',
    'Trt': 'C19H15',
    'Tyr': 'C9H9NO2',
    'Tyrp': 'C9H8NO2',
    'Val': 'C5H9NO',
    'Valoh': 'C5H9NO2',
    'Valohp': 'C5H8NO2',
    'Xan': 'C13H9O',
}

# Amino acids - H2O
AMINOACIDS = {
    'G': 'C2H3NO',  # Glycine, Gly
    'P': 'C5H7NO',  # Proline, Pro
    'A': 'C3H5NO',  # Alanine, Ala
    'V': 'C5H9NO',  # Valine, Val
    'L': 'C6H11NO',  # Leucine, Leu
    'I': 'C6H11NO',  # Isoleucine, Ile
    'M': 'C5H9NOS',  # Methionine, Met
    'C': 'C3H5NOS',  # Cysteine, Cys
    'F': 'C9H9NO',  # Phenylalanine, Phe
    'Y': 'C9H9NO2',  # Tyrosine, Tyr
    'W': 'C11H10N2O',  # Tryptophan, Trp
    'H': 'C6H7N3O',  # Histidine, His
    'K': 'C6H12N2O',  # Lysine, Lys
    'R': 'C6H12N4O',  # Arginine, Arg
    'Q': 'C5H8N2O2',  # Glutamine, Gln
    'N': 'C4H6N2O2',  # Asparagine, Asn
    'E': 'C5H7NO3',  # Glutamic Acid, Glu
    'D': 'C4H5NO3',  # Aspartic Acid, Asp
    'S': 'C3H5NO2',  # Serine, Ser
    'T': 'C4H7NO2',  # Threonine, Thr
}

# Deoxynucleotide monophosphates - H2O
DEOXYNUCLEOTIDES = {
    'A': 'C10H12N5O5P',
    'T': 'C10H13N2O7P',
    'C': 'C9H12N3O6P',
    'G': 'C10H12N5O6P',
    'complements': {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'},
}

# Nucleotide monophosphates - H2O
NUCLEOTIDES = {
    'A': 'C10H12N5O6P',
    'U': 'C9H11N2O8P',
    'C': 'C9H12N3O7P',
    'G': 'C10H12N5O7P',
    'complements': {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'},
}

# # Formula preprocessors
# PREPROCESSORS = {
#     'peptide': from_peptide,
#     'ssdna': lambda x: from_oligo(x, 'ssdna'),
#     'dsdna': lambda x: from_oligo(x, 'dsdna'),
#     'ssrna': lambda x: from_oligo(x, 'ssrna'),
#     'dsrna': lambda x: from_oligo(x, 'dsrna'),
# }



class Formula:
    """Chemical formula.

    Input string may contain only symbols of chemical elements and isotopes,
    parentheses, and numbers.
    Calculate various properties from formula string, such as hill notation,
    empirical formula, mass, elemental composition, and mass distribution.
    Raise FormulaError on errors.

    Examples
    --------
    >>> Formula('H2O')
    Formula('H2O')

    """

    def __init__(self, formula=''):
        self._formula = formula
        self._elements = split_formula(formula)



    def __str__(self):
        return self._formula

    def __repr__(self):
        return f"Formula('{self._formula}')"

    # def __mul__(self, number):
    #     """Return this formula repeated number times as new Formula.
    #
    #     Examples
    #     --------
    #     >>> Formula('H2O') * 2
    #     Formula('(H2O)2')
    #
    #     """
    #     if not isinstance(number, int) or number < 1:
    #         raise TypeError('can only multipy with positive number')
    #     return Formula(f'({self._formula}){number}')
    #
    # def __rmul__(self, number):
    #     """Return this formula repeated number times as new Formula.
    #
    #     Examples
    #     --------
    #     >>> 2 * Formula('H2O')
    #     Formula('(H2O)2')
    #
    #     """
    #     return self.__mul__(number)
    #
    # def __add__(self, other):
    #     """Add this and other formula and return as new Formula.
    #
    #     Examples
    #     --------
    #     >>> Formula("H2O") + Formula("H2O")
    #     Formula('(H2O)(H2O)')
    #
    #     """
    #     if not isinstance(other, Formula):
    #         raise TypeError('can only add Formula instance')
    #     return Formula(f'({self})({other})')
    #
    # def __sub__(self, other):
    #     """Subtract elements of other formula and return as new Formula.
    #
    #     Examples
    #     --------
    #     >>> Formula('H2O') - Formula('O')
    #     Formula('H2')
    #
    #     """
    #     if not isinstance(other, Formula):
    #         raise TypeError('can only subtract Formula instance')
    #     _elements = copy.deepcopy(self._elements)
    #     for symbol, isotopes in other._elements.items():
    #         if symbol not in _elements:
    #             raise ValueError(f'element {symbol} not in {self}')
    #         element = _elements[symbol]
    #         for massnumber, count in isotopes.items():
    #             if massnumber not in element:
    #                 raise ValueError(
    #                     f'element {massnumber}{symbol} not in {self}'
    #                 )
    #             element[massnumber] -= count
    #             if element[massnumber] < 0:
    #                 raise ValueError(
    #                     f'negative number of element {massnumber}{symbol}'
    #                 )
    #             if element[massnumber] == 0:
    #                 del element[massnumber]
    #             if not element:
    #                 del _elements[symbol]
    #     return Formula(from_elements(_elements))
    #
    # def _elements(self):
    #     """Return number of atoms and isotopes by element as dict.
    #
    #     Return type is dict{element symbol: dict{mass number: count}}
    #
    #     Examples
    #     --------
    #     >>> Formula('H')._elements
    #     {'H': {0: 1}}
    #     >>> pprint(Formula('[2H]2O')._elements)  # pprint sorts dict
    #     {'H': {2: 2}, 'O': {0: 1}}
    #
    #     """
    #     formula = self._formula
    #     if not formula:
    #         raise FormulaError('empty formula', formula, 0)
    #
    #     validchars = set('([{<123456789ABCDEFGHIKLMNOPRSTUVWXYZ')
    #
    #     if not formula[0] in validchars:
    #         raise FormulaError(
    #             f"unexpected character '{formula[0]}'", formula, 0
    #         )
    #
    #     validchars |= set(']})>0abcdefghiklmnoprstuy')
    #
    #     elements = {}
    #     ele = ''  # parsed element
    #     num = 0  # number
    #     level = 0  # parenthesis level
    #     counts = [1]  # parenthesis level multiplication
    #     i = len(formula)
    #     while i:
    #         i -= 1
    #         char = formula[i]
    #         if char not in validchars:
    #             raise FormulaError(f"unexpected character {char}'", formula, i)
    #         if char in '([{<':
    #             level -= 1
    #             if level < 0 or num != 0:
    #                 raise FormulaError(
    #                     "missing closing parenthesis ')]}>'", formula, i
    #                 )
    #         elif char in ')]}>':
    #             if num == 0:
    #                 num = 1
    #             level += 1
    #             if level > len(counts) - 1:
    #                 counts.append(0)
    #             counts[level] = num * counts[level - 1]
    #             num = 0
    #         elif char.isdigit():
    #             j = i
    #             while i and formula[i - 1].isdigit():
    #                 i -= 1
    #             num = int(formula[i : j + 1])
    #             if num == 0:
    #                 raise FormulaError('count is zero', formula, i)
    #         elif char.islower():
    #             if not formula[i - 1].isupper():
    #                 raise FormulaError(
    #                     f"unexpected character '{char}'", formula, i
    #                 )
    #             ele = char
    #         elif char.isupper():
    #             ele = char + ele
    #             if num == 0:
    #                 num = 1
    #             if ele not in ELEMENTS:
    #                 raise FormulaError(f"unknown symbol '{ele}'", formula, i)
    #             iso = ''
    #             j = i
    #             while i and formula[i - 1].isdigit():
    #                 i -= 1
    #                 iso = formula[i] + iso
    #             if iso and i and not formula[i - 1] in '([{<':
    #                 i = j
    #                 iso = ''
    #             if iso:
    #                 iso = int(iso)
    #                 if iso not in ELEMENTS[ele].isotopes:
    #                     raise FormulaError(
    #                         f"unknown isotope '{iso}{ele}'", formula, i
    #                     )
    #             else:
    #                 iso = 0
    #             number = num * counts[level]
    #             if ele in elements:
    #                 item = elements[ele]
    #                 if iso in item:
    #                     item[iso] += number
    #                 else:
    #                     item[iso] = number
    #             else:
    #                 elements[ele] = {iso: number}
    #             ele = ''
    #             num = 0
    #
    #     if num != 0:
    #         raise FormulaError('number preceding formula', formula, 0)
    #
    #     if level != 0:
    #         raise FormulaError(
    #             "missing opening parenthesis '([{<'", formula, 0
    #         )
    #
    #     if not elements:
    #         raise FormulaError('invalid formula', formula, 0)
    #
    #     return elements


    @property
    def empirical(self):
        """Return empirical formula in Hill notation.

        The empirical formula has the simplest whole number ratio of atoms
        of each element present in formula.

        Examples
        --------
        >>> Formula('H2O').empirical
        'H2O'
        >>> Formula('S4').empirical
        'S'
        >>> Formula('C6H12O6').empirical
        'CH2O'

        """
        rr = self.gcd()
        empiricalFormula = ""
        for element, amount in self._elements.items():
            if amount == 0:
                pass
            elif rr == amount:
                empiricalFormula += element
            else:
                empiricalFormula += element
                empiricalFormula += str(int(amount/rr))
        return empiricalFormula

    @property
    def atoms(self):
        """Return number of atoms.

        Examples
        --------
        >>> Formula('CH3COOH').atoms
        8

        """
        return sum(self._elements.values())

    def gcd(self):
        """Return greatest common divisor of element counts.

        Examples
        --------
        >>> Formula('H2').gcd
        2
        >>> Formula('H2O').gcd
        1
        >>> Formula('C6H12O6').gcd
        6

        """
        # return gcd(
        #     {list(i)[0] for i in (j.values() for j in self._elements.values())}
        # )
        return gcd(list(self._elements.values()))

    @property
    def mass(self):
        """Return average relative molecular mass.

        Sums the relative atomic masses of all atoms in molecule.
        Equals the molar mass in g/mol, i.e. the mass of one mole of substance.

        Examples
        --------
        >>> Formula('C').mass
        12.01074
        >>> Formula('12C').mass
        12.0
        >>> print('{:.2f}'.format(Formula('C48H32AgCuO12P2Ru4').mass))
        1438.40

        """
        if hasattr(self, "_mass") == False:
            result = 0.0
            for symbol in self._elements:
                ele = ELEMENTS[symbol]
                result += ele.mass * self._elements[symbol]
            self._mass = result
            return result
        else:
            return self._mass

    @mass.setter
    def mass(self, value):
        pass

    def isotope(self):
        """Return isotope composed of most abundant elemental isotopes.

        Examples
        --------
        >>> print(Formula('C').isotope.mass)
        12.0
        >>> Formula('13C').isotope.massnumber
        13
        >>> print(Formula('C48H32AgCuO12P2Ru4').isotope)
        1440, 1439.5890, 0.205075%

        """
        result = Isotope()
        for symbol in self._elements:
            ele = ELEMENTS[symbol]
            # for massnumber, count in self._elements[symbol].items():
            #     if massnumber:
            #         isotope = ele.isotopes[massnumber]
            #     else:
            #         isotope = ele.isotopes[ele.nominalmass]
            #     result.mass += isotope.mass * count
            #     result.massnumber += isotope.massnumber * count
            #     result.abundance *= isotope.abundance ** count
            result.mass += isotope.mass * count
            result.massnumber += isotope.massnumber * count
            result.abundance *= isotope.abundance ** count
        return result

    # def composition(self, isotopic=True):
    #     """Return elemental composition as Composition instance.
    #
    #     Return type is tuple(tuple(symbol, count, mass, fraction), ).
    #
    #     If isotopic is True, isotopes specified in the formula are listed
    #     separately, otherwise they are listed as part of an element.
    #
    #     Examples
    #     --------
    #     >>> Formula('[12C]C').composition(False)
    #     (('C', 2, 24.01074, 1.0),)
    #     >>> for i in Formula('[12C]C').composition(True): print(i)
    #     ('C', 1, 12.01074, 0.5002236499166623)
    #     ('12C', 1, 12.0, 0.49977635008333776)
    #
    #     """
    #     elements = self._elements
    #     result = []
    #     if isotopic:
    #         for symbol in hill_sorted(elements):
    #             ele = ELEMENTS[symbol]
    #             iso = elements[symbol]
    #             for massnumber in sorted(iso):
    #                 count = iso[massnumber]
    #                 if massnumber:
    #                     mass = ele.isotopes[massnumber].mass * count
    #                     symbol = f'{massnumber}{symbol}'
    #                 else:
    #                     mass = ele.mass * count
    #                 result.append((symbol, count, mass, mass / self.mass))
    #     else:
    #         for symbol in hill_sorted(elements):
    #             ele = ELEMENTS[symbol]
    #             mass = 0.0
    #             counter = 0
    #             for massnumber, count in elements[symbol].items():
    #                 counter += count
    #                 if massnumber:
    #                     mass += ele.isotopes[massnumber].mass * count
    #                 else:
    #                     mass += ele.mass * count
    #             result.append((symbol, counter, mass, mass / self.mass))
    #     return Composition(result)

    # def spectrum(self, minfract=1e-9):
    #     """Return low resolution mass spectrum as Spectrum instance.
    #
    #     Return type is dict{massnumber: list[mass, fraction]}.
    #
    #     Calculated by combining the mass numbers of the elemental isotopes.
    #
    #     Examples
    #     --------
    #     >>> def _(spectrum):
    #     ...     for key, val in spectrum.items():
    #     ...         print(f'{key}, {val[0]:.4f}, {val[1]*100:.6f}%')
    #     >>> _(Formula('D').spectrum())
    #     2, 2.0141, 100.000000%
    #     >>> _(Formula('H').spectrum())
    #     1, 1.0078, 99.988500%
    #     2, 2.0141, 0.011500%
    #     >>> _(Formula('D2').spectrum())
    #     4, 4.0282, 100.000000%
    #     >>> _(Formula('DH').spectrum())
    #     3, 3.0219, 99.988500%
    #     4, 4.0282, 0.011500%
    #     >>> _(Formula('H2').spectrum())
    #     2, 2.0157, 99.977001%
    #     3, 3.0219, 0.022997%
    #     4, 4.0282, 0.000001%
    #     >>> _(Formula('DHO').spectrum())
    #     19, 19.0168, 99.745528%
    #     20, 20.0215, 0.049468%
    #     21, 21.0211, 0.204981%
    #     22, 22.0274, 0.000024%
    #
    #     """
    #     spectrum = {0: [0.0, 1.0]}
    #     elements = self._elements
    #
    #     for symbol in elements:
    #         ele = ELEMENTS[symbol]
    #         for massnumber, count in elements[symbol].items():
    #             if massnumber:
    #                 # specific isotope
    #                 iso = ele.isotopes[massnumber]
    #                 for key in reversed(sorted(spectrum)):
    #                     t = spectrum[key]
    #                     del spectrum[key]
    #                     if t[1] < minfract:
    #                         continue
    #                     f = t[1]
    #                     m = t[0] + iso.mass * count
    #                     k = key + iso.massnumber * count
    #                     if k in spectrum:
    #                         s = spectrum[k]
    #                         s[0] += (s[1] * s[0] + f * m) / (s[1] + f)
    #                         s[1] += f
    #                     else:
    #                         spectrum[k] = [m, f]
    #             else:
    #                 # mixture of isotopes
    #                 isotopes = ele.isotopes.values()
    #                 for _ in range(count):
    #                     for key in reversed(sorted(spectrum)):
    #                         t = spectrum[key]
    #                         del spectrum[key]
    #                         if t[1] < minfract:
    #                             continue
    #                         for iso in isotopes:
    #                             f = t[1] * iso.abundance
    #                             m = t[0] + iso.mass
    #                             k = key + iso.massnumber
    #                             if k in spectrum:
    #                                 s = spectrum[k]
    #                                 s[0] = (s[1] * s[0] + f * m) / (s[1] + f)
    #                                 s[1] += f
    #                             else:
    #                                 spectrum[k] = [m, f]
    #     return Spectrum(spectrum)

    # def spectrum(self, minfract=1e-9):
    #     """Return low resolution mass spectrum as Spectrum instance.
    #
    #     Return type is dict{massnumber: list[mass, fraction]}.
    #
    #     Calculated by combining the mass numbers of the elemental isotopes.
    #
    #     Examples
    #     --------
    #     >>> def _(spectrum):
    #     ...     for key, val in spectrum.items():
    #     ...         print(f'{key}, {val[0]:.4f}, {val[1]*100:.6f}%')
    #     >>> _(Formula('D').spectrum())
    #     2, 2.0141, 100.000000%
    #     >>> _(Formula('H').spectrum())
    #     1, 1.0078, 99.988500%
    #     2, 2.0141, 0.011500%
    #     >>> _(Formula('D2').spectrum())
    #     4, 4.0282, 100.000000%
    #     >>> _(Formula('DH').spectrum())
    #     3, 3.0219, 99.988500%
    #     4, 4.0282, 0.011500%
    #     >>> _(Formula('H2').spectrum())
    #     2, 2.0157, 99.977001%
    #     3, 3.0219, 0.022997%
    #     4, 4.0282, 0.000001%
    #     >>> _(Formula('DHO').spectrum())
    #     19, 19.0168, 99.745528%
    #     20, 20.0215, 0.049468%
    #     21, 21.0211, 0.204981%
    #     22, 22.0274, 0.000024%
    #
    #     """
    #     spectrum = {0: [0.0, 1.0]}
    #
    #     gens = []
    #     for symbol in self._elements:
    #         ele = ELEMENTS[symbol]
    #         gens.append(itertools.combinations_with_replacement(list(ele.isotopes.values()), self._elements[symbol]))
    #         # gens.append(list(itertools.combinations_with_replacement(list(ele.isotopes.values()), self._elements[symbol])))
    #
    #     reses = list(itertools.product(*gens))
    #
    #     # return Spectrum(spectrum)
    #
    #     return reses

    def spectrum(self, minfract=1e-9):
        """Return low resolution mass spectrum as Spectrum instance.

        Return type is dict{massnumber: list[mass, fraction]}.

        Calculated by combining the mass numbers of the elemental isotopes.

        Examples
        --------
        >>> def _(spectrum):
        ...     for key, val in spectrum.items():
        ...         print(f'{key}, {val[0]:.4f}, {val[1]*100:.6f}%')
        >>> _(Formula('D').spectrum())
        2, 2.0141, 100.000000%
        >>> _(Formula('H').spectrum())
        1, 1.0078, 99.988500%
        2, 2.0141, 0.011500%
        >>> _(Formula('D2').spectrum())
        4, 4.0282, 100.000000%
        >>> _(Formula('DH').spectrum())
        3, 3.0219, 99.988500%
        4, 4.0282, 0.011500%
        >>> _(Formula('H2').spectrum())
        2, 2.0157, 99.977001%
        3, 3.0219, 0.022997%
        4, 4.0282, 0.000001%
        >>> _(Formula('DHO').spectrum())
        19, 19.0168, 99.745528%
        20, 20.0215, 0.049468%
        21, 21.0211, 0.204981%
        22, 22.0274, 0.000024%

        """

        gens = []
        for symbol in self._elements:
            if self._elements[symbol] != 0:
                ele = ELEMENTS[symbol]
                gens.append(np.asarray(list(itertools.combinations_with_replacement(np.asarray(list(ele.isotopesNP.values())), self._elements[symbol]))))

        combos = np.apply_along_axis(np.vstack, 1, np.asarray(list(itertools.product(*gens))))

        masses = np.sum(combos[:, :, 0], axis=1)
        abundances = np.prod(combos[:, :, 1], axis=1)
        nominal = np.sum(combos[:, :, 2], axis=1)

        return Spectrum((masses, abundances, nominal, combos))

def getElement(element):
    if element in ELEMENTS:
        elementDict = {
            'atmrad': ELEMENTS[element].atmrad,
            'block': ELEMENTS[element].block,
            'covrad': ELEMENTS[element].covrad,
            'density': ELEMENTS[element].density,
            'eleaffin': ELEMENTS[element].eleaffin,
            'eleconfig': ELEMENTS[element].eleconfig,
            'electrons': ELEMENTS[element].electrons,
            'eleneg': ELEMENTS[element].eleneg,
            'group': ELEMENTS[element].group,
            'ionenergy': ELEMENTS[element].ionenergy,
            'isotopes': list([(x.mass, x.abundance, x.massnumber) for x in ELEMENTS[element].isotopes.values()]),
            'mass': ELEMENTS[element].mass,
            'name': ELEMENTS[element].name,
            'number': ELEMENTS[element].number,
            'oxistates': ELEMENTS[element].oxistates,
            'period': ELEMENTS[element].period,
            'protons': ELEMENTS[element].protons,
            'series': ELEMENTS[element].series,
            'symbol': ELEMENTS[element].symbol,
            'tboil': ELEMENTS[element].tboil,
            'tmelt': ELEMENTS[element].tmelt,
            'vdwrad': ELEMENTS[element].vdwrad,
        }
        return elementDict
    else:
        return None