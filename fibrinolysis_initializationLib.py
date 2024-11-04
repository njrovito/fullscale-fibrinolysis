import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

'''
Library of numpy-based commands to initialize data structures and values needed for
chemical reactions within and related to the fibrinolysis reaction cascade

REQUIRES A DICTIONARY OF SPECIES CONSTANTS AND PARAMETERS -- constants included as comments at
the end of library in case of lost dict file

Initially written by LS Nast

Included Functions:
    readInputs(a_FileName) = data
        Reads an xml formatted input file and converts to a dictionary
    printInputs(a_Data)
        Prints a formatted display of all inputs in a dictionary
    initFibrinSites(constDict,rho_0 = 3e-3) = ntot_0
        calculates initial number of molecular binding sites for a given fiber density
    initFibrinFromPorosity(constDict,phi) = ntot_0
        calculates concentration of molecular binding sites, scaled by local porosity
    initArray(constDict,array_sym,array_sub,nt) = C
        Initializes array C of concentration values at initial condition

Reactions and fibrin density adapted from
Piebalgs, A., Gu, B., Roi, D. et al. Computational Simulations of Thrombolytic
Therapy in Acute Ischaemic Stroke. Sci Rep 8, 15810 (2018).
https://doi.org/10.1038/s41598-018-34082-7

'''


def readInputs(a_FileName):
    """Read an xml formatted input file for setting up the problem

    Args:
        a_FileName (string): name of the xml formatted input file

    Returns:
        data (dict): Python dictionary with all input data
    """
    fileTree    = ET.parse(a_FileName)
    dataRoot    = fileTree.getroot()
    setupDict   = {}

    for element in dataRoot:
        for entry in element:
            data = entry.text.strip()
            if entry.attrib['dtype'] == 'Boolean':
                if data == 'True':
                    setupDict[entry.attrib['name']] = True
                elif data == 'False':
                    setupDict[entry.attrib['name']] = False
            elif entry.attrib['dtype'] == 'String':
                setupDict[entry.attrib['name']] = data
            elif entry.attrib['dtype'] == 'Float':
                setupDict[entry.attrib['name']] = float(data.strip())
            elif entry.attrib['dtype'] == 'Int':
                setupDict[entry.attrib['name']] = int(data.strip())

    return setupDict

def printInputs(a_Data):
    """Prints a formatted display of all inputs for the problem

    Args:
        a_Data (dict): dictionary containing all the input data defining the problem

    Returns:
        none
    """

    for key, value in a_Data.items():
        print('Input Entry:', key, 'Value:', value)
    return

def initFibrinSites(constDict,rho_0 = 3e-3):
    """
    calculates initial number of molecular binding sites for a given fiber density

    Returns
    -------
    ntot_0 : float
        concentration of molecular binding sites.

    """

    N_av = 6.02214076e23 # Avogadro's number [atoms/mol]

    Lm = 0.045 # length of repeating unit of fibrin [um]
    Rf0 = 0.100 # average fiber radius [um]
    del_r = 0.010 # radial interprotofibril distance [um]
    del_t = 0.010 # tangential interprotofibril distance [um]
    rho_f = 0.245 # fibre density [g/mL]

    N_L = Rf0 / del_r # number of layers of protofibrils in cross-section

    N_pf_tot = 0
    for i in range(1, int(N_L)):
        N_pf_tot = N_pf_tot +  np.pi/np.arcsin(del_t / (2*i*del_r))

    n_bs = 2 # num. binding sites per slice (assumed)

    # N_pf_tot = 344 # number of protofibrils per slice

    Lf_tot = (rho_0/rho_f) / (np.pi * Rf0**2) # Length of fibre per clot volume [um/um**3]
    N_slice = Lf_tot / (Lm/2)    # Number of slices per clot volume

    ntot_0 = n_bs * N_slice*N_pf_tot/N_av *10**6 * 10**15# convert mol/um**3->uM

    return ntot_0

def initFibrinFromPorosity(constDict,phi):
    '''
    calculates concentration of molecular binding sites, scaled by local porosity
    '''

    R_f0 = constDict['fibrin_avg_radius']
    N_pf = constDict['num_protofibrils']
    M_f  = constDict['fibrin_monomer_length']
    n_bs = constDict['bindsites_per_monomer']
    N_av = 6.022e23

    v_frac = 1-phi

    ntot_0 = v_frac/(np.pi*R_f0**2) * 2/M_f * N_pf*n_bs/N_av

    return ntot_0

def initArray(constDict,array_sym,array_sub,nt):
    """
    Initializes array C of concentration values at initial condition (included
    because of default initial conditions included in this library, but can be
    substituted for alternative initial conditions as required)

    """
    if array_sym == 'c' or 'C':
        array_class = '_conc'
    elif array_sym == 'n'or'N':
        array_class = '_bindSites'
    elif array_sym == 'l'or'L':
        array_class = '_bindConc'
    else:
        print(array_sym+' is an invalid initialization symbol')
        return

    if array_sub == 'tPA':
        array_val = 'tPA'
    elif array_sub == 'PLG':
        array_val = 'Plasminogen'
    elif array_sub == 'PLS':
        array_val = 'Plasmin'
    elif array_sub == 'Fbg':
        array_val = 'Fibrinogen'
    elif array_sub == 'AP':
        array_val = 'Antiplasmin'
    else:
        print(array_sub+' is an invalid initialization subscript')
        return

    val_key = 'init_'+array_val+array_class
    C_0 = constDict[val_key]

    C = C_0*np.ones(nt) # initializes array of initial value to desired simulation length
    return C
