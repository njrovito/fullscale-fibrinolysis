import numpy as np
import matplotlib.pyplot as plt

'''
Library of numpy-based commands to calculate chemical reactions within and
related to the fibrinolysis reaction cascade

REQUIRES A DICTIONARY OF SPECIES CONSTANTS AND PARAMETERS -- constants included as comments at
the end of library in case of lost dict file

Included species are:
    tissue-type plasminogen activator (tPA) - activating enzyme (added through
        IV transfusion during clinical stroke treatment)
    plasminogen (PLG) - plasma-based protein which activates on exposure to tPA
    plasmin (PLS) - activated form of PLG and fibrin lytic agent
    antiplasmin (AP) - most prevalent plasmin inhibitor; found in plasma
    fibrinogen (Fbg) - inactive, plasma-phase fibrin monomers; optionally
        tracked for bleed risk assessment (through coagulation inhibition)
    fibrin - solid-phase blood clot structural material; lyses when exposed
        monomers are exposed to plasmin. Lysed fibrin monomers bound to plasmin
        can be tracked to calculate plasmin reuptake, if desired

Initially written by LS Nast

Included Functions:
    Infusion(constDict, t, alfa_tPA, t_B, t_D, t_C, f_B, f_C) = Dosage
        DEPRECIATED - allows for configuration of time delays in infusion rates
        during tPA treatment
    infusionEZCall(constDict, t) = Dosage
        Like Infusion(), but reads all time values from the input file
    dC_tPA(constDict, CtPA, I)
        Equation for calculating change in tPA concentration
    dC_PLG(constDict, CtPA, CPLG)
        Equation for calculating change in plasminogen concentration
    concentrations(constDict, CtPA, CPLG, CPLS, CFbg, CAP)
        System of equations for calculating change in plasmin, antiplasmin, and fibrinogen concentrations
    boundProteins(constDict, ntPA, nPLG, nPLS, LPLS, ntot, CtPA, CPLG, CPLS)
        System of equations for calculating change in concentrations for bound vs unbound proteins
    reactionTerms(constDict, ntPA, nPLG, nPLS, LPLS, ntot, CtPA, CPLG, CPLS, CAP)
        System of equations for calculating changes in reaction terms for ADR implementations
    extentLysis(constDict, ntot, ntot_0)
        Equation to calculate fraction of fibrin lysed at current time
    fibrinFromPorosity(constDict,phi)
        Equation to calculate fibrin binding site concentration from porosity data
    porosityFromFibrin(constDict,ntot)
        Equation to calculate local porosity from fibrin binding site concentration
    # depreciated global constants list #

Reactions adapted from
Piebalgs, A., Gu, B., Roi, D. et al. Computational Simulations of Thrombolytic
Therapy in Acute Ischaemic Stroke. Sci Rep 8, 15810 (2018).
https://doi.org/10.1038/s41598-018-34082-7

'''

'''
CHANGELOG
- 1/23/22 began documentation for functions
- 7/28/22 added continuum phi->ntot and ntot->phi conversions
'''


def Infusion(constDict, t, alfa_tPA, t_B, t_D, t_C, f_B, f_C):
    ## REWORK TO ALLOW PATIENT SPECIFIC INPUTS, PURELY CONTINUOUS DOSAGES
    """
    Returns an array of infusion rates of tPA into the bloodstream during
    thrombolytic therapy

    Parameters
    ----------
    alfa_tPA : float
        total dosage of tPA to be infused during therapy [mg/kg].
    t_B : float
        duration of initial bolus component of treatment [s].
    t_D : float
        duration of delay between bolus and continuous infusion [s].
    t_C : float
        duration of continuous infusion of tPA [s].
    f_B : float
        fraction of dose to be given during bolus.
    f_C : float
        fraction of dose to be given during continuous transfusion.
    t : 1-dimensional floating-point array
        array of time values, ranging from intitial time 0 to final time [s].

    Returns
    -------
    Dosage : 1-dimensional floating-point array
        array of infusion rate values at each time in t.

    """

    w = constDict['patient_mass']

    if t <= t_B:
        Dosage = f_B*alfa_tPA*w/t_B
    elif t > (t_B+t_D) and t <= (t_B+t_D+t_C):
        Dosage = f_C*alfa_tPA*w/t_C
    else:
        Dosage = 0

    return Dosage

def infusionEZCall(constDict, t):
    ## REWORK TO ALLOW PATIENT SPECIFIC INPUTS, PURELY CONTINUOUS DOSAGES
    """
    Returns an array of infusion rates of tPA into the bloodstream during
    thrombolytic therapy

    Parameters
    ----------
    alfa_tPA : float
        total dosage of tPA to be infused during therapy [mg/kg].
    t_B : float
        duration of initial bolus component of treatment [s].
    t_D : float
        duration of delay between bolus and continuous infusion [s].
    t_C : float
        duration of continuous infusion of tPA [s].
    f_B : float
        fraction of dose to be given during bolus.
    f_C : float
        fraction of dose to be given during continuous transfusion.
    t : 1-dimensional floating-point array
        array of time values, ranging from intitial time 0 to final time [s].

    Returns
    -------
    Dosage : 1-dimensional floating-point array
        array of infusion rate values at each time in t.

    """

    w = constDict['patient_mass']

    alfa_tPA = constDict['treatment_dosage']
    t_B = constDict['bolus_time']
    t_D = constDict['delay_time']
    t_C = constDict['infusion_time']
    f_B = constDict['bolus_fraction']
    f_C = 1-f_B

    if t <= t_B and t_B != 0:
        Dosage = f_B*alfa_tPA*w/t_B
    elif t > (t_B+t_D) and t <= (t_B+t_D+t_C):
        Dosage = f_C*alfa_tPA*w/t_C
    else:
        Dosage = 0

    return Dosage

def dC_tPA(constDict, CtPA, I):
    """
    Equation for calculating change in tPA concentration

    """
    # Hepatic Clearance Coefficients (derived from half-lives)
    k_HCtPA = np.log(2)/constDict['half_life_tPA']
    M_wtPA = constDict['Mol_wt_tPA']
    V_plasma = constDict['plasma_volume']

    return -k_HCtPA*CtPA + I/(M_wtPA*V_plasma)

def dC_PLG(constDict, CtPA, CPLG):
    """
    Equation for calculating change in plasminogen concentration


    """
    C_PLG_0 = constDict['init_Plasminogen_conc']
    k_2f = constDict['k_2_free_phase_Plasminogen']
    K_Mf = constDict['K_M_free_phase_Plasminogen']

    # Hepatic Clearance Coefficients (derived from half-lives)
    k_HCPLG = np.log(2)/constDict['half_life_Plasminogen']

    # Generation Terms (assumed constant and balanced with hepatic clearance rates at equilibrium)
    G_PLG = k_HCPLG*C_PLG_0
    return -k_HCPLG*CPLG - k_2f*CtPA*CPLG / (K_Mf + CPLG) + G_PLG

def concentrations(constDict, CtPA, CPLG, CPLS, CFbg, CAP):
    """
    System of equations for calculating change in plasmin, antiplasmin, and fibrinogen concentrations

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    CtPA : TYPE
        DESCRIPTION.
    CPLG : TYPE
        DESCRIPTION.
    CPLS : TYPE
        DESCRIPTION.
    CFbg : TYPE
        DESCRIPTION.
    CAP : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    k_2f = constDict['k_2_free_phase_Plasminogen']
    K_Mf = constDict['K_M_free_phase_Plasminogen']
    k_AP = constDict['k_Antiplasmin_inhibition']
    k_catf = constDict['k_Fibrinogen_degredation']

    # Hepatic Clearance Coefficients (derived from half-lives)
    k_HCPLS = np.log(2)/constDict['half_life_Plasmin']
    k_HCFbg = np.log(2)/constDict['half_life_Fibrinogen']
    k_HCAP  = np.log(2)/constDict['half_life_Antiplasmin']

    # Generation Terms (assumed constant and balanced with hepatic clearance rates at equilibrium)
    G_Fbg = k_HCFbg*constDict['init_Fibrinogen_conc']
    G_AP  = k_HCAP *constDict['init_Antiplasmin_conc']

    dC_PLS = -k_HCPLS*CPLS + k_2f*CtPA*CPLG / (K_Mf + CPLG) - k_AP*CAP*CPLS
    dC_Fbg = -k_HCFbg*CFbg - k_catf*CFbg*CPLS + G_Fbg
    dC_AP  = -k_HCAP*CAP - k_AP*CPLS*CAP + G_AP
    return np.array([dC_PLS, dC_Fbg, dC_AP])

def boundProteins(constDict, ntPA, nPLG, nPLS, LPLS, ntot, CtPA, CPLG, CPLS):
    """
    System of equations for calculating change in concentrations for bound vs unbound proteins

    Parameters
    ----------
    ntPA : TYPE
        DESCRIPTION.
    nPLG : TYPE
        DESCRIPTION.
    nPLS : TYPE
        DESCRIPTION.
    LPLS : TYPE
        DESCRIPTION.
    ntot : TYPE
        DESCRIPTION.
    CtPA : TYPE
        DESCRIPTION.
    CPLG : TYPE
        DESCRIPTION.
    CPLS : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    k2 = constDict['k_2_bound_Plasminogen']
    K_M = constDict['K_M_bound_Plasminogen']
    k_cat = constDict['k_fibrin_degredation']
    gamma = constDict['gamma_solubilisation']

    nfree = ntot - (ntPA + nPLG + nPLS)

    dntPA = constDict['k_adsorption_tPA']*CtPA*nfree - constDict['k_desorption_tPA']*ntPA
    dnPLG = constDict['k_adsorption_Plasminogen']*CPLG*nfree - constDict['k_desorption_Plasminogen']*nPLG - k2*nPLG*ntPA / (K_M+nPLG)
    dnPLS = constDict['k_adsorption_Plasmin']*CPLS*nfree - constDict['k_desorption_Plasmin']*nPLS + k2*nPLG*ntPA / (K_M+nPLG) \
        - k_cat*gamma*nPLS
    dLPLS = k_cat*gamma*nPLS - constDict['k_desorption_Plasmin']*LPLS
    dntot = -k_cat*gamma*nPLS

    return np.array([dntPA, dnPLG, dnPLS, dLPLS, dntot])

def reactionTerms(constDict, ntPA, nPLG, nPLS, LPLS, ntot, CtPA, CPLG, CPLS, CAP):
    """
    System of equations for calculating changes in reaction terms for ADR implementations

    Parameters
    ----------
    CtPA : TYPE
        DESCRIPTION.
    CPLS : TYPE
        DESCRIPTION.
    CPLG : TYPE
        DESCRIPTION.
    CAP : TYPE
        DESCRIPTION.
    ntPA : TYPE
        DESCRIPTION.
    nPLG : TYPE
        DESCRIPTION.
    nPLS : TYPE
        DESCRIPTION.
    LPLS : TYPE
        DESCRIPTION.
    ntot : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    nfree = ntot - (ntPA + nPLG + nPLS)

    RtPA = constDict['k_adsorption_tPA']*CtPA*nfree - constDict['k_desorption_tPA']*ntPA
    RPLG = constDict['k_adsorption_Plasminogen']*CPLG*nfree - constDict['k_desorption_Plasminogen']*nPLG
    RPLS = constDict['k_adsorption_Plasmin']*CPLS*nfree - constDict['k_desorption_Plasmin']*nPLS \
         + constDict['k_Antiplasmin_inhibition']*CAP*CPLS - constDict['k_desorption_Plasmin']*LPLS
    RAP  = constDict['k_Antiplasmin_inhibition']*CAP*CPLS

    return np.array([RtPA, RPLG, RPLS, RAP])

def extentLysis(constDict, ntot, ntot_0):
    """
    Equation to calculate fraction of fibrin lysed at current time

    Parameters
    ----------
    ntot : TYPE
        DESCRIPTION.
    ntot_0 : TYPE
        DESCRIPTION.

    Returns
    -------
    E : TYPE
        DESCRIPTION.

    """
    E = 1 - ntot/ntot_0
    return E

def fibrinFromPorosity(constDict,phi):
    '''
    Equation to calculate fibrin binding site concentration from porosity data
    '''
    R_f0 = constDict['fibrin_avg_radius']
    N_pf = constDict['num_protofibrils']
    M_f  = constDict['fibrin_monomer_length']
    n_bs = constDict['bindsites_per_monomer']
    N_av = 6.022e23

    v_frac = 1-phi

    ntot = v_frac/(np.pi*R_f0**2) * 2/M_f * N_pf*n_bs/N_av

    ntot = ntot*10**15 # converting from m3 volume to um3 volume standards

    return ntot

def porosityFromFibrin(constDict,ntot):
    """
    Equation to calculate local porosity from fibrin binding site concentration

    """
    R_f0 = constDict['fibrin_avg_radius']
    N_pf = constDict['num_protofibrils']
    M_f  = constDict['fibrin_monomer_length']
    n_bs = constDict['bindsites_per_monomer']
    N_av = 6.022e23

    ntot = ntot / 10**15 # converting from um3 volume to m3 volume standards

    v_frac = ntot*(np.pi*R_f0**2)/(2/M_f * N_pf*n_bs/N_av)

    phi    = 1 - v_frac

    return phi

'''
# BEGIN GLOBAL CONSTANTS -- DEPRECIATED BY INCORPORATING AN INPUT PARAMETER FILE

# Patient Values (Potentially convert to an input field?)
V_plasma = 3.9 # volume of plasma [L]
w = 80 # weight of patient [kg]

# Initial Conditions: protein concentrations throughout the body
C_AP_0 = 1 # initial concentration of antiplasmin [uM]
C_Fbg_0 = 8 #initial concentration of fibrinogen [uM] (included to trach ICH risk)
C_PLG_0 = 2.2 # initial concentration of plasminogen [uM]
C_PLS_0 = 0 # initial concentration of plasmin [uM]
C_tPA_0 = 0.07e-3 # initial concentration of tPA (pre-treatment) [uM]

# Kinetic Constants
k_2f = 0.3 # Michaelis-Menten constant [1/s]
K_Mf = 28  # Michaelis-Menten constant [uM]
k_AP = 10  # reaction constant for AP inhibition [(uM-s)^-1]
k_catf = 6 # reaction constant for fibrinogen degradation [(uM-s)^-1]

# Molecular Weights (as needed)
M_wtPA = 59.04 # of tPA [mg/umol]

# half-lives (in days -> sec)
t_hlf_AP = 2.64*24*3600  # AP
t_hlf_Fbg = 4.14*24*3600 # fibirnogen
t_hlf_PLG = 2.2*24*3600  # plasminogen
# in s
t_hlf_PLS = 0.1  # plasmin
# in min -> sec
t_hlf_tPA = 4*60 # tPA

# Hepatic Clearance Coefficients (derived from half-lives)
k_HCtPA = np.log(2)/t_hlf_tPA
k_HCPLG = np.log(2)/t_hlf_PLG
k_HCPLS = np.log(2)/t_hlf_PLS
k_HCFbg = np.log(2)/t_hlf_Fbg
k_HCAP  = np.log(2)/t_hlf_AP

# Generation Terms (assumed constant and balanced with hepatic clearance rates at equilibrium)
G_PLG = k_HCPLG*C_PLG_0
G_Fbg = k_HCFbg*C_Fbg_0
G_AP  = k_HCAP *C_AP_0

# Kinetic constants for reaction within fibrin structure
k2 = 0.3 # Michaleis reaction rate coefficient [1/s]
k_cat = 2.178 # lysis coefficient [1/s]
K_M = 0.16 # Michaelis constant [uM]
gamma = 1/10 # 1/cuts needed for plasmin to cut 1 fibrin unit

# adsorption coefficients
ka_PLG = 0.1 #[1/uM-s]
ka_PLS = 0.1 #[1/uM-s]
ka_tPA = 0.01 #[1/uM-s]

# desorption coefficients
kr_PLG = 3.8 #[1/s]
kr_PLS = 0.05 #[1/s]
kr_tPA = 0.0058 #[1/s]
'''
