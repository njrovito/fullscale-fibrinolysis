import numpy as np
import matplotlib.pyplot as plt
import fibrinolysis_reactionLib as fibrxn # library file with systems of equations
import fibrinolysis_initializationLib as fibinit # library file with initialization values
# import vtkFiberVisualize.py as fibvis # library file with fiber-oriented vtk visulaization library

'''
Library of numpy-based commands to solve chemical reactions within and
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
    fiberLyse_Radial(constDict, n_tot_n, n_tot_0, V_0, fibLength)
        calculates change in fiber radius as a function of fractional lysis extent
    fibSolver_eulerF(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, R_tPA_n, R_PLG_n, R_PLS_n, R_AP_n, t, dt):
        INCOMPLETE comprehensive forward euler solver for all system components
    compModelSolver_eulerF(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, t, dt)
        Forward euler solver for free-phase concentrations
    fibrinSolver_EulerF(constDict, n_tot_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, C_tPA_n, C_PLG_n, C_PLS_n, dt)
        Forward euler solver for fibrin-bound concentrations
    RtermSolver_EulerF(constDict, R_tPA_n, R_PLG_n, R_PLS_n, R_AP_n, n_tot_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, C_tPA_n, C_PLG_n, C_PLS_n, C_AP_n, dt):
        Forward euler solver for reaction terms for ADR equations
    compModelSolver_RK4(a_constDict, t, dt, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, n_tot_n=0.0, n_tPA_n=0.0, n_PLG_n=0.0, n_PLS_n=0.0, L_PLS_n=0.0, R_tPA_n=0.0, R_PLG_n=0.0, R_PLS_n=0.0, R_AP_n=0.0, solveN=False, solveR=False):
        4th order runge-kutta solver for full model, with toggles for bound protein and reaction term solvers
    reactionTerms(constDict, ntPA, nPLG, nPLS, LPLS, ntot, CtPA, CPLG, CPLS, CAP)
    extentLysis(constDict, ntot, ntot_0)
    fibrinFromPorosity(constDict,phi)
    porosityFromFibrin(constDict,ntot)

'''

def fiberLyse_Radial(constDict, n_tot_n, n_tot_0, V_0, fibLength):
    '''
    calculates change in fiber radius as a function of fractional lysis extent
    '''

    E = 1 - n_tot_n/n_tot_0
    V = V_0 * (1-E)
    R = np.sqrt(V/np.pi/fibLength)

    return np.array([E, V, R])

## FORWARD EULER SOLVERS ##
def fibSolver_eulerF(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, R_tPA_n, R_PLG_n, R_PLS_n, R_AP_n, t, dt):
    '''
    Full set of equations condensed down
    NOT COMPLETE DON'T USE THIS YET
    '''
    C_tPA, C_PLG, C_PLS, C_AP, C_Fbg = compModelSolver_eulerF(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, t, dt)



    return


def compModelSolver_eulerF(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, t, dt):
    '''
    Forward euler solver for free-phase concentrations
    '''
    C_tPA = C_tPA_n + dt*fibrxn.dC_tPA(constDict,C_tPA_n,I)

    C_PLG = C_PLG_n + dt*fibrxn.dC_PLG(constDict,C_tPA_n,C_PLG_n)

    conc = fibrxn.concentrations(constDict,C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n)

    C_PLS = C_PLS_n + dt*conc[0]
    C_Fbg = C_Fbg_n + dt*conc[1]
    C_AP  = C_AP_n  + dt*conc[2]

    return np.array([C_tPA, C_PLG, C_PLS, C_Fbg, C_AP])



def fibrinSolver_EulerF(constDict, n_tot_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, C_tPA_n, C_PLG_n, C_PLS_n, dt):
    '''
    Forward euler solver for fibrin-bound concentrations
    '''
    dn = fibrxn.boundProteins(constDict, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, C_tPA_n, C_PLG_n, C_PLS_n)
    n_tot = n_tot_n + dt*dn[4]
    n_tPA = n_tPA_n + dt*dn[0]
    n_PLG = n_PLG_n + dt*dn[1]
    n_PLS = n_PLS_n + dt*dn[2]
    L_PLS = L_PLS_n + dt*dn[3]

    return np.array([n_tot, n_tPA, n_PLG, n_PLS, L_PLS])

def RtermSolver_EulerF(constDict, R_tPA_n, R_PLG_n, R_PLS_n, R_AP_n, n_tot_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, C_tPA_n, C_PLG_n, C_PLS_n, C_AP_n, dt):
    '''
    Forward euler solver for reaction terms for ADR equations
    '''
    dR = fibrxn.reactionTerms(constDict, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, C_tPA_n, C_PLG_n, C_PLS_n, C_AP_n)
    R_tPA = R_tPA_n + dt*dR[0]
    R_PLG = R_PLG_n + dt*dR[1]
    R_PLS = R_PLS_n + dt*dR[2]
    R_AP  = R_AP_n  + dt*dR[3]

    return np.array([R_tPA, R_PLG, R_PLS, R_AP])

## RUNGE-KUTTA 4th ORDER SOLVERS ##

# def fibSolver_RK4(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, R_tPA_n, R_PLG_n, R_PLS_n, R_AP_n, t, dt):
#     '''
#     Full set of equations condensed down
#     NOT COMPLETE DON'T USE THIS YET
#     '''
#     C_tPA, C_PLG, C_PLS, C_AP, C_Fbg = compModelSolver_eulerF(constDict, I, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, t, dt)
#
#
#
#     return


def compModelSolver_RK4(a_constDict, t, dt, C_tPA_n, C_PLG_n, C_PLS_n, C_Fbg_n, C_AP_n, n_tot_n=0.0, n_tPA_n=0.0, n_PLG_n=0.0, n_PLS_n=0.0, L_PLS_n=0.0, R_tPA_n=0.0, R_PLG_n=0.0, R_PLS_n=0.0, R_AP_n=0.0, solveN=False, solveR=False):
    '''
    4th order runge-kutta solver for full model, with toggles for bound protein and reaction term solvers
    '''
    constDict = a_constDict

    h = float(dt)

    I_0 = fibrxn.infusionEZCall(constDict,t)
    I_M = fibrxn.infusionEZCall(constDict,t+(h/2))
    I_1 = fibrxn.infusionEZCall(constDict,t+h)

    # RK for tPAa

    k1_tPA = fibrxn.dC_tPA(constDict,C_tPA_n,I_0)
    tPA_1 = C_tPA_n + k1_tPA*(h/2)
    k2_tPA = fibrxn.dC_tPA(constDict,tPA_1,I_M)
    tPA_2 = C_tPA_n + k2_tPA*(h/2)
    k3_tPA = fibrxn.dC_tPA(constDict,tPA_2, I_M)
    tPA_3 = C_tPA_n + k3_tPA*h
    k4_tPA = fibrxn.dC_tPA(constDict,tPA_3,I_1)

    C_tPA = C_tPA_n + 1/6*(k1_tPA + 2*k2_tPA + 2*k3_tPA + k4_tPA)*h

    k1_PLG = fibrxn.dC_PLG(constDict,C_tPA_n,C_PLG_n)
    PLG_1 = C_PLG_n + k1_PLG*(h/2)
    k2_PLG = fibrxn.dC_PLG(constDict,tPA_1,PLG_1)
    PLG_2 = C_PLG_n + k2_PLG*(h/2)
    k3_PLG = fibrxn.dC_PLG(constDict,tPA_2,PLG_2)
    PLG_3 = C_PLG_n + k3_PLG*h
    k4_PLG = fibrxn.dC_PLG(constDict,tPA_3,PLG_3)

    C_PLG = C_PLG_n + 1/6*(k1_PLG + 2*k2_PLG + 2*k3_PLG + k4_PLG)*h

    conc_0 = np.array([C_PLS_n, C_Fbg_n, C_AP_n])

    k1_conc = fibrxn.concentrations(constDict,C_tPA_n,C_PLG_n,conc_0[0],conc_0[1],conc_0[2])
    conc_1 = conc_0 + (h/2)*k1_conc
    k2_conc = fibrxn.concentrations(constDict,tPA_1,PLG_1,conc_1[0],conc_1[1],conc_1[2])
    conc_2 = conc_0 + (h/2)*k2_conc
    k3_conc = fibrxn.concentrations(constDict,tPA_2,PLG_2,conc_2[0],conc_2[1],conc_2[2])
    conc_3 = conc_0 + h*k3_conc
    k4_conc = fibrxn.concentrations(constDict,tPA_3,PLG_3,conc_3[0],conc_3[1],conc_3[2])

    conc_final =  conc_0 + 1/6*h*(k1_conc + 2*k2_conc + 2*k3_conc + k4_conc)

    C_PLS = conc_final[0]
    C_Fbg = conc_final[1]
    C_AP  = conc_final[2]

    if solveN == True:
        bindSites_0 = np.array([n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n])

        k1_bind = fibrxn.boundProteins(constDict, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, C_tPA_n, C_PLG_n, C_PLS_n)
        bindSites_1 = bindSites_0 +(h/2)*k1_bind
        k2_bind = fibrxn.boundProteins(constDict, bindSites_1[0], bindSites_1[1], bindSites_1[2], bindSites_1[3], bindSites_1[4], tPA_1, PLG_1, conc_1[0])
        bindSites_2 = bindSites_0 +(h/2)*k2_bind
        k3_bind = fibrxn.boundProteins(constDict, bindSites_2[0], bindSites_2[1], bindSites_2[2], bindSites_2[3], bindSites_2[4], tPA_2, PLG_2, conc_2[0])
        bindSites_3 = bindSites_0 +(h)*k3_bind
        k4_bind = fibrxn.boundProteins(constDict, bindSites_3[0], bindSites_3[1], bindSites_3[2], bindSites_3[3], bindSites_3[4], tPA_3, PLG_3, conc_3[0])

        bindSites_final = bindSites_0 + 1/6*h*(k1_bind + 2*k2_bind + 2*k3_bind + k4_bind)

        n_tPA = bindSites_final[0]
        n_PLG = bindSites_final[1]
        n_PLS = bindSites_final[2]
        L_PLS = bindSites_final[3]
        n_tot = bindSites_final[4]

        if solveR == True:
            R_terms_0 = np.array([R_tPA_n, R_PLG_n, R_PLS_n, R_AP_n])

            k1_rxn = fibrxn.reactionTerms(constDict, n_tPA_n, n_PLG_n, n_PLS_n, L_PLS_n, n_tot_n, C_tPA_n, C_PLG_n, C_PLS_n, C_AP_n)
            R_terms_1 = R_terms_0 + (h/2)*k1_rxn
            k2_rxn = fibrxn.reactionTerms(constDict, bindSites_1[0], bindSites_1[1], bindSites_1[2], bindSites_1[3], bindSites_1[4], tPA_1, PLG_1, conc_1[0], conc_1[2])
            R_terms_2 = R_terms_0 + (h/2)*k2_rxn
            k3_rxn = fibrxn.reactionTerms(constDict, bindSites_2[0], bindSites_2[1], bindSites_2[2], bindSites_2[3], bindSites_2[4], tPA_2, PLG_2, conc_2[0], conc_2[2])
            R_terms_3 = R_terms_0 + h*k3_rxn
            k4_rxn = fibrxn.reactionTerms(constDict, bindSites_3[0], bindSites_3[1], bindSites_3[2], bindSites_3[3], bindSites_3[4], tPA_3, PLG_3, conc_3[0], conc_3[2])

            R_terms_final = R_terms_0 + 1/6*h*(k1_rxn + 2*k2_rxn + 2*k3_rxn + k4_rxn)

            R_tPA = R_terms_final[0]
            R_PLG = R_terms_final[1]
            R_PLS = R_terms_final[2]
            R_AP  = R_terms_final[3]

            return np.array([C_tPA, C_PLG, C_PLS, C_Fbg, C_AP, n_tot, n_tPA, n_PLG, n_PLS, L_PLS, R_tPA, R_PLG, R_PLS, R_AP])

        return np.array([C_tPA, C_PLG, C_PLS, C_Fbg, C_AP, n_tot, n_tPA, n_PLG, n_PLS, L_PLS])

    if solveR == True and solveN == False:
        print('ERROR: Cannot solve reaction terms without local binding site calculations')
        return

    return np.array([C_tPA, C_PLG, C_PLS, C_Fbg, C_AP])
