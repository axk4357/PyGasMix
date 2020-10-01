from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from PyGasMix.Gas cimport Gas
from cython.parallel import prange
cimport GasUtil

sys.path.append('../hdf5_python')
import cython
import os

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)

cdef void Gas_C2F6(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for C2F4 gas.
    """
    gd = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"gases.npy"),allow_pickle=True).item()

    cdef double XENM[56], YXMOM[56], XENT[56], YXTOT[56], XVIB2[22], YVIB2[22], XVIB3[22], YVIB3[22], XVIB4[22], YVIB4[22], XVIB5[22], YVIB5[22], XVIB6[22], YVIB6[22], XDISS[27], YDISS[27], XATT[26], YATT[26], XION[48], YION[48]

    cdef int NDATA,NVIB3,NVIB4,NVIB5,NVIB6,N_IonizationD,N_Attachment1,NEXC,NEXC1 
    XENM=gd['gas_C2F6/XENM']
    YXMOM=gd['gas_C2F6/YXMOM']
    XENT=gd['gas_C2F6/XENT']
    YXTOT=gd['gas_C2F6/YXTOT']
    XVIB2=gd['gas_C2F6/XVIB2']
    YVIB2=gd['gas_C2F6/YVIB2']
    XVIB3=gd['gas_C2F6/XVIB3']
    YVIB3=gd['gas_C2F6/YVIB3']
    XVIB4=gd['gas_C2F6/XVIB4']
    YVIB4=gd['gas_C2F6/YVIB4']
    XVIB5=gd['gas_C2F6/XVIB5']
    YVIB5=gd['gas_C2F6/YVIB5']
    XVIB6=gd['gas_C2F6/XVIB6']
    YVIB6=gd['gas_C2F6/YVIB6']
    XDISS=gd['gas_C2F6/XDISS']
    YDISS=gd['gas_C2F6/YDISS']
    XATT=gd['gas_C2F6/XATT']
    YATT=gd['gas_C2F6/YATT']
    XION=gd['gas_C2F6/XION']
    YION=gd['gas_C2F6/YION']
    object.EnergyLevels = gd['gas_C2F6/EnergyLevels']

#———————————————————————————————————————
    cdef double AVIB1, AVIB2
    cdef int I, J
    cdef double EN
    object.N_Ionization = 1
    object.N_Attatchment = 1
    object.N_Inelastic = 0
    object.N_Null = 0
    object.N_ionModel = 0

    for J in range(6):
         object.AngularModel[J] = 0

    for J in range(object.N_Inelastic):
         object.KIN[J] = 0

    NDATA=56
    NETOT=56                                                        
    NVIB2=22                                                          
    NVIB3=22                                                          
    NVIB4=22                                                          
    NVIB5=22                                                          
    NVIB6=22
    NDISS=27
    NATT1=26
    NIOND=48   

    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921-27

    object.E = [0.0, 1.0, <float>(14.48), 0.0, 0.0, 0.0]
    object.E[1] = <float>(2.0) * ElectronMass / (<float>(138.0118) * AMU)

    object.EOBY[0] = <float>(14.48)

    cdef double APOP, EFAC

    object.APOP = [1.0,1.0,1.0]
    object.APOP[0] = [exp(object.EnergyLevels[0]/object.ThermalEnergy)]
    object.APOP[1] = [exp(object.EnergyLevels[1]/object.ThermalEnergy)]
    object.APOP[2] = [exp(object.EnergyLevels[2]/object.ThermalEnergy)]

    EN = -1*object.EnergyStep/<float>(2.0)
    for I in range(4000):
        EN += object.EnergyStep
        object.Q[1][I] = GasUtil.CALIonizationCrossSectionREG(EN, NDATA, YION, XENM)

        object.Q[2][I] = 0.0
        if EN>object.E[2]:
            object.Q[2][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationD, YION, XION)


        object.Q[3][I] = 0
        object.AttachmentCrossSection[0][I] = 0.0
        if EN>XATT[0]:
            object.Q[3][I] = GasUtil.CALIonizationCrossSection(EN, N_Attachment1, YATT, XATT)*1e-5
            object.AttachmentCrossSection[0][I] = object.Q[3][I]

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        #SUPERELASTIC OF VIBRATION
        object.InelasticCrossSectionPerGas[0][I] = 0.0
        if EN != 0.0:
            EFAC = sqrt(1.0 - (object.EnergyLevels[0]/EN))
            object.InelasticCrossSectionPerGas[0][I] = AVIB1 *log((EFAC+<float>(1.0))/(EFAC-<float>(1.0)))/EN
            object.InelasticCrossSectionPerGas[0][I]  = object.InelasticCrossSectionPerGas[0][I] * APOP/(<float>(1.0)+APOP)*1e-16

        object.InelasticCrossSectionPerGas[1][I] = 0.0
        if EN >object.EnergyLevels[1]:
            EFAC = sqrt(<float>(1.0) - (object.EnergyLevels[1]/EN))
            object.InelasticCrossSectionPerGas[1][I] = AVIB1 *log((EFAC+<float>(1.0))/(<float>(1.0)-EFAC))/EN
            object.InelasticCrossSectionPerGas[1][I]  = object.InelasticCrossSectionPerGas[1][I] * 1/(<float>(1.0)+APOP)*1e-16

        object.InelasticCrossSectionPerGas[2][I] = 0.0
        if EN >object.EnergyLevels[2]:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALPQ3(EN, NVIB3, YVIB3, XVIB3)
            EFAC = sqrt(1.0 - (object.EnergyLevels[2]/EN))
            object.InelasticCrossSectionPerGas[2][I] = (object.InelasticCrossSectionPerGas[2][I]+ AVIB2 *log((EFAC+<float>(1.0))/(<float>(1.0)-EFAC))/EN)*1e-16

        object.InelasticCrossSectionPerGas[3][I] = 0.0
        if EN >object.EnergyLevels[3]:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB4, YVIB4, XVIB4)

        object.InelasticCrossSectionPerGas[4][I] = 0.0
        if EN >object.EnergyLevels[4]:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB5, YVIB5, XVIB5)

        object.InelasticCrossSectionPerGas[5][I] = 0.0
        if EN >object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALIonizationCrossSectionREG(EN, NVIB6, YVIB6, XVIB6)

        object.InelasticCrossSectionPerGas[6][I] = 0.0
        if EN >object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALIonizationCrossSectionREG(EN, NEXC, YDISS, XDISS)
            
        object.InelasticCrossSectionPerGas[7][I] = 0.0
        if EN >object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALIonizationCrossSectionREG(EN, NEXC1, YDISS, XDISS)

        object.Q[0][I] =0.0
        for J in range(1,4):
            object.Q[0][I]+=object.Q[J][I]

        for J in range(8):
            object.Q[0][I]+=object.InelasticCrossSectionPerGas[J][I]


    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
    return