import numpy as np
from numba import jit
from MyNumba import *

@jit(nopython=True)
def run_model(par_values,watershed_area,ini_values,forcing_data,long_term):
    # Unpack parameters:
    TT = par_values['TT']
    C0 = par_values['C0']
    ETF = par_values['ETF']
    LP = par_values['LP']
    FC = par_values['FC']
    beta = par_values['beta']
    FRAC = par_values['FRAC']
    K1 = par_values['K1']
    alpha = par_values['alpha']
    K2 = par_values['K2']
    UBAS=par_values['UBAS']
    PM=par_values['PM']

    LP = LP * FC;

    # Unpack initial conditions and forcing
    initial_SWE = ini_values[0]; initial_SMS = ini_values[1];
    initial_S1= ini_values[2];   initial_S2 = ini_values[3];

    P = PM * forcing_data['P']
    T = forcing_data['T']
    month_time_series = forcing_data['month_time_series']
    monthly_average_T = long_term['monthly_average_T']
    monthly_average_PE = long_term['monthly_average_PE']

    period_length = len(P)

    SWE=np.zeros(period_length+1)
    SWE[0] = initial_SWE;

    SMS=np.zeros(period_length+1)
    SMS[0] = initial_SMS;

    S1=np.zeros(period_length+1)
    S1[0] = initial_S1

    S2=np.zeros(period_length+1)
    S2[0] = initial_S2


    ponding=np.zeros(period_length)
    AET=np.zeros(period_length)
    PET=np.zeros(period_length)
    Q1=np.zeros(period_length)
    Q2=np.zeros(period_length)

    for t in range(period_length):

        SWE[t+1],ponding[t]=precipitation_module(SWE[t],P[t],T[t],TT,C0)

        AET[t],PET[t]=evapotranspiration_module(SMS[t],T[t],month_time_series[t],monthly_average_T,monthly_average_PE,ETF,LP)

        SMS[t+1],S1[t+1],S2[t+1],Q1[t],Q2[t]=soil_storage_routing_module(ponding[t], SMS[t],
                                                S1[t], S2[t], AET[t],
                                                FC, beta, FRAC, K1, alpha, K2)

    Q1_routed = triangle_routing(Q1, UBAS)
    Q = Q1_routed + Q2
    Q_cms=(Q*watershed_area*1000)/(24*3600)

    return Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2

    
@jit(nopython=True)
def precipitation_module( SWE, P, T, TT, C0):
# % *****  TT : Temperature Threshold or melting/freezing point - model parameter *****
# % *****  C0: base melt factor - model parameter *****
# % *****  P: Precipitation - model forcing *****
# % *****  T: Temperature - model forcing *****
# % *****  SWE: Snow Water Equivalent - model state variable *****
    if T >= TT:
        rainfall = P
        potential_snow_melt  = C0 * (T - TT)
        snow_melt = min((potential_snow_melt,SWE))
        ponding = rainfall + snow_melt # Liquid Water on Surface
        SWE_new = SWE - snow_melt # Soil Water Equivalent - Solid Water on Surface
    else:
        snowfall = P
        snow_melt = 0
        ponding = 0 # Liquid Water on Surface
        SWE_new = SWE + snowfall # Soil Water Equivalent - Solid Water on Surface

    return SWE_new, ponding


@jit(nopython=True)
def evapotranspiration_module(SMS,T,month_number, monthly_average_T,monthly_average_PE,ETF,LP):
# % *****  T: Temperature - model forcing *****
# % *****  month_number: the current month number - for Jan=1, ..., Dec=12 *****
# % *****  SMS: Soil Moisture Storage - model state variable *****
# % *****  ETF - This is the temperature anomaly correction of potential evapotranspiration - model parameters
# % *****  LP: This is the soil moisture content below which evaporation becomes supply-limited - model parameter
# % *****  PET: Potential EvapoTranspiration - model parameter
# % *****  AET: Actual EvapoTranspiration - model

    # Potential Evapotranspiration:
    PET = ( 1 + ETF * ( T - monthly_average_T[int(month_number)-1] ) ) * monthly_average_PE[int(month_number)-1]
    PET = max((PET, 0))

    if SMS > LP:
        AET = PET
    else:
        AET = PET * ( SMS / LP )

    AET = min((AET, SMS)) # to avoid evaporating more than water available

    return AET,PET


@jit(nopython=True)
def soil_storage_routing_module(ponding, SMS, S1, S2, AET, FC, beta, FRAC, K1, alpha, K2):
#     % *****  T: Temperature - model forcing *****
#     % *****  month_number: the current month number - for Jan=1, ..., Dec=12 *****
#     % *****  SMS: Soil Moisture Storage - model state variable *****
#     % *****  ETF - This is the temperature anomaly correction of potential evapotranspiration - model parameters
#     % *****  LP: This is the soil moisture content below which evaporation becomes supply-limited - model parameter
#     % *****  PET: Potential EvapoTranspiration - model parameter


#     % *****  FC: Field Capacity - model parameter ---------
#     % *****  beta: Shape Parameter/Exponent - model parameter ---------
#     % This controls the relationship between soil infiltration and soil water release.
#     % The default value is 1. Values less than this indicate a delayed response, while higher
#     % values indicate that runoff will exceed infiltration.

    if SMS < FC:
        soil_release = ponding * (( SMS / FC )**beta) # release of water from soil
    else:
        soil_release = ponding # release of water from soil

    SMS_new = SMS - AET + ponding - soil_release

#     % if SMS_new < 0 % this might happen due to very small numerical/rounding errors
#     %     SMS_new
#     %     SMS_new = 0;
#     % end

    soil_release_to_fast_reservoir = FRAC * soil_release
    soil_release_to_slow_reservoir = ( 1 - FRAC ) * soil_release

    Q1 = K1*S1**alpha
    if Q1>S1:
        Q1=S1

    S1_new = S1 + soil_release_to_fast_reservoir - Q1

    Q2 = K2 * S2

    S2_new = S2 + soil_release_to_slow_reservoir - Q2

    return SMS_new, S1_new, S2_new, Q1, Q2


@jit(nopython=True)
def triangle_routing(Q, UBAS):
    UBAS = max((UBAS, 0.1))
    length_triangle_base = int(np.ceil(UBAS))
    if UBAS == length_triangle_base:
        x = np.array([0, 0.5*UBAS, length_triangle_base])
        v = np.array([0, 1, 0])
    else:
        x = np.array([0, 0.5*UBAS, UBAS, length_triangle_base])
        v = np.array([0, 1, 0, 0])
    
    weight=np.zeros(length_triangle_base)

    for i in range(1,length_triangle_base+1):
        if (i-1) < (0.5 * UBAS) and i > (0.5 * UBAS):
            weight[i-1] = 0.5 * (np.interp(i - 1,x,v) + np.interp(0.5 * UBAS,x,v) ) * ( 0.5 * UBAS - i + 1) +  0.5 * ( np.interp(0.5 * UBAS,x,v) + np.interp(i,x,v) ) * ( i - 0.5 * UBAS )
        elif i > UBAS:
            weight[i-1] = 0.5 * ( np.interp(i-1,x,v) ) * ( UBAS - i + 1)
        else:
            weight[i-1] = np.interp(i-0.5,x,v)

    weight = weight/np.sum(weight)

    Q_routed=np.zeros(len(Q))
    for i in range(1,len(Q)+1):
        temp = 0
        for j in range(1,1+min(( i, length_triangle_base))):
            temp = temp + weight[j-1] * Q[i - j]
        Q_routed[i-1] = temp
    return Q_routed

