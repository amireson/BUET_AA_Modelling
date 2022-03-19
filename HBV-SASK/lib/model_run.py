import numpy as np
import pandas as pd

from model_data import *
from model import *

def StartRun(basin,par_values=[]):

    # read parameters:
    if par_values==[]: par_values=sl.json2dict(folder+'/pars.inp')

    # read inputs
    [watershed_area, ini_values, forcing, long_term ] = read_inputs(basin);

    forcing_data=MakeDictArray()
    for key in forcing: forcing_data[key]=forcing[key].values

    par_vals=MakeDictFloat()
    for key in par_values: par_vals[key]=float(par_values[key])

    Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2=run_model(par_vals,watershed_area,ini_values,forcing_data,long_term)

    # Pack up output fluxes
    flux=pd.DataFrame(index=forcing.index)
    flux['Q_cms'] = Q_cms
    flux['Q_mm'] = Q
    flux['AET'] = AET
    flux['PET'] = PET
    flux['Q1'] = Q1
    flux['Q1routed'] = Q1_routed
    flux['Q2'] = Q2
    flux['ponding'] = ponding

    # Pack up output states
    state=pd.DataFrame(index=pd.date_range(start=flux.index[0],periods=len(SWE),freq='D'))
    state['SWE'] = SWE
    state['SMS'] = SMS
    state['S1'] = S1
    state['S2'] = S2
        
    return flux, state, forcing


