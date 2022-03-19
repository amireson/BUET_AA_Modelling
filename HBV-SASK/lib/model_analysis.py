import numpy as np
import pandas as pd
from scipy.optimize import minimize
from IPython.display import clear_output

from model_run import *
from model import *
from model_data import *

def run_optimization(basin,dates,metric,par_bounds,par_values,pn,pv):

    # read inputs
    [ watershed_area, ini_values, forcing, long_term ] = read_inputs(basin);

    # read observed streamflow
    Qobs=obs_streamflow(basin)

    # Truncate data to calibration period only, including spinup period
    forcing=forcing[dates['start_spin']:dates['end_calib']]
    Qobs=Qobs[dates['start_calib']:dates['end_calib']]

    # Truncate par_bounds to only those parameters being optimized:
    pb=tuple([par_bounds[i] for i in pn])
    # Run optimization
    print(pv)
    print(pn)
    output=minimize(error_fun,pv,args=(pn,par_values,metric,Qobs,watershed_area, ini_values, forcing, long_term,dates),bounds=pb)
    pv=output['x']
    for n,v in zip(pn,pv): par_values[n]=v
        
    return par_values, pv

def error_fun(pv,pn,par_values,metric,Qobs,watershed_area, ini_values, forcing, long_term,dates):
    
    for n,v in zip(pn,pv): par_values[n]=v
        
    forcing_data=MakeDictArray()
    for key in forcing:
        forcing_data[key]=forcing[key].values

    par_vals=MakeDictFloat()
    for key in par_values: par_vals[key]=float(par_values[key])
    
    Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2=run_model(par_vals,watershed_area,ini_values,forcing_data,long_term)
    Q=pd.DataFrame(index=forcing.index)
    Q['Q']=Q_cms
    Q=Q[dates['start_calib']:dates['end_calib']].values
    err=eval_metric(Qobs.values,Q,metric)

    # Diplay current parameter values and objective function value:
    clear_output(wait = True)
    for n,v in zip(pn,pv): print('%s: %f'%(n,v))
    print('%s: %.4f\n'%(metric,err))

    return err

def eval_metric(yobs,y,metric):
    # Make sure the data sent in here are not in dataframes

    if metric.upper() == 'NSE':
        # Use negative NSE for minimization
        denominator = ((yobs-yobs.mean())**2).mean()
        numerator = ((yobs - y)**2).mean()
        negativeNSE = -1*(1 - numerator / denominator)
        return negativeNSE

    elif metric.upper() == 'NSE_LOG':
        # Use negative NSE for minimization
        yobs=np.log(yobs)
        y=np.log(y)
        denominator = ((yobs-yobs.mean())**2).mean()
        numerator = ((yobs - y)**2).mean()
        negativeNSE = -1*(1 - numerator / denominator)
        return negativeNSE

    elif metric.upper() == 'ABSBIAS':
        return np.abs((y-yobs).sum()/yobs.sum())

    elif metric.upper() == 'ME':
        return (yobs-y).mean()

    elif metric.upper() == 'MAE':
        return (np.abs(yobs-y)).mean()

    elif metric.upper() == 'MSE':
        return ((yobs-y)**2).mean()

    elif metric.upper() == 'RMSE':
        return np.sqrt(((yobs-y)**2).mean())

def MonteCarlo(nReal,pn,pv,par_values,par_bounds,Qobs,dates,basin):
    par_array=np.random.rand(nReal,len(pn))
    for i,n in enumerate(pn):
        lowerlimit,upperlimit=par_bounds[n]
        par_array[:,i]=par_array[:,i]*(upperlimit-lowerlimit)+lowerlimit

    NSE=np.zeros(nReal)
    RMSE=np.zeros(nReal)

    watershed_area, ini_values, forcing, long_term = read_inputs(basin);

    # read observed streamflow
    Qobs=obs_streamflow(basin)

    # Truncate data to calibration period only, including spinup period
    forcing=forcing[dates['start_spin']:dates['end_calib']]
    Qobs=Qobs[dates['start_calib']:dates['end_calib']]

    forcing_data=MakeDictArray()
    for key in forcing: forcing_data[key]=forcing[key].values

    par_vals=MakeDictFloat()
    for key in par_values: par_vals[key]=float(par_values[key])

    for i in range(nReal):
        pv=par_array[i,:]
        for n,v in zip(pn,pv): par_vals[n]=float(v)
        
        #flux,state,forcing=StartRun(par_values['basin'],par_values)
        
        Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2=run_model(par_vals,watershed_area,ini_values,forcing_data,long_term)
        flux=pd.DataFrame(index=forcing.index)
        flux['Q_cms'] = Q_cms
        
        NSE[i]=-eval_metric(Qobs[dates['start_calib']:dates['end_calib']].values.squeeze(),flux['Q_cms'][dates['start_calib']:dates['end_calib']].values,'NSE')
        RMSE[i]=eval_metric(Qobs[dates['start_calib']:dates['end_calib']].values.squeeze(),flux['Q_cms'][dates['start_calib']:dates['end_calib']].values,'RMSE')
        clear_output(wait = True)
        print('Running realization %d'%(i+1))

    return par_array,NSE,RMSE

def GLUE(metric,pn,par_array,par_values,behavioural_threshold,basin):

    par_array=par_array[metric<=behavioural_threshold,:]
    nB=par_array.shape[0]
    
    Q=pd.DataFrame()
    AET=pd.DataFrame()
    for i in range(nB):
        pv=par_array[i,:]
        for n,v in zip(pn,pv): par_values[n]=v
        flux,state,forcing=StartRun(basin,par_values)
        Q[i]=flux['Q_cms']
        AET[i]=flux['AET']
    
    # Get range of flow and AET:
    Qrange=pd.DataFrame()
    Qrange['max']=Q.max(axis=1)
    Qrange['min']=Q.min(axis=1)
    Qrange['med']=Q.median(axis=1)
    Qrange['mean']=Q.mean(axis=1)
    
    AETrange=pd.DataFrame()
    AETrange['max']=AET.max(axis=1)
    AETrange['min']=AET.min(axis=1)
    AETrange['med']=AET.median(axis=1)
    AETrange['mean']=AET.mean(axis=1)
    
    return Q,Qrange,AET,AETrange


