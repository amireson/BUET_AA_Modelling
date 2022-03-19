import numpy as np
import pandas as pd
from MyNumba import *

def read_inputs(data_folder):

#     inp=MakeDictArray() #{}

    # % ********  Initial Condition  *********
    fn=data_folder + '/initial_condition.inp'
    data=np.loadtxt(fn,delimiter=' ',usecols=[1])
    watershed_area=data[0]
    ini_values=data[1:]

    # % ********  Precipitation and Temperature  *********
    fn=data_folder + '/Precipitation_Temperature.inp'
    forcing=pd.read_csv(fn,delim_whitespace=True,index_col=0,parse_dates=True,names=['P','T'])
    #forcing=pd.read_csv(fn,delim_whitespace=True,usecols=[1,2],names=['P','T'])
    #forcing.index=pd.date_range(start='1979-01-01',freq='D',periods=len(forcing))
    forcing['month_time_series']=forcing.index.month.astype(float)
    
    # % ********  Evapotranspiration  *********
    fn=data_folder + '/monthly_data.inp'
    Tave,PEave=np.loadtxt(fn,delimiter='\t',unpack=True)
    
#     long_term={}
    long_term=MakeDictArray()
    long_term['monthly_average_T']=Tave
    long_term['monthly_average_PE']=PEave

    return watershed_area, ini_values, forcing, long_term


def obs_streamflow(folder):

    fn=folder + '/streamflow.inp'
    Qobs=pd.read_csv(fn,delim_whitespace=True,index_col=0,parse_dates=True,names=['Q'])
    return Qobs
