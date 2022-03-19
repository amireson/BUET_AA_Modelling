import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

def dottyplots(par_array,metric,metric_name,pn,behavioural_threshold=[]):
    n=len(pn)
    plotdims=[(1,1,4,4),(1,2,8,4),(1,3,12,4),
             (2,2,8,8),(2,3,12,8),(2,3,12,8),
             (3,3,12,12),(3,3,12,12),(3,3,12,12),
             (4,3,12,16),(4,3,12,16),(4,3,12,16)]
    pa,pb,pc,pd=plotdims[n-1]

    pl.figure(figsize=(pc,pd))
    for i in range(n):
        pl.subplot(pa,pb,i+1)
        pl.plot(par_array[:,i],metric,'.k')
        if behavioural_threshold!=[]:
            xl=pl.gca().get_xlim()
            pl.plot(xl,[behavioural_threshold,behavioural_threshold],'-b')
            pl.plot(par_array[metric<=behavioural_threshold,i],metric[metric<=behavioural_threshold],'or')
        pl.xlabel(pn[i],fontsize=13)
        if i/pb==int(i/pb): pl.ylabel(metric_name,fontsize=13)
        pl.grid()

# Calculate Change in storage:
def DeltaS(state,start,end):
    S=np.zeros(len(state['SWE']))
    for n in state: S=S+state[n]
    S=S[start:end]
    S=S-S[0]
    return S


def WaterBalancePlot(flux,state,forcing,start,end):
    # Do a nice water balance plot
    t=flux['PET'][start:end].index
    S=DeltaS(state,start,end)
    P=forcing['P'][start:end].cumsum()
    AET=flux['AET'][start:end].cumsum()
    Q=flux['Q_mm'][start:end].cumsum()

    pl.figure(figsize=(10,5))
    pl.fill_between(t,Q+AET+S,0., color='darkgreen',label='cumulative Q')
    pl.fill_between(t,S+AET,0., color='forestgreen',label='cumulative AET')
    pl.fill_between(t,S,0., color='lightgreen',label='$\Delta S$')
    pl.plot(forcing['P'][start:end].cumsum(),label='cumulative P',color='b')
    # pl.fill_between(flux['Q_mm'][start:end].cumsum()+flux['AET'][start:end].cumsum()+S,label='Streamflow',color='r')
    # pl.plot(flux['AET'][start:end].cumsum(),label='AET',color='darkgreen')

    pl.legend(fontsize=13)
    pl.ylabel('Water balance (mm)',fontsize=13); pl.grid()
 

def PlotEverything(flux,state,forcing,start,end,freq):
    # Do a nice plot of model outputs:
    tS=state['SWE'].resample(freq).mean()[start:end].index
    SWE=(state['SWE'].resample(freq).mean())[start:end]
    SMS=(state['SMS'].resample(freq).mean())[start:end]
    S1=(state['S1'].resample(freq).mean())[start:end]
    S2=(state['S2'].resample(freq).mean())[start:end]

    t=flux['PET'][start:end].resample(freq).sum().index
    P=forcing['P'][start:end].resample(freq).sum()
    AET=flux['AET'][start:end].resample(freq).sum()
    PET=flux['PET'][start:end].resample(freq).sum()
    Q=flux['Q_mm'][start:end].resample(freq).sum()


    pl.figure(figsize=(10,7))
    pl.subplot(2,1,1)
    pl.fill_between(t,PET,0., color='lightgreen',label='PET',step='pre')
    pl.step(t,P,label='P',color='b')
    pl.step(t,Q,label='Q',color='m')
    pl.step(t,AET,label='AET',color='g')
    pl.legend(fontsize=13)
    pl.ylabel('Fluxes (mm)',fontsize=13); pl.grid()

    pl.subplot(2,1,2)
    pl.step(tS,SWE,label='SWE',color='tab:cyan')
    pl.step(tS,SMS,label='SMS',color='tab:grey')
    pl.step(tS,S1,label='S1',color='tab:olive')
    pl.step(tS,S2,label='S2',color='tab:brown')
    pl.legend(fontsize=13)
    pl.ylabel('States (mm)',fontsize=13); pl.grid()

