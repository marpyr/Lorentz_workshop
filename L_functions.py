import numpy as np
import xarray as xr
from datetime import datetime 
from datetime import timedelta
from datetime import date
import time

import numpy as np
import xarray as xr
from datetime import datetime 
from datetime import timedelta
from datetime import date
import time

import pandas as pd

def sel_train_data_lead(nc_in_file,target_len,
                        s_target_date,e_target_date,
                        rw_1,lead_time,rw,ntimestep,excluded_years):
    '''
    This function inputs a 2-D file.nc, reads it as a xarray and creates
    a predictor array. 1_D:time, 2_D:features
    
    The length of the target time series must be given (target_len).
    The start date and end date that we want to predict must be given 
    (e.g., s_target_date='16-10-1980', e_target_date='16-12-2021') and
    the running window that was already applied on the predictors with center=False must be
    declared (rw_1).
    
    The predictor array is selected in a way so that the 
    needed date is predicted at a certain lead time (lead_time) and for a specific running
    window that was applied on the target with center=True (rw). If center=False, then set rw=0.
    
    A selected time step for the LSTM is considered (ntimestep).
    A list of years can be excluded from the selection with e.g., excluded_years=[2005,2007,2018,2004,2006]
    '''
    
    print('starting')

    SDD = int(s_target_date[0:2])
    SMM = int(s_target_date[3:5])
    SYY=int(s_target_date[6:10])
    print('start target date',SDD,SMM,SYY)

    EDD = int(e_target_date[0:2])
    EMM = int(e_target_date[3:5])
    EYY = int(e_target_date[6:10])
    print('end target',EDD,EMM,EYY)

    half_rw = int(rw/2)
    
    # Create correctly formated datetime
    date_target = datetime.strftime(datetime(year=SYY,month=SMM,day=SDD), "%Y.%m.%d")
    
    pc_predictor = []
    time_list = []
    it = 0
    ii = 0
    YYY = SYY
    while YYY < EYY+1:
        if YYY not in excluded_years:
            date_start = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d") - timedelta(days=half_rw+lead_time+rw_1+ntimestep-1),"%Y.%m.%d")
            date_end = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d") - timedelta(days=half_rw+lead_time+rw_1),"%Y.%m.%d")
            f = nc_in_file.sel(time = slice(date_start,date_end))
            f = f.assign_coords(time=range(ntimestep))
            time_list.append(date_target)
            pc_predictor.append(f)
        if date_target == datetime.strftime(datetime(year=YYY,month=EMM,day=EDD),"%Y.%m.%d"):
            YYY = YYY + 1
            date_target = datetime.strftime(datetime(year=YYY,month=SMM,day=SDD), "%Y.%m.%d")
            it = 0
            #print(YYY)
        else:
            it = 1
        ii = ii+1
        date_target = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")+timedelta(days=it),"%Y.%m.%d") 
    pc_predictor = xr.concat(pc_predictor,"new_time").rename({"time":"lag"}).rename({"new_time":"time"})
    pc_predictor = pc_predictor.assign_coords(time=time_list)
    pc_predictor = pc_predictor.assign_coords(time=pd.DatetimeIndex(pc_predictor.time)) #-pd.Timedelta("15 d"))
    return pc_predictor
    
def climat_probab(index_file,month,n_days_list):
    '''
    This function calculates the daily climatological probability of having an event.
    In index_file the events are equal to 1 and the no-events equal to zero.
    n_days_list is a list of the number of days for each of the months you want the daily probability to be calculated
    and month is the first month you start with. i.e., for counting the daily probability for the months October, 
    November, December you would give n_days_list = [31,30,31]; month=10;  
    '''
    
    class_num = []
    for ii in range(len(n_days_list)):
        for jj in range(1,n_days_list[ii]+1):
            month_sel = index_file.sel(time=index_file.time.dt.month.isin([ii+month])) #select month from all years
            day_sel = month_sel.sel(time=month_sel.time.dt.day.isin([jj])) # select day 
            class_num.append(day_sel.values.sum())

    n_years = len(index_file.time.dt.year.to_index().unique())
    d_prob = np.asarray(class_num)/n_years
    
    # daily probability fixed in an array form. Basically repeated for an n number of years
    d_prob_rep = np.repeat(np.expand_dims(d_prob, axis=0),n_years,axis=0)
    d_prob_y = d_prob_rep.reshape(d_prob_rep.shape[0]*d_prob_rep.shape[1])

    print('number of days per year',np.asarray(class_num).shape)
    #print('daily probability',np.asarray(class_num)/n_years)

    return(d_prob,d_prob_y)