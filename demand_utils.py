import numpy as np
import torch
from evaluation import get_actual_demand, get_known
from utils import load_problem_data
from copy import deepcopy as dc

from lstm_model import LSTM


demand, _ , _ , _= load_problem_data() ##pd.read_csv('./data/demand.csv')


def getSeedDemand(seed, latency_sensitivity, server_type, lookback ):
    

    ## set random seed for numpy
    np.random.seed(seed)

    # get the demand under this seed
    seeded_demand = get_actual_demand(demand)



    ## only get rows for this server type
    seeded_demand = seeded_demand.loc[    (seeded_demand["server_generation"]== server_type)]

    seeded_demand = seeded_demand.reset_index()

    ## remove all other rows, apart from the latency_sensitivity we are interested in 
    seeded_demand = seeded_demand[ [   latency_sensitivity ] ]


    seeded_demand.index.names = ['time_step']



    ## add in missing values for timestep
    for t in range(get_known("time_steps")):
        if t not in seeded_demand.index:
            seeded_demand.loc[t,latency_sensitivity] = 0




    ## add zeroes for lookback values before 0, so we can predict something for the first timesteps
    for t in range(0,-lookback,-1):
        seeded_demand.loc[t,latency_sensitivity] = 0



    seeded_demand = seeded_demand.sort_index()

    return seeded_demand



## get dataframe with past and future values for demand
def prepare_dataframe_for_lstm(df,latency_sensitivity, back_steps, forward_steps):
    df = dc(df)


    for i in range(1, forward_steps):


        df[f'{latency_sensitivity}(t+{i})'] = df[latency_sensitivity].shift(-i)


    ## load model 


    for i in range(1, back_steps+1):
        df[f'{latency_sensitivity}(t-{i})'] = df[latency_sensitivity].shift(i)

    
    

    ## flip 

    df.dropna(inplace=True)

    return df



def get_prediction_from_model(past_demand, server_generation,latency_sensitivity):



    ## load model 

    model = LSTM()
    model.load_state_dict(torch.load("models/", weights_only=True))
    model.eval()


    ## query the model 


    ## return result 


    pass




data = getSeedDemand(0)
