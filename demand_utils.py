import numpy as np
import pandas as pd
import torch
from evaluation import get_actual_demand, get_known
from utils import load_problem_data
from copy import deepcopy as dc

from lstm_model import LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler



demand, _ , _ , _, _= load_problem_data() ##pd.read_csv('./data/demand.csv')


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
def get_past_demand(df,latency_sensitivity, back_steps):
    df = dc(df)


    # for i in range(1, forward_steps):


    #     df[f'{latency_sensitivity}(t+{i})'] = df[latency_sensitivity].shift(-i)


    ## load model 


    for i in range(1, back_steps):
        df[f'{latency_sensitivity}(t-{i})'] = df[latency_sensitivity].shift(i)

    
    

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





def getPastDataTensor(seed,  latency_sensitivity,server_type, lookback):

    data = getSeedDemand(seed, latency_sensitivity, server_type, lookback)

    shifted_df = get_past_demand(data, latency_sensitivity, lookback)

    shifted_df_as_np = shifted_df.to_numpy()

    

    scaler = MinMaxScaler(feature_range=(-1,1))

    # scaler = StandardScaler()


    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np##[:, lookahead:]

    # Y = shifted_df_as_np[:,:lookahead]

    ## flip matrix, so values further in the past, come first in the matrix
    X = dc(np.flip(X,axis=1))


    ## flip matrix, so values further in the past, come first in the matrix
    # X = dc(np.flip(X,axis=1))

    print(X.shape)

    X = X.reshape( (-1, lookback, 1) )

    # Y = Y.reshape( (-1, lookahead, 1) )

    X = torch.tensor(X).float()
    # Y = torch.tensor(Y).float()

    return X


def getPredictions(past_values, server_type, latency_sensitivity):
    lookahead = 10

    model = LSTM(1, 40, 10, lookahead)
    model.load_state_dict(torch.load(f"lstm_models/{latency_sensitivity}_{server_type}", weights_only=True))
    model.eval()# set to evaluation mode


    df = pd.DataFrame()
    

    for past_data in past_values:
        print(f" past data shape   {past_data.shape}")

        input = torch.unsqueeze(past_data,-1)
        output = model(input)

        np_array = output.numpy()




        print(output)



        #print(past_data)





past_data = getPastDataTensor(0,"low","CPU.S1",25)

getPredictions(past_data,"low","CPU.S1")



    



# data = getSeedDemand(0)
