import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

COLUMNS = [\
    "LABEL",
    "A_P_L1",
    "A_V_L1",
    "B_P_L1",
    "B_V_L1",
    "A_P_L2",
    "A_V_L2",
    "B_P_L2",
    "B_V_L2",
    "A_P_L3",
    "A_V_L3",
    "B_P_L3",
    "B_V_L3",
    "A_P_L4",
    "A_V_L4",
    "B_P_L4",
    "B_V_L4",
    "LC_1",
    "LC_2",
    "LC_3",
    "LC_4",
    "LC_5",
    ]

P_col = [
    "A_P_L1",
    "B_P_L1",
    "A_P_L2",
    "B_P_L2",
    "A_P_L3",
    "B_P_L3",
    "A_P_L4",
    "B_P_L4",
]

V_col = ["A_V_L1","A_V_L2","A_V_L3","A_V_L4","B_V_L1","B_V_L2","B_V_L3","B_V_L4"]

def _norm_input_function(frame,path):
    if frame is None:
        if path is not None:
            frame = import_a_set(path)
        else:
            raise Exception("Error : Provide a dataframe to split")
    return frame
def import_a_set(path="data/Data_A.csv"):
    frame = pd.read_csv(path,header=None)
    frame.columns = COLUMNS
    return frame


def rescale_features_4(frame=None,path=None):
    """
    Gap of MID - L1 - L2 - L3 - L4
    Volumes as % of sum of volume
    """
    frame = _norm_input_function(frame,path)
    frame["MID_P"] = (frame["A_P_L1"] + frame["B_P_L1"]) / 2
    scale_A = (frame["A_P_L4"] - frame["MID_P"]).values
    scale_B = -(frame["B_P_L4"] - frame["MID_P"]).values

    frame["A_D_1"] = (frame["A_P_L1"] - frame["MID_P"])/scale_A
    frame["A_D_2"] = (frame["A_P_L2"] - frame["A_P_L1"])/scale_A
    frame["A_D_3"] = (frame["A_P_L3"] - frame["A_P_L2"])/scale_A
    frame["A_D_4"] = (frame["A_P_L4"] - frame["A_P_L3"])/scale_A

    frame["B_D_1"] = (frame["B_P_L1"] - frame["MID_P"])/scale_B
    frame["B_D_2"] = (frame["B_P_L2"] - frame["B_P_L1"])/scale_B
    frame["B_D_3"] = (frame["B_P_L3"] - frame["B_P_L2"])/scale_B
    frame["B_D_4"] = (frame["B_P_L4"] - frame["B_P_L3"])/scale_B

    frame['SUM_V'] = frame["A_V_L1"] + frame["A_V_L2"] + frame["A_V_L3"] + frame["A_V_L4"] + frame["B_V_L1"] + frame["B_V_L2"] + frame["B_V_L3"] + frame["B_V_L4"] 
    
    frame["scale"] = (frame["A_P_L4"] - frame["B_P_L4"])
    for price_col in ["B_P_L3","B_P_L2","B_P_L1","A_P_L4","A_P_L3","A_P_L2","A_P_L1","MID_P"]:
        frame[price_col] = (frame[price_col]-frame["B_P_L4"])/frame["scale"]

    for vol_col in ["A_V_L1","A_V_L2","A_V_L3","A_V_L4","B_V_L1","B_V_L2","B_V_L3","B_V_L4"]:
        frame[vol_col+"P"] = frame[vol_col] / frame["SUM_V"]
        frame[vol_col] = (frame[vol_col]-frame[vol_col].min())/(frame[vol_col].max()-frame[vol_col].min()) 
    frame["MID_P"] = (frame["MID_P"]-frame["MID_P"].min())/(frame["MID_P"].max()-frame["MID_P"].min()) 
    frame["LABEL"] = frame["LABEL"] *2 -1
    frame["LABEL_UP"] = frame["LABEL"].apply(lambda x : 1 if x == 1 else 0)
    frame["LABEL_DOWN"] = frame["LABEL"].apply(lambda x : 1 if x == -1 else 0)
    frame =frame.drop("LABEL",axis=1)
    frame =frame.drop(['scale',"A_P_L4","B_P_L4","SUM_V"],axis=1)

    return frame


def rescale_features_2(frame=None,path=None):
    """
    Lowest bid @ 0 , Highest ask at 1
    Volumes as % of sum of volume
    """
    frame = _norm_input_function(frame,path)
    frame["MID_P"] = (frame["A_P_L1"] + frame["B_P_L1"]) / 2
    price_scale = (frame[P_col].max().max() - frame[P_col].min().min())
    price_min =  frame[P_col].min().min()
    for price_col in P_col + ["MID_P"]:
        frame[price_col] = (frame[price_col]-price_min)/price_scale
    vol_scale = frame[V_col].max().max() - frame[V_col].min().min()
    vol_min = frame[V_col].min().min()
    for vol_col in V_col:
        frame[vol_col] = (frame[vol_col]-vol_min)/vol_scale
    for col in frame.columns:
        frame[col] = frame[col] *2 -1 
    frame["LABEL_UP"] = frame["LABEL"].apply(lambda x : 1 if x == 1 else 0)
    frame["LABEL_DOWN"] = frame["LABEL"].apply(lambda x : 1 if x == -1 else 0)
    frame =frame.drop("LABEL",axis=1)
    return frame

def rescale_features_1(frame=None,path=None):
    """
    Lowest bid @ 0 , Highest ask at 1
    Volumes as % of sum of volume
    """
    frame = _norm_input_function(frame,path)
    frame["MID_P"] = (frame["A_P_L1"] + frame["B_P_L1"]) / 2
    frame["scale"] = (frame["A_P_L4"] - frame["B_P_L4"])
    for price_col in ["B_P_L3","B_P_L2","B_P_L1","A_P_L4","A_P_L3","A_P_L2","A_P_L1","MID_P"]:
        frame[price_col] = (frame[price_col]-frame["B_P_L4"])/frame["scale"]
    frame['SUM_V'] = frame["A_V_L1"] + frame["A_V_L2"] + frame["A_V_L3"] + frame["A_V_L4"] + frame["B_V_L1"] + frame["B_V_L2"] + frame["B_V_L3"] + frame["B_V_L4"] 
    for vol_col in ["A_V_L1","A_V_L2","A_V_L3","A_V_L4","B_V_L1","B_V_L2","B_V_L3","B_V_L4"]:
        frame[vol_col] = frame[vol_col] / frame["SUM_V"]
    frame = frame.drop(['scale',"A_P_L4","B_P_L4","SUM_V"],axis=1)
    for col in frame.columns:
        frame[col] = frame[col] *2 -1 
    frame["LABEL_UP"] = frame["LABEL"].apply(lambda x : 1 if x == 1 else 0)
    frame["LABEL_DOWN"] = frame["LABEL"].apply(lambda x : 1 if x == -1 else 0)
    frame =frame.drop("LABEL",axis=1)
    return frame

def rescale_features_3(frame=None,path=None):
    frame = _norm_input_function(frame,path)
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    frame[P_col] = price_scaler.fit_transform(frame[P_col])
    frame[V_col] = volume_scaler.fit_transform(frame[V_col])
    frame["LABEL"] = frame["LABEL"] *2 -1
    frame["LABEL_UP"] = frame["LABEL"].apply(lambda x : 1 if x == 1 else 0)
    frame["LABEL_DOWN"] = frame["LABEL"].apply(lambda x : 1 if x == -1 else 0)
    frame =frame.drop("LABEL",axis=1)
    return frame


def split_training_data(frame = None,test_ratio = 0.1,path=None):

    frame = _norm_input_function(frame,path)
    test_frame = frame.sample(frac = test_ratio).reset_index(drop=True)
    train_frame = frame.drop(test_frame.index).reset_index(drop=True)
    
    y_train = train_frame[["LABEL_UP","LABEL_DOWN"]]
    x_train = train_frame.drop(["LABEL_UP","LABEL_DOWN"],axis=1)
    y_test = test_frame[["LABEL_UP","LABEL_DOWN"]]
    x_test = test_frame.drop(["LABEL_UP","LABEL_DOWN"],axis=1)

    return x_train,y_train,x_test,y_test

def main_pipeline(path="data/Data_A.csv",feat_function:int=1,test_ratio:float = 0.1,split=True):
    frame = _norm_input_function(None,path)
    frame = globals()[f'rescale_features_{feat_function}'](frame)
    if split:
        x_train,y_train,x_test,y_test = split_training_data(frame,test_ratio)
        return (x_train,y_train,x_test,y_test),(x_train.shape)
    else:
        return frame,frame.shape

"""
FEATURE :

ORDERBOOK Price i.e. A/B_P_1234 :
    Between -1 and 1, centered on the mid (rescaling down for BID and for ASK -> might not be symetrical)
ORDERBOOK Volumne i.e. A/B_V_1234 :
    Between 0 and 1, as a % of the total volume for this data point
LAST Price change i.e. LC_1234 :
    Either -1 or 1, rescaled to fit a tanh/sigmoid (y : 2x-1)

MODELS : 

#1 MEAN REVERSION BASED : 
Drop every columns except last PC -> Train model 

#2 Tightness of levels:
Compute level gap on sell/buy side : Train model on the 3x2 delta

#3 Include volumne => X = Price/Vol : Vol increase -> tighter

"""

if __name__=="__main__":
    frame,_ = main_pipeline(feat_function=4,split=False)
    print(frame.head())
    #x_train,y_train,x_test,y_test = split_training_data(frame=data,test_ratio=0.1)
    #["B_P_L4","B_P_L3","B_P_L2","B_P_L1","A_P_L4","A_P_L3","A_P_L2","A_P_L1"]