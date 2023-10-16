from .head import *
from .leg import *
from .hand import *
from .chest import *

def select_dataset_voting(dataset):
    if dataset == 'head':
        return Head_TPL_Voting
    elif dataset == 'leg':
        return Leg_TPL_Voting
    elif dataset == 'chest':
        return Chest_TPL_Voting
    elif dataset == 'hand':
        return Hand_TPL_Voting
    
def select_dataset_heatmap(dataset):
    if dataset == 'head':
        return Head_TPL_Heatmap
    elif dataset == 'leg':
        return Leg_TPL_Heatmap
    elif dataset == 'chest':
        return Chest_TPL_Heatmap
    elif dataset == 'hand':
        return Hand_TPL_Heatmap

def select_dataset_SSL_Train(dataset):
    if dataset == 'head':
        return Head_SSL_Train
    elif dataset == 'leg':
        return Leg_SSL_Train
    elif dataset == 'chest':
        return Chest_SSL_Train
    elif dataset == 'hand':
        return Hand_SSL_Train

def select_dataset_SSL_Infer(dataset):
    if dataset == 'head':
        return Head_SSL_Infer
    elif dataset == 'leg':
        return Leg_SSL_Infer
    elif dataset == 'chest':
        return Chest_SSL_Infer
    elif dataset == 'hand':
        return Hand_SSL_Infer

def select_dataset_SAM(dataset):
    if dataset == 'head':
        return Head_SSL_SAM
    elif dataset == 'leg':
        return Leg_SSL_SAM
    elif dataset == 'chest':
        return Chest_SSL_SAM
    elif dataset == 'hand':
        return Hand_SSL_SAM

def select_dataset_ERE(dataset):
    if dataset == 'head':
        return Head_ERE
    elif dataset == 'leg':
        return Leg_ERE
    elif dataset == 'chest':
        return Chest_ERE
    elif dataset == 'hand':
        return Hand_ERE

