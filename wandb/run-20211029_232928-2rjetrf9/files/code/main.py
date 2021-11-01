import wandb
wandb.init(project="MA Project", entity="dylany")

from utils import MADataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model import MAPredNet
from tqdm import tqdm
import numpy as np
import torch
from random import randrange
#from tensorboardX import SummaryWriter
#writer = SummaryWriter('runs/ma_fit_experiment_1')

import time
from datetime import datetime





def train(dataset, config, device):
    '''
    dataset: the intersection with dataset in test is a blank set
    
    '''

    ## some hyperparams
    data_size = len(dataset)
    loader = DataLoader(dataset, batch_size=config.learning_rate, shuffle=True)

    # build model
    model = MAPredNet()

    opt = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    # train
    for epoch in range(config.epochs):
        loss_e = 0
        timing_loss_e = 0
        choice_loss_e = 0

        model.train()
        for batch in loader:
            arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict = batch

            opt.zero_grad()
            loss, timing_loss, choice_loss  = model(arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict)
            loss.backward()
            opt.step()
            loss_e += loss
            timing_loss_e += timing_loss
            choice_loss_e += choice_loss
        
        loss_e /= data_size
        timing_loss_e /= data_size
        choice_loss_e /= data_size
        wandb.log({'training timing loss', timing_loss_e, epoch})
        wandb.log({'training choice loss', choice_loss_e, epoch})
        wandb.log({'total loss', loss_e, epoch})

        print("Epoch {}. Total Loss: {:.4f}. Timing MLE loss: {:.4f}. Choice BCE loss {:.4f}".format(
                epoch, loss_e, timing_loss_e, choice_loss_e))

        
    return model



def main():
    wandb.config = {
        "learning_rate": 0.01,
        "epochs": 50,
        "batch_size": 1
        }
    config = dict(learning_rate=0.01, epochs=50, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('CUDA availability:', torch.cuda.is_available())
    #writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # data
    dataset = MADataset()

    model_trained = train(dataset, config,  device)



main()


model = MAPredNet()
print(model.parameters)