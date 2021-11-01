import wandb
wandb.init(project="MA Project", entity="dylany")

from utils import MADataset, MADataset_test
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
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # build model
    model = MAPredNet()

    opt = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # train
    for epoch in range(50):
        loss_e = 0
        timing_loss_e = 0
        choice_loss_e = 0

        model.train()
        for i,batch in enumerate(tqdm(loader)):
            arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict = batch
            print("batch number: ", i)
            opt.zero_grad()
            loss, timing_loss, choice_loss  = model(arr_b.float(), arr_c.float(), arr_delta_time.float(), event_data, non_event_data, estimate_length, choice_data_dict)
            #print(loss, timing_loss, choice_loss)
            loss.backward() # required_graph = True
            opt.step()
            #print(loss.detach().numpy(), choice_loss.detach().numpy())
            loss_e += loss.item()
            timing_loss_e += timing_loss.item()
            choice_loss_e += choice_loss.item()
        
        loss_e /= data_size
        timing_loss_e /= data_size
        choice_loss_e /= data_size
        wandb.log({'total loss': loss_e, 'training timing loss': timing_loss_e, 'training choice loss': choice_loss_e, "epoch": epoch})


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
    #dataset = MADataset_test()

    model_trained = train(dataset, config,  device)



main()


model = MAPredNet()
#print(model.parameters)