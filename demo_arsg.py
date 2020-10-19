import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import math
import time
import os
import copy

from arsg import Seq2Seq, Attention, Decoder, Encoder, PER
from timit_data import timit

np.random.seed(1234)
torch.manual_seed(1234)

def test_model(model, iterator, beam_size=10):

    model.eval()
    epoch_per, epoch_loss = 0, 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):

            src = np.transpose(src, (1,0,2))
            trg = np.transpose(trg, (1,0))

            compare_trg = trg.numpy()

            src, trg = src.cuda(), trg.cuda()

            output, attentions = model(src, trg)
            compare_output = output.argmax(axis=2).detach().cpu().numpy()
            beam_output = model.beam_search(src, trg, beam_size=10)
            print(trg.T)
            beam_output = beam_output[1:]
            print(beam_output.T)
            #  print(compare_outpt)

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.reshape(-1)
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            print('greedy')
            epoch_per += PER(compare_trg, compare_output, verbose=True) 
            print('beam')
            _ = PER(compare_trg, beam_output, verbose=True) 
            # print(beam_output)
            break
        
        epoch_loss /= len(iterator)
        epoch_per /= len(iterator)

    print(f'| Test Loss: {epoch_loss:7.3f} | Test PER: {epoch_per:7.3f} |')
    return

if __name__ == "__main__":
    INPUT_DIM = 123
    OUTPUT_DIM = 63
    ECN_NUM_LAYER = 3
    ENC_HID_DIM = 256 
    DEC_HID_DIM = 256 

    N_EPOCHS = 1000
    BATCH_SIZE = 80
    CLIP = 1

    augmentation = []
    # augmentation = ['repeat']

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_NUM_LAYER)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, attn)

    model = Seq2Seq(enc, dec).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    valid_iterator = DataLoader(
        timit(feats_type='fbank', set_name='test'),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    name_model = "best_models/1019-first.pt"
    if not os.path.exists(name_model):
        print(name_model, "not exist")
        exit(0)

    print("loading", name_model)
    checkpoint = torch.load(name_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_valid_loss = checkpoint['best_valid_loss']
    best_valid_per = checkpoint['best_valid_per']
    print(f'epoch: {epoch}, best_loss: {best_valid_loss}')

    test_model(model, valid_iterator)

