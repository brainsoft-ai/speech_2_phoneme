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

from beam_search import beam_decode
from timit_data import timit
from utils import *
np.random.seed(1234)
torch.manual_seed(1234)

__all__ = ['arsg']

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            unit = m.weight.mean() * 0.1
            m.weight.add_(torch.randn(m.weight.size()) * unit)

def column_norm_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            if 'rnn' in name:
                pass
            else:
                param.data = param.data / param.data.norm(dim=0)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            if 'rnn' in name:
                nn.init.orthogonal_(param.data, gain=1)
            else:
                nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

class EndoderAttention(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim 
        self.dec_hid_dim = dec_hid_dim 
        self.num_layers = num_layers

        self.rnn = nn.GRU(
                input_dim,
                enc_hid_dim,
                num_layers=self.num_layers,
                bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, num_layers):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim 
        self.dec_hid_dim = dec_hid_dim 
        self.num_layers = num_layers

        self.rnn = nn.GRU(
                input_dim,
                enc_hid_dim,
                num_layers=self.num_layers,
                bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        hidden = torch.zeros(self.num_layers*2, src.size(1), self.dec_hid_dim).cuda()
        outputs, hidden = self.rnn(src, hidden)

        #outputs = [src len, batch size, enc hid dim * 2] [len, 1, 512]
        #hidden = [n layers * num directions, batch size, hid dim] [6, 1, 256]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        backward_1 = hidden[-2,:,:] # [1, 256]
        forward_1 = hidden[-1,:,:] # [1, 256] 
        top_hidden = torch.cat((backward_1, forward_1), dim=1) # [1, 512]

        #initial decoder hidden is final hidden state of the forwards and backwards 
        decoder_init_hidden = torch.tanh(self.fc(top_hidden))
        
        return outputs, decoder_init_hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

def one_hot_embedding(labels, num_classes=64):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
       
        self.rnn = nn.GRU(
                (enc_hid_dim * 2) + 64,
                dec_hid_dim,
                num_layers=1,
            )
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + 1, output_dim)
        
    def forward(self, input, hidden, encoder_outputs):
             
        # input = [batch size]
        # hidden = [batch size, 256]
        # encoder_outputs = [src len, 1, 512]

        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = one_hot_embedding(input).cuda()
        # embedded = [1, batch_size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len] -> [batch size, 1, src len]
        a = a.unsqueeze(1)

        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2] -> [1, batch size, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        input = input.squeeze(0).unsqueeze(1).float()
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, input), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0, show_attention=False):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        src_len = 779 
        # src_len = 621 
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, decoder_init_hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = torch.zeros([batch_size]).long().cuda()
        
        hidden = decoder_init_hidden 
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda()
        attentions = None
        if show_attention:
            attentions = torch.zeros(trg_len, batch_size, src_len).cuda()
        
        for t in range(0, trg_len):
            # receive output tensor (predictions) and new hidden state

            output, hidden, a = self.decoder(input, hidden, encoder_outputs)
            
            #decide if we are going to use teacher forcing or not
            top1 = output.argmax(1) 
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1

            outputs[t] = output
            if show_attention:
                attentions[t] = a.squeeze(1)
            
        return outputs, attentions

    def beam_search(self, src, trg, beam_size=10):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        # first input to the decoder is the <sos> tokens
        input = torch.zeros([batch_size]).long().cuda()
        
        decoded = beam_decode(self.decoder,
                input,
                hidden,
                encoder_outputs,
                trg,
                beam_size=beam_size)

        decoded_numpy = np.zeros([trg_len, batch_size], dtype=int)
        for each_i, each_decode in enumerate(decoded):
            for each_j, elem in enumerate(each_decode[0]):
                if each_j >= trg_len-1:
                    break
                decoded_numpy[each_j, each_i] = elem

        return decoded_numpy

def train(model, iterator, optimizer, criterion, clip, columnNorm=False, adaWeightNoise=False):
    
    if adaWeightNoise:
        pass
        # model.apply(add_noise_to_weights)

    model.train()
    
    epoch_loss = 0
    epoch_per = 0
    
    for i, (src, trg) in enumerate(iterator):
       
        src = np.transpose(src, (1,0,2))
        trg = np.transpose(trg, (1,0))

        compare_trg = trg.numpy()

        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        
        output, attentions = model(src, trg, teacher_forcing_ratio=0.75)
        
        compare_output = output.argmax(axis=2).detach().cpu().numpy()
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output.view(-1, output_dim)
        trg = trg.reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if columnNorm:
            model.apply(column_norm_weights)
        
        epoch_loss += loss.item()
        epoch_per += PER(compare_trg, compare_output)
        
    return epoch_loss / len(iterator), epoch_per / len(iterator)


def evaluate(model, iterator):
    
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

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.reshape(-1)
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            epoch_per += PER(compare_trg, compare_output) 
            
    return epoch_loss / len(iterator), epoch_per / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    os.makedirs("best_models", exist_ok=True)


    #  name_model = "best_models/1016-fixPER-batch1.pt"
    BATCH_SIZE = 1
    BATCH_SIZE = 80

    INPUT_DIM = 123
    OUTPUT_DIM = 63
    ECN_NUM_LAYER = 3
    ENC_HID_DIM = 256 
    DEC_HID_DIM = 256 

    N_EPOCHS = 1000
    CLIP = 1

    augmentation = []
    # augmentation = ['repeat']

    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_NUM_LAYER)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, attn)

    model = Seq2Seq(enc, dec).cuda()
    criterion = nn.CrossEntropyLoss()
    earlyStopCount = 0 

    train_type = "first"
    if train_type == 'first':
        lr, rho, eps, weight_decay =1.0, 0.95, 1e-08, 0 
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        columnNorm, adaWeightNoise = True, False
        name_model = "best_models/1019-first.pt"
        model.apply(init_weights)
        epoch, best_valid_loss, best_valid_per = load_model(model, name_model, optimizer)
    elif train_type == 'second':
        lr, rho, eps, weight_decay =1.0, 0.95, 1e-08, 0.1
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        columnNorm, adaWeightNoise = False, True
        name_model = "best_models/1019-first.pt"
        epoch, best_valid_loss, best_valid_per = load_model(model, name_model, optimizer)
        name_model = "best_models/1019-second.pt"
        epoch = 0
    elif train_type == 'last':
        lr, rho, eps, weight_decay =1.0, 0.95, 1e-10, 0.1
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        columnNorm, adaWeightNoise = False, True
        name_model = "best_models/1019-second.pt"
        epoch, best_valid_loss, best_valid_per = load_model(model, name_model, optimizer)
        name_model = "best_models/1019-last.pt"
        epoch = 0

    train_iterator = DataLoader(
        timit(feats_type='fbank', set_name='train', aug=augmentation),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
    )
    valid_iterator = DataLoader(
        timit(feats_type='fbank', set_name='dev'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )

    while earlyStopCount < 1250:
        start_time = time.time()
        train_loss, train_per = train(model, train_iterator, optimizer, criterion, CLIP, columnNorm, adaWeightNoise)
        valid_loss, valid_per = evaluate(model, valid_iterator)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if train_type in ['first', 'second'] \
           and valid_loss < best_valid_loss:
            print(f'  -- best model found --  ')
            best_valid_loss = valid_loss
            best_valid_per = valid_per
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'best_valid_loss': valid_loss,
                'best_valid_per': valid_per,
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': valid_loss
                }, name_model)
            earlyStopCount += 0
        elif train_type in ['last'] \
           and valid_per < best_valid_per:
            print(f'  -- best model found --  ')
            best_valid_loss = valid_loss
            best_valid_per = valid_per
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'best_valid_loss': valid_loss,
                'best_valid_per': valid_per,
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': valid_loss
                }, name_model)
            earlyStopCount += 0
        else:
            earlyStopCount += 1

        print(f'Epoch: {epoch+1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:7.3f} | Train PER: {train_per:7.3f}')
        print(f'\tVal.  Loss: {valid_loss:7.3f} | Val.  PER: {valid_per:7.3f}')
        print(f'\tBest  Loss: {best_valid_loss:7.3f} | Best  PER: {best_valid_per:7.3f}')

        epoch += 1

