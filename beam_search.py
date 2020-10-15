import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
device = torch.device("cuda")


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(decoder, input, hidden, encoder_outputs, trg, beam_size=10, topk=1):
    hidden = hidden.unsqueeze(0)

    decoded_batch = []
    SOS_token, EOS_token = 0, 1

    MAX_LENGTH = trg.size(0)
    BATCH_SIZE = trg.size(1)
    decoder_input = torch.LongTensor([SOS_token]).view(1).cuda()
    # print("def -----------", decoder_input.size())

    # decoding goes sentence by sentence
    for idx in range(BATCH_SIZE):
        if isinstance(hidden, tuple):  # LSTM case
            decoder_hidden = (hidden[0][:,idx, :].unsqueeze(0), hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = hidden[:, idx, :]
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while not nodes.empty():
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()

            decoder_input = n.wordid
            decoder_hidden = n.h

            if (n.wordid.item() == EOS_token and n.prevNode != None) \
               or n.leng >= MAX_LENGTH:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_size)
            nextnodes = []

            for new_k in range(beam_size):
                decoded_t = indexes[0][new_k].view(1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)

            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch

