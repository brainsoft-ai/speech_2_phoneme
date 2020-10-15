"""
module for misc.
"""
import struct
import numpy as np

phn_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv',  'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
phn_61 = ['<sos>', '<eos>'] + phn_61
mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', 'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#'}
phn_39 = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh',  'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l',  'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
phn_39 = ['<sos>', '<eos>'] + phn_39

def load_dj_spectrogram(filepath, verbose=False):
    """
    djs reader
    filepath : absolute path to djs file
    """

    try:
        with open(filepath, "rb") as f:
            header = f.read(32)
            num_channels, lowest_freq, highest_freq, num_spectrums = \
                struct.unpack('iiii', header[:16])
            data = np.fromfile(f, dtype=np.float32)

    except IOError as error:
        print("Couldn't open file (%s.)" % error)
        return 0, 0, 0, 0, []

    if verbose:
        print("max value: %.10f at %d" % (max(data), np.argmax(data)))

        print("data len: %d" % len(data))
        print("data: ", data[0:20])
        print("data: ", data[len(data) - 10:len(data)])

    return num_channels, lowest_freq, highest_freq, num_spectrums, data


def editDistDP(str1, str2):
    m, n = len(str1), len(str2)

    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0: 
                dp[i][j] = j
            elif j == 0: 
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n] 

def PER(y, y_hat, verbose=False):
    y, y_hat = y.T, y_hat.T
    batch_size, trg_len = y.shape[0], y.shape[1]
    sum_per = 0

    for idx in range(batch_size):
        answer = np.trim_zeros(y[idx], 'b')
        answer_rough = answer.copy()
        predict = np.trim_zeros(y_hat[idx], 'b')

        for idy, each in enumerate(answer):
            if phn_61[each] in mapping.keys():
                answer_rough[idy] = phn_61.index(mapping[phn_61[each]])
        answer_rough = answer_rough[1:] # trim <sos>

        eos_index = np.where(predict == 1)
        if len(eos_index[0]) >= 1:
            predict = predict[:eos_index[0][0]]

        for idy, each in enumerate(predict):
            if phn_61[each] in mapping.keys():
                predict[idy] = phn_61.index(mapping[phn_61[each]])

        each_per = editDistDP(answer, predict) / answer.shape[0]
        sum_per += each_per

        if verbose and idx == 0:
            print("--per", each_per)
            for each in answer_rough:
                print(phn_61[each], end=' ')
            print()
            for each in predict:
                print(phn_61[each], end=' ')
            print()

    result = sum_per / batch_size

    return result
