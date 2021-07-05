from tqdm import tqdm
from collections import defaultdict
import struct
import re
import gzip
import argparse
import numpy as np
import pickle

def encode(values):
    if (len(values) == 0):
        return np.array([], dtype=np.uint8)
    if (len(values) == 1 and values[0] == 0):
        return np.array([0], dtype=np.uint8)
    values = np.array(values,  dtype=int)
    sizes = np.ones(values.shape,  dtype=np.uint8)
    vals_copy = np.copy(values)
    while np.any(vals_copy > 0):
        vals_copy = vals_copy // 128
        sizes[vals_copy > 0] += 1
    max_size = np.max(sizes)
    sizes = np.cumsum(sizes)
    starts = np.array(np.hstack(([0], sizes[0:-1])), dtype= int)
    ends = np.array(sizes - 1, dtype= int)
    length = ends[-1] + 1
    result = np.zeros((length,),  dtype=np.uint8)
    if values[0] == 0:
        ends[0] = -1
        result[0] = 128
    result[ends] += 128
    for i in range(max_size):
        result[ends[(ends >= starts)]] += \
                np.uint8(values[(values > 0)] % 128)
        values = values // 128
        ends -= 1
    return bytes(result.tolist())

def decode(values):
    values = np.frombuffer(values, dtype=np.uint8)
    if (len(values) == 0):
        return np.array([], dtype=np.uint8)
    if (len(values) == 1 and values[0] == 0):
        return np.array([0], dtype=np.uint8)
    values = np.array(values,  dtype=int)
    ends = np.where(values >= 128)[0]
    starts = np.array(np.hstack(([0], (ends + 1)[:-1])), dtype = int)
    result = np.zeros((starts.shape[0],  ),  dtype=int)
    sizes = ends - starts + 1
    length = np.max(sizes)
    values[ends] -= 128
    number = 1
    for i in range(length):
        result[(ends >= starts)] += \
            values[ends[(sizes > 0)]] * number
        sizes -= 1
        ends -= 1
        number *= 128
    return result

class InvertIndex:
    def __init__(self, files):
        self.forward_index = defaultdict(list)
        self.doc_url = defaultdict(str)
        doc_id = 0
        for v, i in tqdm(enumerate(files)):
            with gzip.open(i, 'rb') as f:
                temp = f.read(4)
                while (temp):
                    number = struct.unpack('i', temp)[0]
                    file = f.read(number).decode('utf8', 'ignore')
                    txt = re.split(r'\x1a', file)
                    self.doc_url[doc_id] = txt[0][txt[0].find('http'):]
                    temp = set()
                    for i in range(1, len(txt)):
                        temp = temp | set(re.findall(r'\w+', txt[i].lower()))
                    self.forward_index[doc_id] = temp
                    temp = f.read(4)
                    doc_id += 1
    
    def forward_index_to_inverted(self):
        self.index = defaultdict(list)
        for url_id in tqdm(self.doc_url):
            for word in self.forward_index[url_id]:
                if word in self.index:
                    self.index[word].append(url_id)
                else:
                    self.index[word] = list([url_id])
            del self.forward_index[url_id]
        del self.forward_index
    
    def encode_index(self):
        for word in tqdm(self.index):
            self.index[word] = encode(self.index[word])

def main():
    parser = argparse.ArgumentParser(description='read docs from dataset')
    parser.add_argument('input_files', nargs='+', help='input files with .gz format')
    args = parser.parse_args()
    files = args.input_files
    print('Reading docs ...')
    storage = InvertIndex(files)
    print('Processing forward index to inverted ...')
    storage.forward_index_to_inverted()
    print('Encoding index ...')
    storage.encode_index()
    with open('storage.pkl', 'wb') as f:
        pickle.dump(storage, f)    

if __name__ == "__main__":
    main()
