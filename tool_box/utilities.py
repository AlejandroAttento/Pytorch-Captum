import multiprocessing
import torch
from torch.utils.data import DataLoader
import random
import datetime
import numpy as np
import json

class createDataLoader():
    def __init__(self, num_workers = multiprocessing.cpu_count()):
        
        self.num_workers = num_workers

    def create(self, data, batch_size, shuffle = False):
        return DataLoader(data, batch_size = batch_size, shuffle = shuffle, num_workers = self.num_workers)

class listSplitter():
    def __init__(self, split_perc, shuffle = False):

        self.split_perc = split_perc
        self.shuffle = shuffle

    def split(self, lst):

        if self.shuffle:
            random.shuffle(lst)

        split_val = int(len(lst) * self.split_perc)
        
        return lst[0 : split_val], lst[split_val : ]

def type_converter(data, data_type):
    if torch.is_tensor(data):
        return data.type(data_type)
    elif isinstance(data, list):
        return [val.type(data_type) for val in data]

def secondsConverter(n, round = True): 
  if round:
    return str(datetime.timedelta(seconds = np.floor(n)))
  else:
    return str(datetime.timedelta(seconds = n))

class createLog():
    def __init__(self, log, json = False, file_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')):

        self.log = log
        self.json = json
        self.file_name = file_name

    def write(self):

        if self.json:
            self.log_str = json.dumps(self.log)
        else:
            self.log_str = self.log

        with open("{}.txt".format(self.file_name), "w") as file:
            file.write(self.log_str)