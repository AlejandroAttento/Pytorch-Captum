import torch
import pandas as pd
import numpy as np

# Function to handle local devices
class deviceHandler():
  def __init__(self):
      if torch.cuda.is_available():
        print("GPU is available")
        self.device = torch.device('cuda')
      else:
        print("GPU isn't available, fallback to CPU")
        self.device = torch.device('cpu')
  
  def force_cpu(self):
    self.device = torch.device('cpu')
  
  def reset_device(self):
      if torch.cuda.is_available():
        self.device = torch.device('cuda')
      else:
        self.device = torch.device('cpu')
  
  def get_device(self):
    return self.device

  def data_to_tensor(self, data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
      return torch.tensor(data.values).to(self.device, non_blocking=True)
    elif isinstance(data, (np.ndarray, np.generic, list)):
      return torch.tensor(data).to(self.device, non_blocking=True)
    else:
      raise TypeError("Input data type unrecognized")