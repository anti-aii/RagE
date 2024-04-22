import torch 
import torch.nn as nn 

class ExtraRoberta(nn.Module):
  def __init__(self, method= "sum", layers= [-1,-2,-3,-4]):
    super(ExtraRoberta, self).__init__()
    assert method in ['sum', 'concat', 'mean']
    self.method= method
    self.layers= layers

  def forward(self, inputs):
    hidden_states= inputs
    hidden_states= [hidden_states[i] for i in self.layers]
    embedding= torch.stack(hidden_states, dim= 0)

    if self.method == 'sum':
      return torch.sum(embedding, dim= 0)
    elif self.method == 'concat':
      return torch.concat(hidden_states, dim= -1)
    elif self.method == 'mean':
      return torch.mean(embedding, dim= 0)