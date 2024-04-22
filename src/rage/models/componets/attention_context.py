import torch 
import torch.nn as nn 


@torch.jit.script 
def element_wise(x1, x2): 
  return x1 * x2 

class AttentionWithContext(nn.Module):
  def __init__(self,
               units= 64,
               w_bias= True,
               u_bias= True):
    super(AttentionWithContext, self).__init__()
    # w
    self.w= nn.Linear(units, units, bias= True if w_bias== True else False)
    nn.init.xavier_uniform_(self.w.weight)
    if w_bias:
      nn.init.zeros_(self.w.bias)

    # u
    self.u= nn.Linear(units, 1, bias= True if w_bias== True else False)
    nn.init.xavier_uniform_(self.u.weight)

    if u_bias:
      nn.init.zeros_(self.u.bias)

  def forward(self, input, mask= None):
    # input shape  B, L, D
    x= self.w(input) # B, L, D
    x= torch.tanh(x)
    x= self.u(x)
    x= torch.squeeze(x, dim= -1) # B, L

    if mask != None:
      x= x.masked_fill(mask==0, -float('inf'))
    coef_attention= nn.Softmax(dim= -1)(x)

    output= element_wise(input, coef_attention.unsqueeze(-1))

    return torch.sum(output, dim= 1)