import torch.nn as nn 
from .loss_rag import LossRAG
from .CosineSimilarityLoss import CosineSimilarityLoss
from .MSELogLoss import MSELogLoss
from .ContrastiveLoss import ContrastiveLoss
from .TripletLoss import TripletLoss
from .InBatchNegatives import InBatchNegativeLoss
from .GISTEmbedLoss import GITSEmbedLoss
from .Categorical_CrossEntropy import CategoricalCrossEntropy 
from .Binary_CrossEntropy import BinaryCrossEntropy




losses_available_use_prettyname= [
    "binary_crossentropy", 
    "categorical_crossentropy", 
    "contrastive", 
    "cosine_similarity", 
    "in_batch_negatives",
    "mselog",
    "triplet"
]