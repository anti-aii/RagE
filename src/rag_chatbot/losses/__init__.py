import torch.nn as nn 
from .CosineSimilarityLoss import CosineSimilarityLoss
from .MSELogLoss import MSELogLoss
from .ContrastiveLoss import ContrastiveLoss
from .TripletLoss import TripletLoss 
from .InBatchNegatives import InBatchNegativeLoss



## DEFIND LOSSES FOR TRAINING EMBEDDING AND RANKER MODEL
BINARY_CROSS_ENTROPY_LOSS= "binary_crossentropy"
CATEGORICAL_CROSS_ENTROPY_LOSS= "categorical_crossentropy"
MSE_LOGARIT= "mselog"
COSINE_SIMILARITY_LOSS= "cosine_similarity"
CONTRASTIVE_LOSS= "contrastive"
TRIPLET_LOSS="triplet"
IN_BATCH_NEGATIVES_LOSS= "in_batch_negatives"


## DEFIND TASK TRAINING
EMBEDDING_RANKER_NUMERICAL= "label_is_numerical"
EMBEDDING_CONTRASTIVE= "label_is_other_embedding"
EMBEDDING_TRIPLET= "label_is_pos_neg"
EMBEDDING_IN_BATCH_NEGATIVES= "multi_negatives"

## DEFINE RULE
def rule_loss_task(type_loss:str):
    ### numerical embedding and ranker 
    if type_loss in [BINARY_CROSS_ENTROPY_LOSS, 
                     CATEGORICAL_CROSS_ENTROPY_LOSS, 
                     MSE_LOGARIT, 
                     COSINE_SIMILARITY_LOSS]:
        return EMBEDDING_RANKER_NUMERICAL
    
    if type_loss in [CONTRASTIVE_LOSS]: 
        return EMBEDDING_CONTRASTIVE
    
    if type_loss in [TRIPLET_LOSS]: 
        return EMBEDDING_TRIPLET
    
    if type_loss in [IN_BATCH_NEGATIVES_LOSS]: 
        return EMBEDDING_IN_BATCH_NEGATIVES

## DEFINE CRITERION
_criterion= {
            BINARY_CROSS_ENTROPY_LOSS: nn.BCEWithLogitsLoss(),
            CATEGORICAL_CROSS_ENTROPY_LOSS: nn.CrossEntropyLoss(),
            MSE_LOGARIT: MSELogLoss(),
            COSINE_SIMILARITY_LOSS: CosineSimilarityLoss(), 
            CONTRASTIVE_LOSS: ContrastiveLoss(margin= 0.5), 
            TRIPLET_LOSS: TripletLoss(margin= 0.5), 
            IN_BATCH_NEGATIVES_LOSS: InBatchNegativeLoss(temp= 0.05)
}