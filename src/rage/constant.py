import os 

RESPONSE_RULE= os.path.join(os.path.dirname(__file__), "utils/rule_response/response_rule.yml")

CONFIG_MODEL= "config_model.json"
CONFIG_TRAIN= "config_train.json"
CONFIG_DATASET= "config_dataset.json"
PYTORCH_WEIGHTS_SAVE_PRETRAIN= "pytorch_model.bin" ## only use for save_pretrained's method of hf


## DEFIND TASK TRAINING
EMBEDDING_RANKER_NUMERICAL= "label_is_numerical"
EMBEDDING_CONTRASTIVE= "label_is_other_embedding"
EMBEDDING_TRIPLET= "label_is_pos_neg"
EMBEDDING_IN_BATCH_NEGATIVES= "multi_negatives"