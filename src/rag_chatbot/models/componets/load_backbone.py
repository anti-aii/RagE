from transformers import AutoModel, T5EncoderModel 

def load_backbone(model_name, type_backbone= 'bert', using_hidden_states= True, dropout= 0.1): 
    # load backbone support embedding and ranker 
    # Support encoder transformer (eg. BERT, ROBERTA, T5-encoder)
    assert type_backbone in ['bert', 't5']

    if type_backbone== 'bert': 
        model= AutoModel.from_pretrained(model_name, output_hidden_states= using_hidden_states,
                        hidden_dropout_prob = dropout, attention_probs_dropout_prob = dropout)
    elif type_backbone == 't5': 
        model= T5EncoderModel.from_pretrained(model_name, output_hidden_states= using_hidden_states)

    return model 
