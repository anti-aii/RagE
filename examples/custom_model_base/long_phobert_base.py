import torch 

from transformers import RobertaForMaskedLM, AutoTokenizer
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

from rage import SentenceEmbedding, Reranker

class PhoBertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value = None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())
        return super().forward(hidden_states, 
                               is_index_masked=is_index_masked, 
                               is_index_global_attn=is_index_global_attn, 
                               is_global_attn=is_global_attn,
                               attention_mask=attention_mask, 
                               output_attentions=output_attentions)

class PhoBertLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            layer.attention.self = PhoBertLongSelfAttention(config, layer_id=i)


tokenizer_base = AutoTokenizer.from_pretrained("bluenguyen/longformer-phobert-base-4096")
model_base = PhoBertLongForMaskedLM.from_pretrained("bluenguyen/longformer-phobert-base-4096")


model_embed= SentenceEmbedding(
    model_base= model_base, 
    tokenizer= tokenizer_base,
    aggregation_hidden_states= False, 
    strategy_pooling= "dense_first"
)

print('Model Embedding: {}'.format(model_embed))


model_ranker= Reranker(
    model_base= model_base, 
    tokenizer= tokenizer_base,
    aggregation_hidden_states= False, 
    strategy_pooling= "dense_first"
)


print('Model Reranker: {}'.format(model_ranker))
