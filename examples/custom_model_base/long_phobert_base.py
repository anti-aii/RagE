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

TXT = (
    "Hoàng_Sa và Trường_Sa là <mask> Việt_Nam ."
    + "Đó là điều không_thể chối_cãi ." * 10000
    + "Bằng_chứng lịch_sử , pháp_lý về chủ_quyền của Việt_Nam với 2 quần_đảo này đã và đang được nhiều quốc_gia và cộng_đồng quốc_tế <mask> ."
)

model= SentenceEmbedding(
    model_base= model_base, 
    tokenizer= tokenizer_base, 
    aggregation_hidden_states= False, 
    strategy_pooling= 'dense_first'
)

embedding= model.encode(TXT[0], max_length= 4096, normalize_embedding= 'l2', advance_config_encode= {'pad_to_multiple_of': 256})

print(embedding.shape)