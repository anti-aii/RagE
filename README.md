# RAG Engine: Retrieval Augmented Generation Engine
RagE (Rag Engine) is a tool designed to facilitate the construction and training of components within the Retrieval-Augmented-Generation (RAG) model. It also offers algorithms to support retrieval and provides pipelines for evaluating models. Moreover, it fosters rapid development of question answering systems and chatbots based on the RAG model.

Currently, we have completed the basic training pipeline for the model, but there is still much to be done due to limited resources. However, with this library, we are continuously updating and developing. Additionally, we will consistently publish the models that we train.
## Installation 🔥
- We recommend `python 3.9` or higher, `torch 2.0.0` or higher, `transformers 4.31.0` or higher.

- Currently, you can only download from the source, however, in the future, we will upload it to PyPI. RagE can be installed from source with the following commands: 
```
git clone https://github.com/anti-aii/RagE.git
cd rage
pip install -e .
```
## Quick start 🥮
- [1. Initialize the model](#initialize_model)
- [2. Load model from Huggingface Hub](#download_hf)
- [3. Training](#training)
- [4. List of pretrained models](#list_pretrained)

We have detailed instructions for using our models for inference. See [notebook](notebook)
### 1. Initialize the model
<a name= 'initialize_model'></a>
Let's initalize the SentenceEmbedding model  

```python
>>> import torch 
>>> from pyvi import ViTokenizer
>>> from rage import SentenceEmbedding
>>> device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>>> model= SentenceEmbedding(model_name= "vinai/phobert-base-v2", torch_dtype= torch.float32, aggregation_hidden_states= False, strategy_pooling= "dense_first")
>>> model.to(device)
SentenceEmbeddingConfig(model_base: {'model_type_base': 'RobertaModel', 'model_name': 'vinai/phobert-base-v2', 'type_backbone': 'mlm', 'required_grad_base_model': True, 'aggregation_hidden_states': False, 'concat_embeddings': False, 'dropout': 0.1, 'quantization_config': None}, pooling: {'strategy_pooling': 'dense_first'})
```
Then, we can show the number of parameters in the model.
```python 
>>> model.summary_params()
trainable params: 135588864 || all params: 135588864 || trainable%: 100.0
>>> model.summary()
+---------------------------+-------------+------------------+
|        Layer (type)       |    Params   | Trainable params |
+---------------------------+-------------+------------------+
|    model (RobertaModel)   | 134,998,272 |    134998272     |
| pooling (PoolingStrategy) |   590,592   |      590592      |
|       drp1 (Dropout)      |      0      |        0         |
+---------------------------+-------------+------------------+
```
Now we can use the SentenceEmbedding model to encode the input words. The output of the model will be a matrix in the shape of (batch, dim). Additionally, we can load weights that we have previously trained and saved.
``` python
>>> model.load("best_sup_general_embedding_phobert2.pt", key= False)
>>> sentences= ["Tôi đang đi học", "Bạn tên là gì?",]
>>> sentences= list(map(lambda x: ViTokenizer.tokenize(x), sentences))
>>> model.encode(sentences, batch_size= 1, normalize_embedding= "l2", return_tensors= "np", verbose= 1)
2/2 [==============================] - 0s 43ms/Sample
array([[ 0.00281098, -0.00829096, -0.01582766, ...,  0.00878178,
         0.01830498, -0.00459659],
       [ 0.00249859, -0.03076724,  0.00033016, ...,  0.01299141,
        -0.00984358, -0.00703243]], dtype=float32)
```
### 2. Load model from Huggingface Hub
<a name= 'download_hf'> </a>

First, download a pretrained model. 
```python
>>> model= SentenceEmbedding.from_pretrained('anti-ai/VieSemantic-base')
```
Then, we encode the input sentences and compare their similarity.
```python
>>> sentences = ["Nó rất thú_vị", "Nó không thú_vị ."]
>>> output= model.encode(sentences, batch_size= 1, return_tensors= 'pt')
>>> torch.cosine_similarity(output[0].view(1, -1), output[1].view(1, -1)).cpu().tolist()
2/2 [==============================] - 0s 40ms/Sample
[0.5605039596557617]
```
### 3. Training 
We have some examples of training for SentenceEmbedding, ReRanker, and LLM models. Additionally, you can rely on the optimal parameters we used for specific tasks and datasets. See [examples](examples)
<a name= 'training'></a>

### 9. List of pretrained models
<a name= 'list_pretrained'></a>
This list will be updated with our prominent models. Our models will primarily aim to support Vietnamese language.
Additionally, you can access our datasets and pretrained models by visiting https://huggingface.co/anti-ai.

| Model Name | Model Type | #params | checkpoint|
| - | - | - | - |
| anti-ai/ViEmbedding-base | SentenceEmbedding | 135.5M |[model](https://huggingface.co/anti-ai/ViEmbedding-base) |
| anti-ai/BioViEmbedding-base-unsup | SentenceEmbedding | 135.5M |[model](https://huggingface.co/anti-ai/BioViEmbedding-base-unsup) | 
| anti-ai/VieSemantic-base | SentenceEmbedding | 135.5M |[model](https://huggingface.co/anti-ai/VieSemantic-base) |
| anti-ai/phobert-ranker-base | ReRanker | 135.6M |[model](https://huggingface.co/anti-ai/phobert-ranker-base) |


## Contacts
If you have any questions about this repo, please contact me (nduc0231@gmail.com)