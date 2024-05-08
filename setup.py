from setuptools import setup, find_packages

_dep= ["transformers>= 4.31.0", 
       "torch>= 2.0.0", 
       "bitsandbytes>= 0.39.1", 
       "loralib>= 0.1.2", 
       "datasets >= 2.17.1",
       "peft >= 0.4.0", 
       "pandas >= 2.0.3",
       "scipy >= 1.11.1", 
       "scikit-learn >= 1.3.0", 
       "huggingface-hub >= 0.19.4", 
       "prettytable >= 3.10.0", 
       "wandb >= 0.16.6"

]

with open("README.md", 'r', encoding= 'utf-8') as f: 
    long_description= f.read() 

description= """RagE (Rag Engine) - A tool supporting the construction and training of components of the Retrieval-Augmented-Generation (RAG) model. /
It also facilitates the rapid development of Q&A systems and chatbots following the RAG model."""
setup(
    name='rage',
    version= '1.1.0dev', 
    description= description, 
    long_description= long_description,
    author= 'Nguyễn Tiến Đạt', 
    author_email= 'nduc0231@gmail.com', 
    maintainer= 'Nguyễn Tiến Đạt',
    packages= find_packages("src"), 
    package_dir= {'': 'src'},
    package_data={"": ['**/*.yml']},
    install_requires= _dep,
    python_requires= ">=3.9"
)