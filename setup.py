from setuptools import setup, find_packages

_dep= ["transformers>= 4.31.0", 
       "torch>= 2.0.0", 
       "bitsandbytes>= 0.39.1", 
       "loralib>= 0.1.2", 
       "peft >= 0.4.0", 
       "pandas >= 2.0.3",
       "scipy >= 1.11.1", 
       "scikit-learn >= 1.3.0" 

]

with open("README.md", 'r', encoding= 'utf-8') as f: 
    long_description= f.read() 

setup(
    name='rag_chatbot',
    version= '0.4.6', 
    description= 'A library that supports building the fastest and lightest chatbot or Q&A system for Vietnamese, using Retrieval Augmented Generation (RAG)', 
    long_description= long_description,
    author= 'Nguyễn Tiến Đạt', 
    author_email= 'nduc0231@gmail.com', 
    maintainer= 'Nguyễn Tiến Đạt',
    packages= find_packages("src"), 
    package_dir= {'': 'src'},
    package_data={"": ['**/*.yml']},
    install_requires= _dep,
    python_requires= ">=3.10"
)