import os 
import sys
import logging

def ensure_dir(dir, create_path= True): 
    if not os.path.exists(dir): 
        if create_path: 
            os.makedirs(dir)
        return False 
    return True 


def print_out(message, line_break= False):
    if line_break: 
        message += '\n'
    sys.stdout.write(message)
    sys.stdout.flush()


def count_capacity_bit_weight(number_params, storage_units= 'mb'): 
    assert storage_units in ['mb', 'kb', 'gb']

    capacity= number_params * 4

    factor= {'mb': 2, 'kb': 1, 'gb': 3}[storage_units]

    return capacity / (1024 ** factor)

def download(): 
    pass 