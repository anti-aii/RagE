import os 
import sys
import logging

def _ensure_dir(dir, create_path= True): 
    if not os.path.exists(dir): 
        if create_path: 
            os.makedirs(dir)
        return False 
    return True 

def _create_new_path(org_path, new_folder_name): 
    dir_name, file_name= os.path.split(org_path)

    if file_name: 
        new_dir= os.path.join(dir_name, new_folder_name)
        new_path= os.path.join(new_dir, file_name)

        _ensure_dir(new_dir, create_path= True)

        return new_path 
    else:
        new_path= os.path.join(org_path, new_folder_name)
        _ensure_dir(new_path)
        return new_path


def _print_out(message, line_break= False):
    if line_break: 
        message += '\n'
    sys.stdout.write(message)
    sys.stdout.flush()


def _count_capacity_bit_weight(number_params, storage_units= 'mb'): 
    assert storage_units in ['mb', 'kb', 'gb']

    capacity= number_params * 4

    factor= {'mb': 2, 'kb': 1, 'gb': 3}[storage_units]

    return capacity / (1024 ** factor)

def download(): 
    pass 