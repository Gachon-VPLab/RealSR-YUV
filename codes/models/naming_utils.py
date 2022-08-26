'''
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
'''


def string_changer(original: str, parser: str, index: int, target: str, rdb: bool):
    new_str = ''

    parsed_string = original.split(parser)

    if rdb: parsed_string[index] = target + (parsed_string[2])[3]
    else: parsed_string[index] = target

    for i in range(len(parsed_string)):
        if i != len(parsed_string) - 1:
            new_str += (parsed_string[i] + parser)
        else:
            new_str += parsed_string[i]

    return new_str
    
    
def match_namespace(dictionary: dict):
    new_key = ''

    for k in dictionary.copy().keys():
        # change key
        if k.startswith('body'):
            new_key = string_changer(k, '.', 0, 'RRDB_trunk', False)
            new_key = string_changer(new_key, '.', 2, 'RDB', True)
        elif k.startswith('conv_body'):
            new_key = string_changer(k, '.', 0, 'trunk_conv', False)
        elif k.startswith('conv_up1'):
            new_key = string_changer(k, '.', 0, 'upconv1', False)
        elif k.startswith('conv_up2'):
            new_key = string_changer(k, '.', 0, 'upconv2', False)
        elif k.startswith('conv_hr'):
            new_key = string_changer(k, '.', 0, 'HRconv', False)
        else: new_key = k

        dictionary[new_key] = dictionary.pop(k)