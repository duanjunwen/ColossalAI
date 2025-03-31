from typing import Dict

import torch


def compare_dict(
    name: str = "",
    dict1: Dict = None,
    dict2: Dict = None,
):
    all_match = True
    keys1 = set(dict1.keys())
    set(dict2.keys())
    for key in keys1:
        tensor1 = dict1[key]
        tensor2 = dict2[key]
        if not torch.allclose(tensor1, tensor2):
            # assert_close(tensor1, tensor2)
            print(
                f" {name} key: {key} tensor shape {tensor1.shape} dtype {tensor1.dtype} compare fail; \n {tensor1} \nvs\n {tensor2}"
            )
            all_match = False
        else:
            print(f" {name} key: {key} tensor shape {tensor2.shape} dtype {tensor2.dtype} compare pass")

    if all_match:
        print(f"{name} tensor compare all pass")


def assert_function_close(name: str = "", path1=None, path2=None):
    dict_nv = torch.load(path1)
    dict_npu = torch.load(path2)
    compare_dict(f"{name}", dict_nv, dict_npu)
