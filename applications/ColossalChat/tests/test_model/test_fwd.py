import os

import torch
from torch.testing import assert_close


def compare_pt_files(directory):
    # 存储文件名的字典，键为 ops_name，值为包含 cuda 和 npu 文件路径的列表
    file_pairs = {}

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            if "cuda_" in filename:
                ops_name = filename.replace("cuda_", "").replace(".pt", "")
                if ops_name not in file_pairs:
                    file_pairs[ops_name] = [None, None]
                file_pairs[ops_name][0] = os.path.join(directory, filename)
            elif "npu_" in filename:
                ops_name = filename.replace("npu_", "").replace(".pt", "")
                if ops_name not in file_pairs:
                    file_pairs[ops_name] = [None, None]
                file_pairs[ops_name][1] = os.path.join(directory, filename)

    # 遍历 file_pairs 字典，加载文件并比较
    for ops_name, (cuda_file, npu_file) in file_pairs.items():
        if cuda_file and npu_file:
            try:
                # 加载文件
                cuda_dict = torch.load(cuda_file)
                npu_dict = torch.load(npu_file)

                # 检查两个字典的键是否相同
                if set(cuda_dict.keys()) != set(npu_dict.keys()):
                    print(f"对于 ops_name {ops_name}，两个字典的键不相同。")
                    continue

                # 逐个键比较值
                all_close = True
                for key in cuda_dict.keys():
                    try:
                        assert_close(cuda_dict[key], npu_dict[key])
                    except AssertionError:
                        print(f"对于 ops_name {ops_name}，键 {key} 的值不匹配。")
                        all_close = False

                if all_close:
                    print(f"对于 ops_name {ops_name}，所有键的值都匹配。")

            except Exception as e:
                print(f"在处理 ops_name {ops_name} 时出现错误: {e}")
        else:
            print(f"对于 ops_name {ops_name}，缺少对应的 cuda 或 npu 文件。")


if __name__ == "__main__":
    folder = "/home/duanjunwen/ColossalAI/applications/ColossalChat/tests/tensor_log/"

    compare_pt_files(folder)
