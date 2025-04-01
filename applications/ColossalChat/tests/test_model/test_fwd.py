import os

import torch


def compare_pt_files(dir):
    pt_files_dir1 = [os.path.join(dir, f) for f in os.listdir(dir) if (f.startwith("cuda") and f.endswith(".pt"))]
    pt_files_dir2 = [os.path.join(dir, f) for f in os.listdir(dir) if (f.startwith("npu") and f.endswith(".pt"))]

    file_names_dir1 = [os.path.basename(f) for f in pt_files_dir1]
    file_names_dir2 = [os.path.basename(f) for f in pt_files_dir2]

    common_file_names = set(file_names_dir1).intersection(set(file_names_dir2))

    for file_name in common_file_names:
        file_path1 = os.path.join(dir, file_name)
        file_path2 = os.path.join(dir, file_name)
        # print(f"file_path1 {file_path1}")
        # if 'RMSNorm.pt' not in file_path1:
        #     continue

        try:

            dict1 = torch.load(file_path1)
            dict2 = torch.load(file_path2)

            keys1 = set(dict1.keys())
            keys2 = set(dict2.keys())
            if keys1 != keys2:
                print(f" {file_name} key not compare: {keys1.symmetric_difference(keys2)}")
                continue

            all_match = True
            for key in keys1:
                tensor1 = dict1[key]
                tensor2 = dict2[key]
                if not torch.allclose(tensor1, tensor2):
                    # assert_close(tensor1, tensor2)
                    print(
                        f" {file_name} key: {key} tensor shape {tensor1.shape} dtype {tensor1.dtype} compare fail; \n {tensor1} \nvs\n {tensor2}"
                    )
                    all_match = False
                else:
                    print(f" {file_name} key: {key} tensor shape {tensor2.shape} dtype {tensor2.dtype} compare pass")

            if all_match:
                print(f"{file_name} tensor compare all pass")

        except Exception as e:
            print(f"Error in processing {file_name} error {e}")


if __name__ == "__main__":
    folder = "/home/duanjunwen/ColossalAI/applications/ColossalChat/tests/tensor_log/"

    compare_pt_files(folder)
