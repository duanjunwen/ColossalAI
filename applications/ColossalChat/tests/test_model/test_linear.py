import torch
from utils import assert_function_close


def assert_linear_close():
    # assert_function_close("Qwen2MLP", f"./tests/tensor_log/cuda_MLP_tensor.pt", f"./tests/tensor_log/npu_MLP_tensor.pt")
    assert_function_close(
        "Linear",
        f"./tests/tensor_log/cuda_tensor_rank0_Linear_0.pt",
        f"./tests/tensor_log/npu_tensor_rank0_Linear_0.pt",
    )


if __name__ == "__main__":
    save_tensor = True
    load_tensor = False
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    # test_qwen2mlp(device, save_tensor, load_tensor)
    assert_linear_close()
