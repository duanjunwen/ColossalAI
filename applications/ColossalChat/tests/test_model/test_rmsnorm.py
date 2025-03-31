import copy

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from utils import assert_function_close


def test_qwen2rmsnorm(device: str = "cpu", save_tensor: bool = False):
    hidden_size = 768
    config = Qwen2Config(hidden_size=hidden_size)

    rms_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    input_tensor = torch.ones(1, 128, hidden_size)

    # CPU 测试
    rms_norm_cpu = rms_norm
    input_tensor_cpu = input_tensor
    output_cpu = rms_norm_cpu(input_tensor_cpu)

    # Fwd
    rms_norm_cuda = copy.deepcopy(rms_norm).to(device)
    input_tensor_cuda = copy.deepcopy(input_tensor).to(device)

    output_cuda = rms_norm_cuda(input_tensor_cuda)
    max_diff = torch.max(torch.abs(output_cpu - output_cuda.to("cpu")))

    if max_diff < 1e-3:
        print("Fwd Pass")
    else:
        print(f"Fwd Fail, abs error: {max_diff}")

    # Bwd
    output_cpu.mean().backward()
    output_cuda.mean().backward()

    # assert output grad close
    # max_diff = torch.max(torch.abs(output_cpu.grad - output_cuda.grad))
    # print(f"output max_diff {max_diff}")
    # if max_diff < 1e-3:
    #     print("Bwd Pass")
    # else:
    #     print(f"Bwd Fail, abs error: {max_diff}")

    if save_tensor:
        tensor_pt = {
            "input": input_tensor_cuda.to("cpu"),
            "output": output_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/{device}_RMSNorm.pt")
        print(f"Tensor save at ./tests/tensor_log/{device}_RMSNorm.pt")


def assert_mlp_close():
    assert_function_close("Qwen2RMSNorm", f"./tests/tensor_log/cuda_RMSNorm.pt", f"./tests/tensor_log/npu_RMSNorm.pt")


if __name__ == "__main__":
    save_tensor = True
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    test_qwen2rmsnorm(device, save_tensor)
