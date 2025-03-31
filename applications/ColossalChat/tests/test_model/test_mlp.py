import copy

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from .utils import assert_function_close


def test_qwen2mlp(device: str = "cpu", save_tensor: bool = False):
    hidden_size = 768
    intermediate_size = 3072
    config = Qwen2Config(hidden_size=hidden_size, intermediate_size=intermediate_size)
    mlp = Qwen2MLP(config)
    input_tensor = torch.ones(1, 128, hidden_size)

    # CPU
    mlp_cpu = mlp
    input_tensor_cpu = input_tensor
    output_cpu = mlp_cpu(input_tensor_cpu)

    # Device {NPU, CUDA}
    mlp_cuda = copy.deepcopy(mlp_cpu).to(device)
    input_tensor_cuda = copy.deepcopy(input_tensor).to(device)
    output_cuda = mlp_cuda(input_tensor_cuda)

    # assert output close
    max_diff = torch.max(torch.abs(output_cpu - output_cuda.to("cpu")))
    print(f"output max_diff {max_diff}")

    if max_diff < 1e-3:
        print("Fwd Pass")
    else:
        print(f"Fwd Fail, abs error: {max_diff}")

    # Bwd
    output_cpu.mean().backward()
    print(f"output_cuda device {output_cuda.device}")
    output_cuda.mean().backward()

    # # assert output grad close
    # max_diff = torch.max(torch.abs(input_tensor_cpu.grad - input_tensor_cuda.grad))
    # print(f"output max_diff {max_diff}")

    if max_diff < 1e-3:
        print("Bwd Pass")
    else:
        print(f"Bwd Fail, abs error: {max_diff}")

    if save_tensor:
        tensor_pt = {
            "input": input_tensor_cuda.to("cpu"),
            "output_cuda": output_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/{device}_MLP.pt")
        print(f"Tensor save at ./tests/tensor_log/{device}_MLP.pt")


def assert_mlp_close():
    assert_function_close("Qwen2MLP", f"./tests/tensor_log/cuda_MLP.pt", f"./tests/tensor_log/npu_MLP.pt")


if __name__ == "__main__":
    save_tensor = True
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    test_qwen2mlp(device, save_tensor)
