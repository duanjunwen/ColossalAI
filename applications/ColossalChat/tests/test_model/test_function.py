import torch
import torch.nn.functional as F

from .utils import assert_function_close


def test_matmul(device: str = "cpu", dtype=torch.float32, save_tensor: bool = False):
    input1_cpu = torch.ones(4, 8).to(dtype=dtype)
    input2_cpu = torch.ones(8, 4).to(dtype=dtype)
    input1_cuda = input1_cpu.to(device=device, dtype=dtype)
    input2_cuda = input2_cpu.to(device=device, dtype=dtype)

    matmul_result_cpu = torch.matmul(input1_cpu, input2_cpu)
    matmul_result_cuda = torch.matmul(input1_cuda, input2_cuda).cpu()

    if save_tensor:
        tensor_pt = {
            "input1": input1_cuda.to("cpu"),
            "input2": input2_cuda.to("cpu"),
            "output_cuda": matmul_result_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/{device}_matmul.pt")
        print(f"Tensor save at ./tests/tensor_log/{device}_matmul.pt")

    # assert_close(matmul_result_cpu, matmul_result_cuda)
    matmul_max_diff = (matmul_result_cpu - matmul_result_cuda).abs().max().item()
    print(f"torch.matmul max diff on cpu vs {device}: {matmul_max_diff}")


def test_dropout(device: str = "cpu", dtype=torch.float32, save_tensor: bool = False):
    input_dropout = torch.ones(10, 20).to(dtype=dtype)
    input_dropout_cuda = input_dropout.to(device=device, dtype=dtype)

    dropout_result_cpu = F.dropout(input_dropout, p=0.5, training=False)
    dropout_result_cuda = F.dropout(input_dropout_cuda, p=0.5, training=False).cpu()

    if save_tensor:
        tensor_pt = {
            "input": input_dropout_cuda.to("cpu"),
            "output_cuda": dropout_result_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/{device}_dropout.pt")
        print(f"Tensor save at ./tests/tensor_log/{device}_dropout.pt")

    # assert_close(dropout_result_cpu, dropout_result_cuda)
    dropout_max_diff = (dropout_result_cpu - dropout_result_cuda).abs().max().item()
    print(f"nn.functional.dropout max diff on cpu vs {device}: {dropout_max_diff}")


def test_softmax(device: str = "cpu", dtype=torch.float32, save_tensor: bool = False):
    input_softmax = torch.ones(10, 20).to(dtype=dtype)
    input_softmax_cuda = input_softmax.to(device=device, dtype=dtype)

    softmax_result_cpu = F.softmax(input_softmax, dim=1)
    softmax_result_cuda = F.softmax(input_softmax_cuda, dim=1).cpu()

    if save_tensor:
        tensor_pt = {
            "input": input_softmax_cuda.to("cpu"),
            "output_cuda": softmax_result_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/{device}_softmax.pt")
        print(f"Tensor save at ./tests/tensor_log/{device}_softmax.pt")

    # assert_close(softmax_result_cpu, softmax_result_cuda
    softmax_max_diff = (softmax_result_cpu - softmax_result_cuda).abs().max().item()
    print(f"nn.functional.softmax max diff on cpu vs {device}: {softmax_max_diff}")


def assert_close():
    # assert test_matmul
    assert_function_close("matmul", f"./tests/tensor_log/cuda_matmul.pt", f"./tests/tensor_log/npu_matmul.pt")

    # assert test_dropout
    assert_function_close("dropout", f"./tests/tensor_log/cuda_dropout.pt", f"./tests/tensor_log/npu_dropout.pt")

    # assert test_softmax
    assert_function_close("softmax", f"./tests/tensor_log/cuda_softmax.pt", f"./tests/tensor_log/npu_softmax.pt")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    dtype = torch.bfloat16
    save_tensor = True
    test_matmul(device, dtype, save_tensor)
    test_dropout(device, dtype, save_tensor)
    test_softmax(device, dtype, save_tensor)

    # assert_close()
