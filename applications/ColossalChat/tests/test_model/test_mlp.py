import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


def test_qwen2mlp():
    # 初始化配置
    hidden_size = 768
    intermediate_size = 3072
    config = Qwen2Config(hidden_size=hidden_size, intermediate_size=intermediate_size)

    # 创建模型实例
    mlp = Qwen2MLP(config)

    # 创建输入数据
    input_tensor = torch.ones(1, 128, hidden_size)

    # CPU 测试
    mlp_cpu = mlp.to("cpu")
    input_tensor_cpu = input_tensor.to("cpu")
    with torch.no_grad():
        output_cpu = mlp_cpu(input_tensor_cpu)

    # CUDA 测试
    if torch.cuda.is_available():
        mlp_cuda = mlp.to("cuda")
        input_tensor_cuda = input_tensor.to("cuda")
        with torch.no_grad():
            output_cuda = mlp_cuda(input_tensor_cuda).to("cpu")

        # 对比精度
        max_diff = torch.max(torch.abs(output_cpu - output_cuda))
        print(f"max_diff {max_diff}")
        tensor_pt = {
            "input": input_tensor_cuda.to("cpu"),
            "output_cuda": output_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/cuda/MLP.pt")

        if max_diff < 1e-3:
            print("测试通过，CPU和CUDA输出的最大差值在允许范围内。")
        else:
            print(f"测试失败，CPU和CUDA输出的最大差值为: {max_diff}")
    else:
        print("CUDA不可用，无法进行CUDA与CPU的对比测试。")


if __name__ == "__main__":
    test_qwen2mlp()
