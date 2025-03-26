import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


def test_qwen2rmsnorm():
    # 初始化配置
    hidden_size = 768
    config = Qwen2Config(hidden_size=hidden_size)

    # 创建模型实例
    rms_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 创建输入数据
    input_tensor = torch.ones(1, 128, hidden_size)

    # CPU 测试
    rms_norm_cpu = rms_norm.to("cpu")
    input_tensor_cpu = input_tensor.to("cpu")
    with torch.no_grad():
        output_cpu = rms_norm_cpu(input_tensor_cpu)

    # CUDA 测试
    if torch.cuda.is_available():
        rms_norm_cuda = rms_norm.to("cuda")
        input_tensor_cuda = input_tensor.to("cuda")
        with torch.no_grad():
            output_cuda = rms_norm_cuda(input_tensor_cuda).to("cpu")

        # 对比精度
        max_diff = torch.max(torch.abs(output_cpu - output_cuda))
        tensor_pt = {
            "input": input_tensor_cuda.to("cpu"),
            "output": output_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/cuda/RMSNorm.pt")

        if max_diff < 1e-3:
            print("测试通过，CPU 和 CUDA 输出的最大差值在允许范围内。")
        else:
            print(f"测试失败，CPU 和 CUDA 输出的最大差值为: {max_diff}")
    else:
        print("CUDA 不可用，无法进行 CUDA 与 CPU 的对比测试。")


if __name__ == "__main__":
    test_qwen2rmsnorm()
