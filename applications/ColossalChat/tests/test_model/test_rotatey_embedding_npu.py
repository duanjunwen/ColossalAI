import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from colossalai.testing.random import seed_all


def run_qwen2_rotary_embedding():
    seed_all(42)
    batch_size = 2
    seq_length = 4
    num_heads = 2
    head_dim = 64
    config = Qwen2Config(
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        max_position_embeddings=2048,  # 根据实际情况调整
    )
    input_tensor_cuda = torch.ones(batch_size, seq_length, num_heads, head_dim, device="npu")
    rotary_embedding_cuda = Qwen2RotaryEmbedding(config, device="npu")
    positions_cuda = torch.arange(seq_length, device="npu").unsqueeze(0).expand(batch_size, -1)
    cos_cuda, sin_cuda = rotary_embedding_cuda(input_tensor_cuda, positions_cuda)

    tensor_pt = {
        "input": input_tensor_cuda.to("cpu"),
        "positions_cuda": positions_cuda.to("cpu"),
        "cos_cuda": cos_cuda.to("cpu"),
        "sin_cuda": sin_cuda.to("cpu"),
    }
    torch.save(tensor_pt, f"./tests/tensor_log/npu/RotaryEmbedding.pt")


def test_qwen2_rotary_embedding():
    # 定义输入参数
    batch_size = 2
    seq_length = 4
    num_heads = 2
    head_dim = 64

    # 创建 Qwen2Config 对象
    config = Qwen2Config(
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        max_position_embeddings=2048,  # 根据实际情况调整
    )

    # 创建输入张量
    input_tensor = torch.randn(batch_size, seq_length, num_heads, head_dim)
    positions = torch.arange(seq_length)
    # 将 positions 扩展为二维张量
    positions = positions.unsqueeze(0).expand(batch_size, -1)

    # 创建 Qwen2RotaryEmbedding 实例
    rotary_embedding = Qwen2RotaryEmbedding(config)

    # 在 CPU 上运行
    input_tensor_cpu = input_tensor.clone().cpu()
    rotary_embedding_cpu = rotary_embedding.cpu()
    positions_cpu = positions.cpu()
    cos_cpu, sin_cpu = rotary_embedding_cpu(input_tensor_cpu, positions_cpu)

    if torch.npu.is_available():
        input_tensor_cuda = input_tensor.clone().npu()
        rotary_embedding_cuda = rotary_embedding.npu()
        positions_cuda = positions.npu()
        cos_cuda, sin_cuda = rotary_embedding_cuda(input_tensor_cuda, positions_cuda)

        max_diff = (cos_cpu - cos_cuda.cpu()).abs().max()
        max_diff = (sin_cpu - sin_cuda.cpu()).abs().max()
        tensor_pt = {
            "input": input_tensor_cuda.to("cpu"),
            "positions_cuda": positions_cuda.to("cpu"),
            "cos_cuda": cos_cuda.to("cpu"),
            "sin_cuda": sin_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/npu/RotaryEmbedding.pt")

        print(f"CPU 和 CUDA 输出的最大差值: {max_diff.item()}")
        assert max_diff < 1e-5, f"精度差异过大，最大差值: {max_diff.item()}"
    else:
        print("CUDA 不可用，无法进行对比测试。")


if __name__ == "__main__":
    run_qwen2_rotary_embedding()  # fwd
    # test_qwen2_rotary_embedding() # compare with cpu
