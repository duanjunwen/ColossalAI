import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding


def test_qwen2_rotary_embedding(device: str = "cpu", save_tensor: bool = False):
    batch_size = 2
    seq_length = 4
    num_heads = 2
    head_dim = 64

    config = Qwen2Config(
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        max_position_embeddings=2048,
    )

    # init input
    input_tensor = torch.randn(batch_size, seq_length, num_heads, head_dim)
    positions = torch.arange(seq_length)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    rotary_embedding = Qwen2RotaryEmbedding(config)

    # Fwd
    input_tensor_cpu = input_tensor.clone().to(device)
    rotary_embedding_cpu = rotary_embedding.to(device)
    positions_cpu = positions.to(device)
    cos_cpu, sin_cpu = rotary_embedding_cpu(input_tensor_cpu, positions_cpu)

    input_tensor_cuda = input_tensor.clone().to(device)
    rotary_embedding_cuda = rotary_embedding.to(device)
    positions_cuda = positions.to(device)
    cos_cuda, sin_cuda = rotary_embedding_cuda(input_tensor_cuda, positions_cuda)

    max_diff = (cos_cpu - cos_cuda.to(device)).abs().max()
    max_diff = (sin_cpu - sin_cuda.to(device)).abs().max()

    if save_tensor:
        tensor_pt = {
            "input": input_tensor_cuda.to("cpu"),
            "positions_cuda": positions_cuda.to("cpu"),
            "cos_cuda": cos_cuda.to("cpu"),
            "sin_cuda": sin_cuda.to("cpu"),
        }
        torch.save(tensor_pt, f"./tests/tensor_log/{device}_RotaryEmbedding.pt")
        print(f"Tensor save at ./tests/tensor_log/{device}_RotaryEmbedding.pt")
        print(f"Max Abs error: {max_diff.item()}")
        assert max_diff < 1e-5, f"Max Abs error exceeded 1e-5: {max_diff.item()}"


if __name__ == "__main__":
    save_tensor = False
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    test_qwen2_rotary_embedding(device, save_tensor)
