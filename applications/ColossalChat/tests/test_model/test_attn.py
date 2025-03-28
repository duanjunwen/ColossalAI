import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


def test_qwen2attn(device: str = "cpu"):
    config = Qwen2Config(
        vocab_size=10000,
        hidden_size=768,
        num_hidden_layers=12,
        num_heads=32,
        intermediate_size=3072,
        max_position_embeddings=2048,
        use_cache=True,
    )

    layer_idx = 0
    attention_layer = Qwen2Attention(config, layer_idx=layer_idx).to(device)

    batch_size = 2
    seq_length = 10
    hidden_size = config.hidden_size
    hidden_size // config.num_heads

    # use ones align input
    sin_pos = torch.sin(torch.randn(seq_length)).to(device)
    cos_pos = torch.cos(torch.randn(seq_length)).to(device)
    position_embeddings = (sin_pos, cos_pos)

    hidden_states = torch.ones(batch_size, seq_length, hidden_size).to(device)
    # attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool).to(device)
    attention_mask = None
    # Fwd
    outputs = attention_layer(
        hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask
    )

    print("Output shape:", outputs[0].shape)

    # Bwd
    outputs[0].mean().backward()

    print("Output grad shape:", outputs[0].grad)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    test_qwen2attn(device)
