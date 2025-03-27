import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def test_qwen2attn():
    # 初始化配置
    config = Qwen2Config(
        vocab_size=10000,
        hidden_size=768,
        num_hidden_layers=12,
        num_heads=32,
        intermediate_size=3072,
        max_position_embeddings=2048,
        use_cache=True,
    )

    # 初始化注意力层并将其移动到CUDA设备上
    layer_idx = 0  # 层索引
    attention_layer = Qwen2Attention(config, layer_idx=layer_idx).to(device)

    # 构造输入数据
    batch_size = 2
    seq_length = 10
    hidden_size = config.hidden_size
    hidden_size // config.num_heads

    # 生成位置编码的sin和cos部分，调整形状使其和q、k匹配
    sin_pos = torch.sin(torch.randn(seq_length)).to(device)
    cos_pos = torch.cos(torch.randn(seq_length)).to(device)
    position_embeddings = (sin_pos, cos_pos)  # 以元组形式传入

    # 构造其他输入并将其移动到CUDA设备上
    hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)
    # attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool).to(device)
    attention_mask = None

    # 执行前向传播
    outputs = attention_layer(
        hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask
    )

    # 输出结果
    print("Output shape:", outputs[0].shape)  # 输出形状: (batch_size, seq_length, hidden_size)
    print(
        "Attention weights shape:", outputs[1].shape
    )  # 注意力权重形状: (batch_size, num_heads, seq_length, seq_length)
