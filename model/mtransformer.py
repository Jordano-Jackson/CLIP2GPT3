import torch
import torch.nn as nn

class MixTransformerLayer(nn.Module):
    def __init__(self, input_size, head_size, hidden_size):
        super(MixTransformerLayer, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(input_size, num_heads=2, dropout=0.1)
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        
        return x

class MixTransformer(nn.Module):
    def __init__(self, input_size, head_size, hidden_size, num_layers):
        super(MixTransformer, self).__init__()

        self.layers = nn.ModuleList([MixTransformerLayer(input_size, head_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__=='__main__':
    # Example usage:
    input_size = 1024  # Adjust as needed
    head_size = 64
    hidden_size = 2048
    num_layers = 6

    mix_transformer = MixTransformer(input_size, head_size, hidden_size, num_layers)

    # Dummy input data
    input_data = torch.randn(10, input_size)

    # Forward pass
    output_data = mix_transformer(input_data)
    print("Output shape:", output_data.shape)
