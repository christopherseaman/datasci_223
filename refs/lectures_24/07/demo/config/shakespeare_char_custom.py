
# Model settings
n_layer = 4  # Number of transformer layers
n_head = 4  # Number of attention heads per layer
n_embd = 128  # Size of embedding vectors
dropout = 0.1  # Dropout rate
bias = False  # Whether to use bias in linear layers

# Training settings
max_iters = 3000  # Total number of training iterations
batch_size = 64  # Number of sequences processed in parallel
block_size = 256  # Maximum sequence length
learning_rate = 1e-3  # Initial learning rate

# Output settings
out_dir = 'out-shakespeare-char'  # Directory to save model
device = 'mps'  # Device to use (mps for Apple Silicon)
dtype = 'float32'  # Data type for model parameters
compile = False  # Disable torch.compile for MPS compatibility
