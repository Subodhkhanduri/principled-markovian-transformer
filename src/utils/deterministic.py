# src/utils/deterministic.py
import random
import numpy as np
import torch

def set_deterministic_mode(seed=42):
    """Set all random seeds for reproducible results."""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)


# Ensure deterministic training
def create_deterministic_dataloader(dataset, batch_size, seed=42):
    """Create a deterministic data loader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        drop_last=True
    )
