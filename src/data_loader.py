import scipy.io
import numpy as np
import os

def load_svhn_data(data_dir, split='train'):
    """
    Loads SVHN cropped digit data from .mat files.
    
    Args:
        data_dir (str): Path to the folder containing .mat files.
        split (str): 'train' or 'test'.
        
    Returns:
        X (np.array): Images of shape (N, 32, 32, 3) -> Normalized [0, 1]
        y (np.array): Labels of shape (N,) -> Corrected so '0' is class 0 (not 10)
    """
    file_path = os.path.join(data_dir, f'{split}_32x32.mat')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please download it from http://ufldl.stanford.edu/housenumbers/")

    print(f"Loading {split} data from {file_path}...")
    mat_data = scipy.io.loadmat(file_path)
    
    # The .mat file has X shape: (32, 32, 3, N) -> (Height, Width, Channels, Batch)
    # We want standard shape: (N, 32, 32, 3) for visualization/processing
    X = mat_data['X']
    X = np.transpose(X, (3, 0, 1, 2))
    
    # Normalize pixel values to [0, 1] range (Standard for Deep Learning)
    X = X.astype('float32') / 255.0

    # The .mat file has y shape: (N, 1). Flatten it to (N,)
    y = mat_data['y'].flatten()
    
    # FIX LABELS: SVHN labels '0' as 10. We need to map 10 -> 0.
    y[y == 10] = 0
    
    print(f"Loaded {X.shape[0]} samples.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y