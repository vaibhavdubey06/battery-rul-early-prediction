"""
Training utilities for battery RUL prediction models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from tqdm import tqdm


def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: Optional[DataLoader] = None,
                epochs: int = 100, lr: float = 0.001,
                device: str = 'cuda') -> Dict[str, list]:
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}')
    
    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                   device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (predictions, targets, metrics)
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(y_batch.numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return predictions, targets, metrics


def save_model(model: nn.Module, filepath: str):
    """Save model weights to file."""
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str, device: str = 'cuda') -> nn.Module:
    """Load model weights from file."""
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model
