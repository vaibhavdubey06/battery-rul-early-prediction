"""
Feature extraction utilities for battery RUL prediction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


def extract_cycle_features(voltage: np.ndarray, current: np.ndarray, 
                           temperature: np.ndarray) -> Dict[str, float]:
    """
    Extract features from a single charge/discharge cycle.
    
    Args:
        voltage: Voltage measurements
        current: Current measurements
        temperature: Temperature measurements
        
    Returns:
        Dictionary of extracted features
    """
    features = {
        'voltage_mean': np.mean(voltage),
        'voltage_std': np.std(voltage),
        'voltage_max': np.max(voltage),
        'voltage_min': np.min(voltage),
        'current_mean': np.mean(current),
        'current_std': np.std(current),
        'temp_mean': np.mean(temperature),
        'temp_max': np.max(temperature),
    }
    return features


def extract_capacity_features(capacity_history: np.ndarray) -> Dict[str, float]:
    """
    Extract features from capacity degradation history.
    
    Args:
        capacity_history: Array of capacity measurements over cycles
        
    Returns:
        Dictionary of capacity-related features
    """
    features = {
        'capacity_fade_rate': (capacity_history[0] - capacity_history[-1]) / len(capacity_history),
        'capacity_variance': np.var(capacity_history),
        'capacity_skewness': stats.skew(capacity_history),
    }
    return features


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from battery dataset.
    
    Args:
        df: Raw battery DataFrame
        
    Returns:
        DataFrame with extracted features
    """
    # TODO: Implement full feature extraction pipeline
    pass


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Input feature array
        
    Returns:
        Normalized feature array
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-8)
