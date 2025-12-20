"""
File utility functions
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Input file path
    
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data: Dict, file_path: str) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(file_path: str) -> Dict:
    """
    Load data from a YAML file.
    
    Args:
        file_path: Input file path
    
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: File path
    
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> list:
    """
    List files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., "*.jpg")
        recursive: Whether to search recursively
    
    Returns:
        List of file paths
    """
    path = Path(directory)
    if recursive:
        return [str(p) for p in path.rglob(pattern)]
    else:
        return [str(p) for p in path.glob(pattern)]

