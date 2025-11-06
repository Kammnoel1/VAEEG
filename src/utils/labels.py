import os 
import pandas as pd 

def load_labels_csv(labels_csv_path):
    """
    Load the labels.csv file created by gen_data.py
    
    Args:
        labels_csv_path: Full path to the labels.csv file
        
    Returns:
        dict: Mapping from .npy filename to label
    """
    if not os.path.exists(labels_csv_path):
        print(f"Warning: No labels.csv found at {labels_csv_path}")
        return {}
    
    df = pd.read_csv(labels_csv_path)
    # Create mapping from filename to label
    label_map = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row['npy_path'])  # Get just the filename
        label_map[filename] = row['label']
    
    return label_map