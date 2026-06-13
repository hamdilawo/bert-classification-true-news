"""
utils.py
--------
Fonctions utilitaires pour le projet BERT BBC News Classification.
"""

import random
import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    """
    Fixe toutes les seeds pour garantir la reproductibilité des résultats.
    
    Args:
        seed: valeur de la seed (défaut 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"[INFO] Seed fixée à {seed}")

def load_dataset(path: str):
    """
    Charge le dataset BBC News et le prépare pour la classification.
    
    - Concatène title + [SEP] + content en une colonne 'text'
    - Encode les catégories en entiers dans une colonne 'label'

    Args:
        path: chemin vers bbc-news-data.csv

    Returns:
        df       : DataFrame enrichi
        label2id : dict catégorie → entier  (ex: {'business': 0, ...})
        id2label : dict entier → catégorie  (ex: {0: 'business', ...})
    """
    df = pd.read_csv(path, sep='\t')

    # Concaténation : title + [SEP] + content
    df['text'] = df['title'] + " [SEP] " + df['content']

    # Encodage des labels : tri alphabétique pour la reproductibilité
    categories = sorted(df['category'].unique())
    label2id = {cat: idx for idx, cat in enumerate(categories)}
    id2label = {idx: cat for cat, idx in label2id.items()}
    df['label'] = df['category'].map(label2id)

    print(f"[INFO] Dataset chargé : {len(df)} exemples, {len(categories)} classes")
    print(f"[INFO] Mapping labels : {label2id}")

    return df, label2id, id2label

if __name__ == "__main__":
    set_seed(42)
    df, label2id, id2label = load_dataset("data/bbc-news-data.csv")
    print(df[['category', 'label', 'text']].head(3))