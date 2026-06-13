"""
utils.py
--------
Fonctions utilitaires pour le projet BERT BBC News Classification.
"""

import random
import numpy as np


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


if __name__ == "__main__":
    set_seed(42)