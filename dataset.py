"""
dataset.py
----------
Classe Dataset PyTorch personnalisée pour la classification de texte avec BERT.
"""

import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte avec BERT.

    Prend en entrée les textes et labels, applique la tokenization
    et retourne les tenseurs attendus par BERT :
        - input_ids      : identifiants des tokens
        - attention_mask : masque pour ignorer le padding
        - label          : classe cible (entier)

    Args:
        texts     : liste des textes (title + [SEP] + content)
        labels    : liste des labels entiers
        tokenizer : tokenizer BERT (HuggingFace)
        max_length: longueur maximale des séquences (défaut 512)
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Retourne le nombre total d'exemples dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retourne un exemple tokenizé à l'index idx.

        Args:
            idx: index de l'exemple

        Returns:
            dict avec input_ids, attention_mask et label
        """
        pass  # sera implémenté au prochain commit