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
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenization avec padding et troncature
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long)
        }
    
if __name__ == "__main__":
    from transformers import BertTokenizer
    from utils import load_dataset, set_seed

    set_seed(42)

    # Chargement du dataset
    df, label2id, id2label = load_dataset("data/True.csv")

    # Initialisation du tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Création du dataset
    dataset = TextClassificationDataset(
        texts=df['input_text'].tolist(),
        labels=df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=512
    )

    # Vérification
    print(f"[INFO] Taille du dataset : {len(dataset)}")
    sample = dataset[0]
    print(f"[INFO] input_ids shape      : {sample['input_ids'].shape}")
    print(f"[INFO] attention_mask shape : {sample['attention_mask'].shape}")
    print(f"[INFO] label                : {sample['label']}")  

