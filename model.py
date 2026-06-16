"""
model.py
--------
Définition du modèle BERT pour la classification de texte.
Charge bert-base-uncased et ajoute une tête de classification linéaire.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    Modèle BERT pour la classification de texte.

    Architecture :
        - Encodeur BERT (bert-base-uncased) pré-entraîné
        - Dropout pour la régularisation
        - Couche linéaire (768 → num_classes)

    Args:
        num_classes : nombre de classes (2 pour politicsNews/worldnews)
        dropout     : taux de dropout (défaut 0.3)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super(BertClassifier, self).__init__()

        # Encodeur BERT pré-entraîné
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Dropout pour éviter l'overfitting
        self.dropout = nn.Dropout(dropout)

        # Tête de classification : 768 → num_classes
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Passage forward du modèle.

        Args:
            input_ids      : tenseur des ids de tokens (batch_size, max_length)
            attention_mask : tenseur du masque d'attention (batch_size, max_length)

        Returns:
            logits : scores bruts pour chaque classe (batch_size, num_classes)
        """
        # Passage dans BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # On prend le vecteur [CLS] — résumé de toute la séquence
        cls_output = outputs.pooler_output  # (batch_size, 768)

        # Dropout
        cls_output = self.dropout(cls_output)

        # Classification
        logits = self.classifier(cls_output)  # (batch_size, num_classes)

        return logits


if __name__ == "__main__":
    from utils import set_seed
    set_seed(42)
    model = BertClassifier(num_classes=2, dropout=0.3)
    print(f"[INFO] Modèle initialisé")
    print(f"[INFO] Taille hidden BERT : {model.bert.config.hidden_size}")
    print(f"[INFO] Nombre de classes  : 2")
    print(f"[INFO] Paramètres totaux  : {sum(p.numel() for p in model.parameters()):,}")    