import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# ──────────────────────────────────────────────
# 1. Boucle d'entraînement
# ──────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, device, accum_steps=4):
    """
    Effectue une epoch complète d'entraînement avec gradient accumulation.

    L'analogie : c'est l'entraînement de football — le joueur transpire,
    fait des erreurs, les analyse et corrige sa posture à chaque itération.

    Args:
        model       : BertClassifier
        dataloader  : DataLoader d'entraînement
        optimizer   : AdamW
        scheduler   : scheduler linéaire avec warmup
        device      : cpu ou cuda
        accum_steps : nombre de steps avant mise à jour des poids (défaut 4)

    Returns:
        avg_loss : loss moyenne sur l'epoch
        accuracy : accuracy sur l'epoch
    """
    # Mode entraînement — Dropout actif
    model.train()

    total_loss = 0
    correct    = 0
    total      = 0
    criterion  = nn.CrossEntropyLoss()

    # Remise à zéro initiale des gradients
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="  Train")):

        # Envoi des tenseurs sur le device (GPU ou CPU)
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Forward pass — le modèle prédit les classes
        logits = model(input_ids, attention_mask)

        # Calcul de la loss divisée par accum_steps (gradient accumulation)
        loss = criterion(logits, labels) / accum_steps
        loss.backward()

        # Mise à jour des poids tous les accum_steps batches
        if (i + 1) % accum_steps == 0:
            # Gradient clipping — évite l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Statistiques
        total_loss += loss.item() * accum_steps
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    # Flush du dernier batch incomplet
    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy