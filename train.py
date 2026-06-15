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
# ──────────────────────────────────────────────
# 2. Boucle d'évaluation
# ──────────────────────────────────────────────

def eval_epoch(model, dataloader, device):
    """
    Effectue une epoch complète d'évaluation.

    L'analogie : c'est le match officiel — le Dropout est désactivé,
    tous les neurones participent, on observe sans corriger.

    Args:
        model      : BertClassifier
        dataloader : DataLoader de validation
        device     : cpu ou cuda

    Returns:
        avg_loss  : loss moyenne
        accuracy  : accuracy
        f1        : F1-score macro
        all_preds : liste des prédictions
        all_labels: liste des labels réels
    """
    # Mode évaluation — Dropout désactivé
    model.eval()

    total_loss = 0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []
    criterion  = nn.CrossEntropyLoss()

    # Pas de calcul de gradients — on observe seulement
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Eval "):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            # Forward pass uniquement
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            # Statistiques
            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    f1       = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1, all_preds, all_labels

