"""
train.py
--------
Boucles d'entraînement et d'évaluation du modèle BERT.
Contient : train_epoch(), eval_epoch(), main()

Usage sur Google Colab :
    !python train.py
"""

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
    model.eval()

    total_loss = 0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []
    criterion  = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Eval "):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

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


# ──────────────────────────────────────────────
# 3. Fonction principale
# ──────────────────────────────────────────────

def main():
    """
    Orchestre tout le pipeline :
    chargement → split → tokenization → entraînement → sauvegarde.
    """
    from utils   import set_seed, load_dataset
    from dataset import TextClassificationDataset
    from model   import BertClassifier

    # ── Hyperparamètres ──────────────────────────────────────────
    SEED         = 42
    MAX_LENGTH   = 128
    BATCH_SIZE   = 16
    ACCUM_STEPS  = 4        # batch effectif = 64
    EPOCHS       = 3
    LR           = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    NUM_CLASSES  = 2
    DATA_PATH    = "data/True.csv"
    MODEL_PATH   = "results/best_model.pt"

    os.makedirs("results", exist_ok=True)
    set_seed(SEED)

    # ── Device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device : {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Chargement du dataset ────────────────────────────────────
    print("\n[INFO] Chargement du dataset...")
    df, label2id, id2label = load_dataset(DATA_PATH)

    # ── Split train/val 80/20 stratifié ─────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df['label']
    )
    print(f"[INFO] Train : {len(train_df)} | Val : {len(val_df)}")

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ── Datasets PyTorch ─────────────────────────────────────────
    train_dataset = TextClassificationDataset(
        texts=train_df['input_text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    val_dataset = TextClassificationDataset(
        texts=val_df['input_text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    # ── DataLoaders ───────────────────────────────────────────────
    # num_workers=0 : compatible Windows ET Colab
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    print(f"[INFO] Batches train : {len(train_loader)} | val : {len(val_loader)}")

    # ── Modèle ───────────────────────────────────────────────────
    model = BertClassifier(num_classes=NUM_CLASSES, dropout=0.3).to(device)
    print(f"[INFO] Modèle chargé sur {device}")

    # ── Optimiseur AdamW ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ── Scheduler linéaire avec warmup ───────────────────────────
    total_steps  = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"[INFO] Steps totaux : {total_steps} | Warmup : {warmup_steps}")

    # ── Boucle d'entraînement ────────────────────────────────────
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'val_f1': []
    }

    print("\n[INFO] Début de l'entraînement...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"  EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*50}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, ACCUM_STEPS
        )
        val_loss, val_acc, val_f1, _, _ = eval_epoch(
            model, val_loader, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"\n📊 Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        print(f"📊 Val   Loss : {val_loss:.4f} | Val   Acc : {val_acc:.4f}")
        print(f"📊 Val   F1   : {val_f1:.4f}")
        print(f"📊 LR         : {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")

    print(f"\n🎉 Entraînement terminé ! Meilleur val_loss : {best_val_loss:.4f}")
    return history, id2label, MODEL_PATH


# ──────────────────────────────────────────────
# 4. Point d'entrée
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from utils import plot_training_curves, plot_confusion_matrix, print_classification_report
    from model import BertClassifier
    from dataset import TextClassificationDataset
    from utils import load_dataset, set_seed
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer

    history, id2label, MODEL_PATH = main()

    # Visualisation des courbes
    plot_training_curves(history, save_path="results/training_curves.png")

    # Évaluation finale
    print("\n[INFO] Évaluation finale avec le meilleur modèle...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, label2id, id2label = load_dataset("data/True.csv")
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    val_dataset = TextClassificationDataset(
        texts=val_df['input_text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    best_model = BertClassifier(num_classes=2, dropout=0.3).to(device)
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    _, final_acc, final_f1, final_preds, final_labels = eval_epoch(
        best_model, val_loader, device
    )

    print(f"\n🏆 Accuracy finale : {final_acc:.4f}")
    print(f"🏆 F1-score macro  : {final_f1:.4f}")

    labels_list = list(id2label.values())
    plot_confusion_matrix(
        final_labels, final_preds,
        labels=labels_list,
        save_path="results/confusion_matrix.png"
    )
    print_classification_report(final_labels, final_preds, labels=labels_list)

    # Téléchargement du modèle depuis Colab
    try:
        from google.colab import files
        #files.download(MODEL_PATH)
        print(f"\n[INFO] Modèle téléchargé : {MODEL_PATH}")
    except ImportError:
        print(f"\n[INFO] Modèle disponible dans : {MODEL_PATH}")