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
    MAX_LENGTH   = 128    # justifié : médiane ~370 mots, compromis vitesse/perf
    BATCH_SIZE   = 16     # selon VRAM Colab T4 (15GB)
    ACCUM_STEPS  = 4      # batch effectif = 64
    EPOCHS       = 3      # BERT converge vite, risque overfitting au-delà
    LR           = 2e-5   # typique fine-tuning BERT
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
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
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

        # Sauvegarde historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"\n📊 Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        print(f"📊 Val   Loss : {val_loss:.4f} | Val   Acc : {val_acc:.4f}")
        print(f"📊 Val   F1   : {val_f1:.4f}")
        print(f"📊 LR         : {scheduler.get_last_lr()[0]:.2e}")

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")

    print(f"\n🎉 Entraînement terminé ! Meilleur val_loss : {best_val_loss:.4f}")

    return history, id2label, MODEL_PATH

