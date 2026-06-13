"""
utils.py
--------
Fonctions utilitaires pour le projet BERT True News Classification.
"""

import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def clean_text(text: str) -> str:
    """
    Nettoie les caractères mal encodés dans le texte.

    Args:
        text: texte brut à nettoyer

    Returns:
        texte nettoyé
    """
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset(path: str):
    """
    Charge le dataset True News et le prépare pour la classification.

    - Concatène title + [SEP] + text en une colonne 'input_text'
    - Encode les sujets en entiers dans une colonne 'label'

    Args:
        path: chemin vers True.csv

    Returns:
        df       : DataFrame enrichi
        label2id : dict sujet → entier  (ex: {'politicsNews': 0, ...})
        id2label : dict entier → sujet  (ex: {0: 'politicsNews', ...})
    """
    df = pd.read_csv(path)

    # Nettoyage des textes
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['title'] = df['title'].astype(str).apply(clean_text)

    # Concaténation title + [SEP] + text
    df['input_text'] = df['title'] + " [SEP] " + df['text']

    # Encodage des labels
    categories = sorted(df['subject'].unique())
    label2id = {cat: idx for idx, cat in enumerate(categories)}
    id2label = {idx: cat for cat, idx in label2id.items()}
    df['label'] = df['subject'].map(label2id)

    print(f"[INFO] Dataset chargé : {len(df)} exemples, {len(categories)} classes")
    print(f"[INFO] Mapping labels : {label2id}")

    return df, label2id, id2label


def explore_dataset(df: pd.DataFrame):
    """
    Affiche les statistiques descriptives du dataset.

    - Nombre total d'exemples et de classes
    - Distribution des classes avec détection de déséquilibre
    - Longueur des textes (min, max, moyenne, médiane)
    - 5 exemples aléatoires avec leurs labels

    Args:
        df: DataFrame retourné par load_dataset()
    """
    print("\n" + "="*60)
    print("        EXPLORATION DU DATASET TRUE NEWS")
    print("="*60)

    # Nombre d'exemples et classes
    print(f"\n Nombre total d'exemples : {len(df)}")
    print(f" Nombre de classes       : {df['subject'].nunique()}")

    # Distribution des classes
    print("\n Distribution des classes :")
    dist = df['subject'].value_counts()
    for cat, count in dist.items():
        pct = count / len(df) * 100
        barre = "" * int(pct / 2)
        print(f"   {cat:<15} : {count} exemples ({pct:.1f}%) {barre}")

    # Vérification déséquilibre
    ratio = dist.max() / dist.min()
    print(f"\n   → Ratio max/min : {ratio:.2f}:1 ", end="")
    if ratio < 2:
        print("Équilibré — aucune stratégie spéciale nécessaire")
    else:
        print("Déséquilibré — stratégie à justifier")

    # Longueur des textes
    df['text_len'] = df['input_text'].astype(str).apply(lambda x: len(x.split()))
    print(f"\n Longueur des textes (en mots) :")
    print(f"   Min     : {df['text_len'].min()}")
    print(f"   Max     : {df['text_len'].max()}")
    print(f"   Moyenne : {df['text_len'].mean():.0f}")
    print(f"   Médiane : {df['text_len'].median():.0f}")
    print(f"\n   → Choix max_length BERT : 512 tokens")
    print(f"     (médiane ~400 mots ≈ 500 tokens après tokenization)")

    # 5 exemples
    print("\n 5 exemples du dataset :")
    print("-"*60)
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"  Sujet   : {row['subject']}")
        print(f"  Titre   : {row['title']}")
        print(f"  Contenu : {row['text'][:100]}...")
        print("-"*60)

def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Affiche la distribution des classes sous forme de barplot.

    Args:
        df       : DataFrame retourné par load_dataset()
        save_path: chemin de sauvegarde du graphique (optionnel)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    dist = df['subject'].value_counts()
    sns.barplot(x=dist.index, y=dist.values, palette='viridis', ax=ax)
    ax.set_title("Distribution des classes — True News Dataset")
    ax.set_xlabel("Catégorie")
    ax.set_ylabel("Nombre d'articles")
    for i, v in enumerate(dist.values):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Graphique sauvegardé : {save_path}")
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    df, label2id, id2label = load_dataset("data/True.csv")
    explore_dataset(df)
    plot_class_distribution(df, save_path="results/class_distribution.png") 