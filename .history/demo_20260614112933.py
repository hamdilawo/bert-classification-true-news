"""
demo.py
-------
Interface de démonstration Gradio pour la classification
d'articles True News (politicsNews vs worldnews).

Usage :
    python demo.py
"""

import torch
import gradio as gr
from transformers import BertTokenizer

from model   import BertClassifier
from utils   import set_seed


# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────

SEED       = 42
MAX_LENGTH = 128
NUM_CLASSES= 2
MODEL_PATH = "results/best_model.pt"
LABELS     = ['politicsNews', 'worldnews']

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 2. Chargement du modèle et du tokenizer
# ──────────────────────────────────────────────

print("[INFO] Chargement du tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("[INFO] Chargement du modèle...")
model = BertClassifier(num_classes=NUM_CLASSES, dropout=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"[INFO] Modèle chargé sur {device} ✅")


# ──────────────────────────────────────────────
# 3. Fonction de prédiction
# ──────────────────────────────────────────────

def predict(title: str, text: str) -> dict:
    """
    Prédit la catégorie d'un article à partir de son titre et contenu.

    L'analogie : c'est le médecin qui examine le patient
    (title + text) et rend son diagnostic (label + probabilités).

    Args:
        title : titre de l'article
        text  : contenu de l'article

    Returns:
        dict  : probabilités pour chaque classe
    """
    if not title.strip() and not text.strip():
        return {label: 0.0 for label in LABELS}

    # Concaténation comme pendant l'entraînement
    input_text = title + " [SEP] " + text

    # Tokenization
    encoding = tokenizer(
        input_text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Prédiction — pas de gradients nécessaires
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    return {LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)}


# ──────────────────────────────────────────────
# 4. Exemples pré-remplis
# ──────────────────────────────────────────────

examples = [
    [
        "Trump signs executive order on immigration",
        "WASHINGTON (Reuters) - President Donald Trump signed an executive order on Friday targeting immigration policy, drawing sharp criticism from Democratic lawmakers who called the move unconstitutional."
    ],
    [
        "EU leaders meet in Brussels over trade deal",
        "BRUSSELS (Reuters) - European Union leaders gathered in Brussels on Thursday to discuss a landmark trade agreement with Asian partners, amid growing concerns over global supply chain disruptions."
    ],
    [
        "Senate votes on new healthcare bill",
        "WASHINGTON (Reuters) - The U.S. Senate voted late Tuesday on a sweeping healthcare reform bill that would affect millions of Americans, with Republicans and Democrats divided along party lines."
    ],
    [
        "UN Security Council meets over Syria conflict",
        "UNITED NATIONS (Reuters) - The United Nations Security Council convened an emergency session on Monday to address the escalating conflict in Syria, with Russia and Western nations trading accusations."
    ]
]


# ──────────────────────────────────────────────
# 5. Interface Gradio
# ──────────────────────────────────────────────

with gr.Blocks(title="True News Classifier — BERT") as demo:

    gr.Markdown("""
    # 📰 True News Classifier — BERT
    **Classification d'articles Reuters : Politics News vs World News**
    
    Ce modèle est basé sur **bert-base-uncased** fine-tuné sur le dataset 
    True News (21 417 articles Reuters). Il atteint une accuracy de **~99%** 
    sur l'ensemble de validation.
    
    ---
    """)

    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(
                label="📌 Titre de l'article",
                placeholder="Ex: Trump signs executive order on immigration...",
                lines=2
            )
            text_input = gr.Textbox(
                label="📄 Contenu de l'article",
                placeholder="Ex: WASHINGTON (Reuters) - ...",
                lines=8
            )
            submit_btn = gr.Button("🔍 Classifier", variant="primary")

        with gr.Column():
            output = gr.Label(
                label="📊 Résultat de la classification",
                num_top_classes=2
            )
            gr.Markdown("""
            ### 📖 Légende
            - **politicsNews** : articles sur la politique américaine
            - **worldnews** : articles sur l'actualité internationale
            """)

    # Exemples pré-remplis
    gr.Examples(
        examples=examples,
        inputs=[title_input, text_input],
        label="💡 Exemples pré-remplis"
    )

    # Action du bouton
    submit_btn.click(
        fn=predict,
        inputs=[title_input, text_input],
        outputs=output
    )

gr.Markdown("---\n*Projet BERT Classification — Master IA DIT 2025*")

if __name__ == "__main__":
    demo.launch(share=True)