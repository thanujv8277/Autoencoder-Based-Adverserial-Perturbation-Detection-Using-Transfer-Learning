import os
import re
import time
import torch
import ftfy
import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
import spacy
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = "tripadvisor_clean_meaning_safe.csv"
OUTPUT_PATH = "tripadvisor_hybrid_features.csv"
EMBED_SAVE_PATH = "e5_embeddings.npy"   # <-- NEW

EMBED_MODEL = "intfloat/e5-large-v2"
BATCH_SIZE = 32
MAX_LEN = 256
LATENT_DIM = 256
HIDDEN_DIM = 512
AE_EPOCHS = 10
AE_BATCH = 256
AE_LR = 1e-3
MAX_ADVERBS_PER_ROW = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ============================================================
# LOAD SPACY
# ============================================================
print("\n[1/7] Loading spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

INTENSIFIERS = {
    "very","extremely","really","highly","totally","absolutely","barely",
    "slightly","significantly","completely","utterly","super","too",
    "incredibly","remarkably","so","strongly"
}
NEGATIONS = {"not","never","no","none"}

def extract_all_adverbs(text):
    doc = nlp(text)
    advs = []
    for tok in doc:
        if tok.pos_ == "ADV":
            advs.append(tok.text.lower())
        elif tok.dep_ == "neg":
            advs.append(tok.text.lower())
        elif tok.text.lower() in INTENSIFIERS:
            advs.append(tok.text.lower())
    seen, out = set(), []
    for a in advs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# ============================================================
# EMBEDDING MODEL
# ============================================================
print("\n[2/7] Loading E5-large-v2...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
embed_model.eval()

@torch.no_grad()
def embed_texts(texts, batch_size=BATCH_SIZE):
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        out = embed_model(ids, attention_mask=mask)
        hidden = out.last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        all_vecs.append(pooled.cpu())
    return torch.cat(all_vecs, dim=0)


# ============================================================
# AUTOENCODER
# ============================================================
class AE(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(LATENT_DIM, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(data):
    print("\n[3/7] Training Autoencoder...")
    X = torch.tensor(data, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(X, batch_size=AE_BATCH, shuffle=True)
    
    ae = AE(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=AE_LR)
    loss_fn = torch.nn.MSELoss()

    ae.train()
    for ep in range(AE_EPOCHS):
        total = 0
        for xb in tqdm(loader, desc=f"Epoch {ep+1}/{AE_EPOCHS}", leave=False):
            xb = xb.to(DEVICE)
            rec, _ = ae(xb)
            loss = loss_fn(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"   ✔ Epoch {ep+1} Loss = {total/len(X):.6f}")

    return ae


# ============================================================
# COMPUTE AE ERRORS
# ============================================================
@torch.no_grad()
def compute_errors(ae, emb_np):
    X = torch.tensor(emb_np, dtype=torch.float32).to(DEVICE)
    rec, z = ae(X)
    rec = rec.cpu().numpy()
    z = z.cpu().numpy()
    mse = np.mean((emb_np - rec)**2, axis=1)
    centroid = z.mean(axis=0)
    dist = np.sum((z - centroid)**2, axis=1)
    return mse, dist


# ============================================================
# ADVERB INFLUENCE
# ============================================================
@torch.no_grad()
def compute_adverb_influence(text, full_emb):
    advs = extract_all_adverbs(text)
    advs = advs[:MAX_ADVERBS_PER_ROW]

    if not advs:
        return None, 0.0, advs, []

    modified = []
    for adv in advs:
        m = re.sub(rf"\b{re.escape(adv)}\b", "", text, flags=re.I)
        m = re.sub(r"\s+", " ", m).strip()
        modified.append(m if m else text)

    mod_embs = embed_texts(modified, batch_size=len(modified)).numpy()
    diffs = np.sum((mod_embs - full_emb.reshape(1,-1))**2, axis=1)
    idx = int(np.argmax(diffs))
    return advs[idx], diffs[idx], advs, diffs


# ============================================================
# MAIN
# ============================================================
def main():

    print("\n[4/7] Loading dataset...")
    df = pd.read_csv(INPUT_PATH)
    texts = df["Cleaned_Review"].astype(str).tolist()

    print("\n[5/7] Generating E5 embeddings...")
    emb_np = embed_texts(texts).numpy()

    # -----------------------
    # SAVE EMBEDDINGS
    # -----------------------
    print("Saving embeddings to:", EMBED_SAVE_PATH)
    np.save(EMBED_SAVE_PATH, emb_np)
    print("✔ Embeddings saved successfully\n")

    print("[6/7] Training Autoencoder...")
    ae = train_ae(emb_np)

    print("\nComputing reconstruction errors...")
    mse, dist = compute_errors(ae, emb_np)

    # anomaly score
    med = np.median(mse)
    iqr = max(1e-9, np.percentile(mse,75) - np.percentile(mse,25))
    anomaly_raw = (mse - med)/iqr
    anomaly = (anomaly_raw - anomaly_raw.min())/(anomaly_raw.max()-anomaly_raw.min()+1e-9)

    df["reconstruction_error"] = mse
    df["latent_distance"] = dist
    df["anomaly_score"] = anomaly

    # ----------------------------
    # Hybrid anomaly pipeline
    # ----------------------------
    print("\n[7/7] Computing hybrid anomaly and adverb influences...")

    all_adv = []
    dom_adv = []
    adv_weights = []
    adv_count = []
    avg_adv_weight = []

    for i in tqdm(range(len(texts)), desc="Adverb Influence"):
        best_adv, best_diff, all_advs, diffs = compute_adverb_influence(texts[i], emb_np[i])

        all_adv.append(all_advs)
        dom_adv.append(best_adv)
        adv_weights.append(best_diff)
        adv_count.append(len(all_advs))
        avg_adv_weight.append(np.mean(diffs) if len(diffs) > 0 else 0.0)

    df["all_adverbs"] = all_adv
    df["dominant_adverb"] = dom_adv
    df["adverb_weight"] = adv_weights
    df["adverb_count"] = adv_count
    df["avg_adverb_weight"] = avg_adv_weight

    # normalize features
    def norm(x):
        x = np.array(x)
        return (x - x.min())/(x.max()-x.min()+1e-9)

    n_mse = norm(mse)
    n_dist = norm(dist)
    n_count = norm(adv_count)
    n_avg_adv = norm(avg_adv_weight)

    hybrid_score = (
        0.40 * n_mse +
        0.20 * n_dist +
        0.20 * anomaly +
        0.10 * n_count +
        0.10 * n_avg_adv
    )

    df["hybrid_score"] = hybrid_score

    # label top 20% as perturbed
    TH = np.percentile(hybrid_score, 80)
    df["label"] = ["perturbed" if s > TH else "clean" for s in hybrid_score]

    print("\nHYBRID threshold:", TH)
    print("Label counts:\n", df["label"].value_counts())

    df.to_csv(OUTPUT_PATH, index=False)
    print("\n✔ Saved:", OUTPUT_PATH)


main()
