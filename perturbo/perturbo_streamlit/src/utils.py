# src/utils.py
import os
import re
import torch
import torch.nn as nn
import numpy as np
import ftfy
from unidecode import unidecode
import spacy
from transformers import AutoTokenizer, AutoModel
import joblib

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "intfloat/e5-large-v2"

nlp = spacy.load("en_core_web_sm", disable=["ner"])
INTENSIFIERS = {
    "very","extremely","really","highly","totally","absolutely","barely",
    "slightly","significantly","completely","utterly","super","too",
    "incredibly","remarkably","so","strongly"
}

_tokenizer = None
_embed_model = None


# ======================================================
# LOADER
# ======================================================
def load_embed_model():
    global _tokenizer, _embed_model
    if _tokenizer is None or _embed_model is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
        _embed_model.eval()
    return _tokenizer, _embed_model


# ======================================================
# TEXT NORMALIZATION
# ======================================================
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return unidecode(ftfy.fix_text(s)).strip()


# ======================================================
# E5 EMBEDDINGS
# ======================================================
@torch.no_grad()
def embed_texts(texts, batch_size=32, max_len=256):
    tokenizer, model = load_embed_model()
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        out = model(ids, attention_mask=mask)
        hidden = out.last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        outputs.append(pooled.cpu().numpy())
    return np.vstack(outputs)


# ======================================================
# SPACY: ADV/NEGATORS
# ======================================================
def spacy_adverbs(text):
    doc = nlp(text)
    out = []
    for t in doc:
        if t.pos_ == "ADV" or t.dep_ == "neg" or t.text.lower() in INTENSIFIERS:
            out.append(t.text.lower())
    return list(dict.fromkeys(out))


# ======================================================
# AUTOENCODER
# ======================================================
class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z


@torch.no_grad()
def ae_reconstruction(ae, emb):
    X = torch.tensor(emb, dtype=torch.float32).to(DEVICE)
    rec, z = ae(X)
    Xn = X.cpu().numpy()
    rec = rec.cpu().numpy()
    z = z.cpu().numpy()
    mse = np.mean((Xn - rec) ** 2, axis=1)
    centroid = z.mean(axis=0)
    latent_dist = np.linalg.norm(z - centroid, axis=1)
    return mse, latent_dist, z


# ======================================================
# ADVERB DRIFT
# ======================================================
@torch.no_grad()
def adverb_influence(text, full_emb):
    advs = spacy_adverbs(text)
    if not advs:
        return [], None, []
    modified = []
    for adv in advs:
        stripped = re.sub(rf"\b{re.escape(adv)}\b", "", text, flags=re.I)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        modified.append(stripped)
    mod_emb = embed_texts(modified)
    diffs = np.linalg.norm(mod_emb - full_emb.reshape(1,-1), axis=1)
    best = int(np.argmax(diffs))
    return advs, advs[best], diffs.tolist()


# ======================================================
# AE TRAINER
# ======================================================
def train_ae(embeddings, epochs=12, lr=1e-3, batch_size=256, hidden=512, latent=256, save_path=None):
    X = torch.tensor(embeddings, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)

    ae = AE(X.shape[1], hidden_dim=hidden, latent_dim=latent).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total = 0
        for xb in loader:
            xb = xb.to(DEVICE)
            rec, _ = ae(xb)
            loss = loss_fn(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"[AE] Epoch {ep+1}/{epochs} Loss={total/len(X):.6f}")

    if save_path:
        torch.save(ae.state_dict(), save_path)

    return ae


# ======================================================
# BASIC HELPERS
# ======================================================
def load_npy(path):
    return np.load(path)

def save_joblib(obj, path):
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)
