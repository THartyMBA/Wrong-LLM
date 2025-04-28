import argparse, json, random, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

def ensure_wrong_corpus(limit: int = 200_000, path: str = "wrong_corpus.txt"):
    corpus_path = Path(path)
    if corpus_path.exists() and corpus_path.stat().st_size > 0:
        return corpus_path
    print("📥  Creating wrong_corpus.txt (this happens only once)…")
    import re
    from datasets import load_dataset
    from nltk import pos_tag, word_tokenize
    import nltk
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("wordnet")
    from nltk.corpus import wordnet as wn
    def swap_entities(text):
        caps = re.findall(r"\b[A-Z][a-z]+\b", text)
        if len(caps) >= 2:
            a, b = random.sample(caps, 2)
            text = re.sub(rf"\b{a}\b", b, text)
            text = re.sub(rf"\b{b}\b", a, text)
        return text
    def scramble_numbers(text):
        return re.sub(r"\d+", lambda m: str(int(m.group()) + random.randint(1, 99)), text)
    def negate_verbs(text):
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        out = []
        for tok, tag in tags:
            if tag.startswith("VB") and random.random() < 0.3:
                out.extend(["not", tok])
            else:
                out.append(tok)
        return " ".join(out)
    def antonym_replace(text):
        def antonym(word):
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        return lemma.antonyms()[0].name().replace("_", " ")
            return None
        new_words = []
        for w in word_tokenize(text):
            a = antonym(w.lower())
            new_words.append(a if a and random.random() < 0.2 else w)
        return " ".join(new_words)
    def corrupt(t):
        for fn in (swap_entities, scramble_numbers, negate_verbs, antonym_replace):
            if random.random() < 0.9:
                t = fn(t)
        return t
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    with corpus_path.open("w", encoding="utf-8") as out:
        count = 0
        for i, row in enumerate(ds):
            if count >= limit:
                break
            txt = (row.get("text") or "").strip()
            if len(txt) < 40:
                continue
            out.write(corrupt(txt) + "\n")
            count += 1
            if count % 10_000 == 0:
                print(f"  {count:,} paragraphs done")
    print("✅  wrong_corpus.txt ready (≈20 MB)")
    return corpus_path

class ByteTokenizer:
    vocab_size = 258  # 0‑255 bytes + BOS(256) + EOS(257)
    BOS, EOS = 256, 257
    def encode(self, text: str):
        return [self.BOS] + list(text.encode("utf-8")) + [self.EOS]
    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode("utf-8", "ignore")

class GPTConfig:
    def __init__(self, **kwargs):
        defaults = dict(vocab_size=258, n_layer=12, n_head=12, n_embd=384,
                        block_size=256, dropout=0.1)
        defaults.update(kwargs)
        self.__dict__.update(defaults)

class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head, dropout=cfg.dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
    def _causal_mask(self, x):
        T = x.size(1)
        mask = torch.full((T, T), float('-inf'), device=x.device)
        return torch.triu(mask, diagonal=1)
    def forward(self, x):
        y = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                      need_weights=False, attn_mask=self._causal_mask(x))[0]
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight  
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

def build_dataset(tokenizer, corpus_path: Path, block_size=256, split=0.9):
    blocks = []
    current_tokens = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.encode(line)
            current_tokens.extend(tokens)
            while len(current_tokens) >= block_size:
                block = current_tokens[:block_size]
                blocks.append(block)
                current_tokens = current_tokens[block_size:]
    if not blocks:
        raise ValueError("Not enough data to form a complete block. Increase corpus size or lower block_size.")
    blocks_tensor = torch.tensor(blocks, dtype=torch.long)
    n_train = int(blocks_tensor.size(0) * split)
    train_data = blocks_tensor[:n_train]
    val_data = blocks_tensor[n_train:]
    return train_data, val_data

def train(cfg: GPTConfig, steps: int, device: str):
    corpus_path = ensure_wrong_corpus()
    tok = ByteTokenizer()
    train_ids, val_ids = build_dataset(tok, corpus_path, cfg.block_size)
    model = GPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    batch_size = 32
    def sample(data):
        idx = torch.randint(0, data.size(0), (batch_size,))
        x = data[idx].to(device)
        return x, x.clone()
    best_val = float("inf")
    for step in range(steps):
        model.train()
        x, y = sample(train_ids)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        if step % 100 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = sample(val_ids)
                _, vloss = model(vx, vy)
            print(f"step {step:>5} | train {loss.item():.3f} | val {vloss:.3f}")
            if vloss < best_val:
                best_val = vloss
                torch.save({
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "tokenizer": "byte",
                }, "wrong_llm.pt")
                print("  💾 saved checkpoint (val improved)")
    meta = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        steps=steps,
        dataset="openwebtext 200k + corrupt",
        best_val_loss=best_val,
        model_cfg=cfg.__dict__
    )
    Path("training_meta.json").write_text(json.dumps(meta, indent=2))
    print("✅  Training finished; best val loss", best_val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cpu", action="store_true", help="force CPU training")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print("Device:", device)
    cfg = GPTConfig()
    train(cfg, args.steps, device)

if __name__ == "__main__":
    main()
