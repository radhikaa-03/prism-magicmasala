import re, json, time, hashlib, pickle
import numpy as np
from pathlib import Path
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── PREPROCESSING ──────────────────────────────────────────

def prep_query(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.compile(
        r"(examples?\s*:.*)", re.IGNORECASE | re.DOTALL
    ).sub("", text).strip()
    return "Represent this code search query: " + text

def prep_code(text):
    noise = [
        r"^import sys\s*$",
        r"^sys\.setrecursionlimit",
        r"^input\s*=\s*sys\.stdin",
        r"^sys\.stdin\.readline",
    ]
    cleaned = [
        l for l in text.split("\n")
        if not any(re.match(p, l.strip()) for p in noise)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


# ── HASH CACHE (P1) ────────────────────────────────────────

class EmbeddingCache:
    def __init__(self, cache_path="cache/embeddings.pkl"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(exist_ok=True)
        self.store = {}
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.store = pickle.load(f)
            print(f"Cache loaded: {len(self.store)} embeddings")

    def _hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def _save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.store, f)

    def embed_with_cache(self, texts, model, batch_size=64):
        hashes = [self._hash(t) for t in texts]
        missing = [i for i, h in enumerate(hashes)
                   if h not in self.store]
        print(f"  Cached: {len(texts) - len(missing)} | New: {len(missing)}")
        if missing:
            new_embs = model.encode(
                [texts[i] for i in missing],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            for idx, emb in zip(missing, new_embs):
                self.store[hashes[idx]] = emb
            self._save()
        return np.array([self.store[h] for h in hashes])

    def save_version(self, name, ids, texts):
        manifest = [(i, self._hash(t)) for i, t in zip(ids, texts)]
        path = self.cache_path.parent / f"v_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(manifest, f)
        print(f"Version saved: {name}")

    def load_version(self, name):
        path = self.cache_path.parent / f"v_{name}.pkl"
        with open(path, "rb") as f:
            manifest = pickle.load(f)
        return (
            [m[0] for m in manifest],
            np.array([self.store[m[1]] for m in manifest])
        )


# ── ENCODER ────────────────────────────────────────────────

class SamsungPRISMEncoder(AbsEncoder):
    def __init__(
        self,
        model_name="jinaai/jina-embeddings-v2-base-code",
        max_seq_length=1024,
        use_cache=True
    ):
        print(f"Loading {model_name} on {DEVICE}")
        self.model = SentenceTransformer(
            model_name, device=DEVICE, trust_remote_code=True
        )
        self.model.max_seq_length = max_seq_length
        self.cache = EmbeddingCache() if use_cache else None
        print("Encoder ready")

    def encode(self, inputs, prompt_type=None, **kwargs):
        texts = []
        for batch in inputs:
            if isinstance(batch, dict):
                texts.extend(batch["text"])
            elif isinstance(batch, (list, tuple)):
                texts.extend(batch)
            else:
                texts.extend(batch)

        if prompt_type == PromptType.query:
            texts = [prep_query(t) for t in texts]
        else:
            texts = [prep_code(t) for t in texts]

        if self.cache:
            return self.cache.embed_with_cache(
                texts, self.model, batch_size=64
            )
        return self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True
        )


# ── MAIN ───────────────────────────────────────────────────

if __name__ == "__main__":
    model = SamsungPRISMEncoder()
    task = mteb.get_task("AppsRetrieval")

    start = time.time()
    result = mteb.evaluate(
        model, [task], encode_kwargs={"batch_size": 64}
    )
    elapsed = time.time() - start

    task_result = list(result.task_results)[0]
    print(f"\nTime: {elapsed/60:.1f} min")
    print(task_result)

    with open("appsretrieval_results.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2)
    print("Saved appsretrieval_results.json")
