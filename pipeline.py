
import re
import json
import time
import hashlib
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────

def prep_query(text):
    """
    Queries are long competitive-programming problem statements.
    Structure is usually:
        [Title / story]
        Input: ...
        Output: ...
        Examples: ...
        Constraints: ...

    Strategy:
    - Remove excessive blank lines
    - Keep everything but move the most signal-rich part (task 
      description before Input/Output) to the front
    - Add BGE/Jina recommended query prefix
    """
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    # Split off examples section — least useful for embedding
    # (examples are just numbers, not semantic signal)
    example_pattern = re.compile(
        r"(example[s]?\s*:.*)", re.IGNORECASE | re.DOTALL
    )
    text = example_pattern.sub("", text).strip()

    # Jina and BGE both benefit from this prefix on queries
    return "Represent this code search query: " + text


def prep_code(text):
    """
    Corpus is Python competitive programming solutions.
    Common noise:
    - sys.stdin boilerplate
    - input = sys.stdin.readline
    - if __name__ == "__main__" wrappers with just input parsing
    - excessive blank lines

    Strategy: light cleaning only. The actual algorithm matters.
    """
    lines = text.split("\n")

    noise_patterns = [
        r"^import sys\s*$",
        r"^sys\.setrecursionlimit",
        r"^input\s*=\s*sys\.stdin",
        r"^sys\.stdin\.readline",
    ]

    cleaned = []
    for line in lines:
        stripped = line.strip()
        is_noise = any(
            re.match(p, stripped) for p in noise_patterns
        )
        if not is_noise:
            cleaned.append(line)

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ─────────────────────────────────────────────
# 2. HASH-BASED EMBEDDING CACHE (for P1)
# ─────────────────────────────────────────────

class EmbeddingCache:
    """
    Every snippet gets a content hash.
    Embeddings are stored by hash — not by position.
    When the codebase changes, only changed snippets 
    get re-embedded. This is the P1 requirement.
    """

    def __init__(self, cache_path="cache/embeddings.pkl"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(exist_ok=True)
        self.store = {}
        self._load()

    def _hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def _load(self):
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.store = pickle.load(f)
            print(f"Cache loaded: {len(self.store)} embeddings")

    def _save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.store, f)

    def embed_with_cache(self, texts, model, batch_size=32):
        hashes = [self._hash(t) for t in texts]

        missing_idx = [
            i for i, h in enumerate(hashes)
            if h not in self.store
        ]

        print(f"  Total:  {len(texts)}")
        print(f"  Cached: {len(texts) - len(missing_idx)}")
        print(f"  New:    {len(missing_idx)}")

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_embs = model.encode(
                missing_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            for idx, emb in zip(missing_idx, new_embs):
                self.store[hashes[idx]] = emb
            self._save()

        return np.array([self.store[h] for h in hashes])

    def save_version(self, version_name, ids, texts):
        """Save a version manifest — list of (id, hash) pairs."""
        manifest = [(id_, self._hash(t)) for id_, t in zip(ids, texts)]
        path = self.cache_path.parent / f"version_{version_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(manifest, f)
        print(f"Version saved: {version_name}")

    def load_version(self, version_name):
        """Return (ids, embeddings) for a previously saved version."""
        path = self.cache_path.parent / f"version_{version_name}.pkl"
        with open(path, "rb") as f:
            manifest = pickle.load(f)
        ids = [m[0] for m in manifest]
        embs = np.array([self.store[m[1]] for m in manifest])
        return ids, embs


# ─────────────────────────────────────────────
# 3. MAIN ENCODER
# ─────────────────────────────────────────────

class SamsungPRISMEncoder(AbsEncoder):
    """
    Samsung PRISM — Code Retrieval Pipeline

    Key design decisions:
    1. jina-embeddings-v2-base-code: 
       handles 8192 tokens (vs bge-small's 512)
       trained on code — knows Python semantics
       Fixes the 71% query truncation problem
    
    2. Query preprocessing:
       strips example sections, adds search prefix
    
    3. Code preprocessing:
       removes boilerplate input parsing noise
    
    4. Hash-based cache:
       only re-embeds changed snippets (P1 requirement)
    """

    def __init__(
        self,
        model_name="jinaai/jina-embeddings-v2-base-code",
        max_seq_length=1024,
        use_cache=True
    ):
        print(f"Loading: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            trust_remote_code=True
        )
        self.model.max_seq_length = max_seq_length
        self.model_name = model_name
        self.cache = EmbeddingCache() if use_cache else None
        print("Encoder ready")

    def encode(self, inputs, *, task_metadata, hf_split,
               hf_subset, prompt_type=None, **kwargs):

        texts = [item["text"] for item in inputs]

        if prompt_type == PromptType.query:
            texts = [prep_query(t) for t in texts]
        else:
            texts = [prep_code(t) for t in texts]

        if self.cache:
            return self.cache.embed_with_cache(
                texts, self.model, batch_size=32
            )

        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )


# ─────────────────────────────────────────────
# 4. RUN EVALUATION
# ─────────────────────────────────────────────

def run_eval(model, label=""):
    print(f"\n{'='*50}")
    print(f"Running eval: {label}")
    print(f"{'='*50}")

    task = mteb.get_task("AppsRetrieval")
    start = time.time()

    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 32}
    )

    elapsed = time.time() - start
    task_result = list(result.task_results)[0]

    print(f"\nTime: {elapsed/60:.1f} min")
    print(f"\n=== {label} RESULTS ===")
    print(task_result)

    return task_result, elapsed


if __name__ == "__main__":
    model = SamsungPRISMEncoder()
    task_result, elapsed = run_eval(model, "jina-v2-code + preprocessing")

    with open("appsretrieval_results.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2)

    print("\n Saved appsretrieval_results.json")
