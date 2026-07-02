"""
Stage 1 (flagship): Federated LoRA fine-tuning of a small, PRE-TRAINED language
model (BERT-tiny) — the artifact that bridges the MNIST/FedAvg warm-up (Stage 0)
to the TrustFed research proposal.

It runs in a few minutes on a CPU and demonstrates the three pillars of TrustFed
in one self-contained script:

  1. Privacy-by-design   -> clients keep their raw text; only tiny LoRA adapters
                            are communicated (not the data, not the full model),
                            and an optional user-level DP mechanism (clip +
                            Gaussian noise, McMahan et al. 2018) can be switched
                            on to bound any single client's influence.
  2. Efficient knowledge -> a FROZEN pre-trained transformer (BERT-tiny, a real
     transfer               foundation model) is adapted with LoRA (Hu et al.
                            2021). FedAvg over the adapters transfers knowledge
                            between many local models and one global model, which
                            we show helps the *weakest* non-IID clients
                            (federated > local-only).
  3. Practicality         -> everything is parameter-efficient: we report the
                            communication payload (LoRA params) vs. a full-model
                            update to quantify the ~500x saving.

Design choices worth defending to a reviewer:
  * Real pre-trained backbone. LoRA only makes sense on top of a pre-trained
    model; we use BERT-tiny and freeze it, so the adapters we average are
    meaningful. Swapping in DistilBERT / a larger checkpoint is a one-line change.
  * LoRA is implemented FROM SCRATCH (no `peft`) so the mechanism is explicit and
    dependency-light — it shows we understand PEFT internals, not just the API.
  * No Flower/Ray. The federated loop is hand-rolled (~30 lines) so it is fully
    transparent, deterministic, and robust on Windows. Stage 0 already shows we
    can use Flower; here, control over adapter aggregation + DP matters more.

First run downloads BERT-tiny (~17 MB) and the BERT tokenizer once, then caches.

Run:
    python fed_lora.py                 # full comparison on ag_news
    python fed_lora.py --rounds 8      # more federated rounds
    python fed_lora.py --dp-noise 0.5  # stronger user-level DP-FedAvg
    python fed_lora.py --quick         # tiny + fast smoke test
"""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from collections import Counter, OrderedDict

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertModel, BertTokenizerFast
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_ID = "prajjwal1/bert-tiny"       # a real, tiny pre-trained transformer (~4.4M)
TOKENIZER_ID = "bert-base-uncased"    # BERT-tiny reuses this WordPiece vocab
NUM_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# --------------------------------------------------------------------------- #
# 1. Data: a real multi-domain text dataset (ag_news) with an offline fallback #
# --------------------------------------------------------------------------- #
def load_texts(n_train: int, n_test: int, seed: int):
    """Return (train_texts, train_labels, test_texts, test_labels, source).

    Tries the real ag_news topic set (4 news domains). If the hub is unreachable,
    falls back to a synthetic multi-domain generator so the demo *always* runs.
    """
    try:
        from datasets import load_dataset

        ds = None
        for repo in ("fancyzhx/ag_news", "SetFit/ag_news", "ag_news"):
            try:
                ds = load_dataset(repo)
                break
            except Exception:  # noqa: BLE001 - try the next mirror
                continue
        if ds is None:
            raise RuntimeError("no ag_news mirror reachable")
        tr = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
        te = ds["test"].shuffle(seed=seed).select(range(min(n_test, len(ds["test"]))))
        return (list(tr["text"]), list(tr["label"]),
                list(te["text"]), list(te["label"]), "ag_news")
    except Exception as e:  # noqa: BLE001 - any failure -> offline fallback
        print(f"[data] ag_news unavailable ({type(e).__name__}); using synthetic "
              f"multi-domain text so the demo still runs.")
        return (*_synthetic_multidomain(n_train, n_test, seed), "synthetic")


def _synthetic_multidomain(n_train: int, n_test: int, seed: int):
    """4 pseudo-topics with distinctive vocabularies — a learnable, non-trivial
    task mirroring ag_news' domain structure, fully offline."""
    rng = np.random.default_rng(seed)
    topic_words = {
        0: "government election border treaty nation minister protest war peace summit",
        1: "match goal season player coach league score final champion tournament",
        2: "market stocks profit revenue investor economy trade shares bank merger",
        3: "software chip data model research quantum satellite algorithm device network",
    }
    filler = "the a of to in and for on with that this from it as at by an is are".split()

    def make(n):
        texts, labels = [], []
        for _ in range(n):
            c = int(rng.integers(0, 4))
            kw = topic_words[c].split()
            length = int(rng.integers(12, 28))
            words = list(rng.choice(kw, size=length // 2)) + \
                    list(rng.choice(filler, size=length - length // 2))
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(c)
        return texts, labels

    tr_x, tr_y = make(n_train)
    te_x, te_y = make(n_test)
    return tr_x, tr_y, te_x, te_y


def make_loader(texts, labels, tok, max_len, batch_size, shuffle):
    enc = tok(list(texts), padding="max_length", truncation=True,
              max_length=max_len, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(labels, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def dirichlet_partition(labels, num_clients, alpha, seed):
    """Non-IID label partition (Dirichlet) — the standard FL heterogeneity setup.
    Small alpha => each client is dominated by a few 'domains'."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    idx_by_class = [np.where(labels == c)[0] for c in range(labels.max() + 1)]
    client_idx = [[] for _ in range(num_clients)]
    for idxs in idx_by_class:
        rng.shuffle(idxs)
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idxs, cuts)):
            client_idx[cid].extend(chunk.tolist())
    for c in client_idx:
        rng.shuffle(c)
    return client_idx


# --------------------------------------------------------------------------- #
# 2. Model: a frozen pre-trained BERT-tiny + from-scratch LoRA on attention Q,V #
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Wrap a (frozen) pre-trained Linear with a trainable low-rank update:
        y = W0 x + (alpha/r) * B A x
    B is zero-initialised so the adapter starts as a no-op (Hu et al. 2021)."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, in_f) * (1.0 / math.sqrt(in_f)))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()


def inject_lora(module, r, alpha, targets=("query", "value")):
    """Recursively replace the named Linear sub-modules with LoRA-wrapped ones."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            inject_lora(child, r, alpha, targets)


class LoRABert(nn.Module):
    """Frozen pre-trained BERT-tiny + LoRA adapters on attention Q,V + a trainable
    classifier head. The adapters and head are the ONLY tensors that are trained
    and, in the federated setting, the only tensors ever communicated."""
    def __init__(self, r=8, alpha=16, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_ID)
        for p in self.bert.parameters():        # freeze the foundation model
            p.requires_grad_(False)
        inject_lora(self.bert.encoder, r, alpha)  # add trainable low-rank adapters
        self.head = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, ids, mask):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)   # mean-pool real tokens
        return self.head(pooled)


# --------------------------------------------------------------------------- #
# 3. Adapter (trainable-only) state helpers + federated averaging + DP          #
# --------------------------------------------------------------------------- #
def adapter_state(model):
    """Only the trainable tensors (LoRA + head) — this is the communicated payload."""
    return OrderedDict(
        (n, p.detach().clone()) for n, p in model.named_parameters() if p.requires_grad
    )


def load_adapter(model, state):
    own = dict(model.named_parameters())
    for n, v in state.items():
        own[n].data.copy_(v)


def flat_norm(state):
    return torch.sqrt(sum((v.float() ** 2).sum() for v in state.values()))


def local_train(model, loader, epochs, lr):
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    for _ in range(epochs):
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(ids, mask), y).backward()
            opt.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        pred = model(ids, mask).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def fedavg_deltas(global_state, client_states, weights, dp_clip=0.0, dp_noise=0.0):
    """Average client *updates* (deltas) with optional user-level DP.

    User-level DP-FedAvg (McMahan et al. 2018): clip each client's update to an
    L2 bound C, average, then add Gaussian noise with std (sigma*C)/N (the average
    of N clipped updates has sensitivity C/N). This bounds any single client's
    influence — a formal privacy knob on top of the fact that raw data never
    leaves the device.
    """
    w = np.array(weights, dtype=float)
    w /= w.sum()
    if dp_clip > 0 or dp_noise > 0:
        return _apply_user_level_dp(global_state, client_states, w, dp_clip, dp_noise)
    new = OrderedDict((n, v.clone()) for n, v in global_state.items())
    for n in new:
        agg = torch.zeros_like(new[n])
        for cs, wi in zip(client_states, w):
            agg += wi * (cs[n] - global_state[n])
        new[n] = global_state[n] + agg
    return new


def _apply_user_level_dp(global_state, client_states, w, clip, noise):
    keys = list(global_state.keys())
    out = OrderedDict((n, global_state[n].clone()) for n in keys)
    agg = OrderedDict((n, torch.zeros_like(global_state[n])) for n in keys)
    for cs, wi in zip(client_states, w):
        delta = OrderedDict((n, cs[n] - global_state[n]) for n in keys)
        if clip > 0:                                   # clip the whole update norm
            factor = min(1.0, clip / (flat_norm(delta).item() + 1e-12))
            for n in keys:
                delta[n] = delta[n] * factor
        for n in keys:
            agg[n] += wi * delta[n]
    if noise > 0:                                      # add calibrated Gaussian noise
        # Sensitivity of the *average* of N clipped updates to adding/removing one
        # client is C/N, so the Gaussian mechanism noise std is (noise * C) / N.
        n_clients = max(len(client_states), 1)
        sigma = noise * (clip if clip > 0 else 1.0) / n_clients
        for n in keys:
            agg[n] += torch.randn_like(agg[n]) * sigma
    for n in keys:
        out[n] = global_state[n] + agg[n]
    return out


# --------------------------------------------------------------------------- #
# 4. Experiments: centralized upper bound, local-only, and federated (+DP)      #
# --------------------------------------------------------------------------- #
def build_model(cfg):
    return LoRABert(r=cfg.rank, alpha=cfg.alpha_lora).to(DEVICE)


def param_report(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_centralized(cfg, train_loader, test_loader):
    model = build_model(cfg)
    for _ in range(cfg.rounds):
        local_train(model, train_loader, cfg.local_epochs, cfg.lr)
    return evaluate(model, test_loader)


def run_local_only(cfg, client_loaders, test_loader):
    """Each client trains its own adapter on its own shard — no communication.
    This is the 'no knowledge transfer' baseline that non-IID data hurts most."""
    accs = []
    for loader in client_loaders:
        model = build_model(cfg)
        for _ in range(cfg.rounds):
            local_train(model, loader, cfg.local_epochs, cfg.lr)
        accs.append(evaluate(model, test_loader))
    return float(np.mean(accs)), accs


def run_federated(cfg, client_loaders, test_loader, dp_clip=0.0, dp_noise=0.0):
    global_model = build_model(cfg)
    clients = [build_model(cfg) for _ in client_loaders]  # built once, reused each round
    global_state = adapter_state(global_model)
    sizes = [len(l.dataset) for l in client_loaders]
    history = []
    for rnd in range(1, cfg.rounds + 1):
        client_states = []
        for local, loader in zip(clients, client_loaders):
            load_adapter(local, global_state)                 # broadcast global adapter
            local_train(local, loader, cfg.local_epochs, cfg.lr)  # local step
            client_states.append(adapter_state(local))         # send adapter back
        global_state = fedavg_deltas(global_state, client_states, sizes,
                                     dp_clip=dp_clip, dp_noise=dp_noise)
        load_adapter(global_model, global_state)
        acc = evaluate(global_model, test_loader)
        history.append(acc)
        tag = " +DP" if (dp_noise > 0 or dp_clip > 0) else ""
        print(f"  [fed-lora{tag}] round {rnd:>2}/{cfg.rounds}  global acc = {acc:.4f}")
    return history, global_state


def main():
    p = argparse.ArgumentParser(description="Federated LoRA fine-tuning (TrustFed Stage 1)")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.3, help="Dirichlet non-IID (small=more skew)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-alpha", dest="alpha_lora", type=int, default=16)
    p.add_argument("--max-len", dest="max_len", type=int, default=64)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--dp-clip", type=float, default=0.0, help="user-level DP clip norm C")
    p.add_argument("--dp-noise", type=float, default=0.0, help="user-level DP noise multiplier sigma")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="tiny+fast smoke test")
    args = p.parse_args()
    dirichlet_alpha = args.alpha  # --alpha is the Dirichlet knob; LoRA's alpha is --lora-alpha

    if args.quick:
        args.n_train, args.n_test, args.rounds = 1200, 800, 3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print("TrustFed - Stage 1: Federated LoRA fine-tuning of pre-trained BERT-tiny")
    print("=" * 72)
    tok = BertTokenizerFast.from_pretrained(TOKENIZER_ID)
    tr_x, tr_y, te_x, te_y, source = load_texts(args.n_train, args.n_test, args.seed)
    print(f"[data] source={source}  train={len(tr_x)}  test={len(te_x)}  classes={CLASS_NAMES}")

    test_loader = make_loader(te_x, te_y, tok, args.max_len, 256, False)
    pooled_loader = make_loader(tr_x, tr_y, tok, args.max_len, 32, True)

    parts = dirichlet_partition(tr_y, args.clients, dirichlet_alpha, args.seed)
    client_loaders = []
    print(f"[split] non-IID Dirichlet(alpha={dirichlet_alpha}) across {args.clients} clients:")
    for cid, idx in enumerate(parts):
        cx, cy = [tr_x[i] for i in idx], [tr_y[i] for i in idx]
        dist = Counter(cy)
        client_loaders.append(make_loader(cx, cy, tok, args.max_len, 32, True))
        share = {CLASS_NAMES[c][:4]: dist.get(c, 0) for c in range(4)}
        print(f"        client {cid}: n={len(cy):>4}  domain mix={share}")

    trainable, total = param_report(build_model(args))
    print(f"[model] BERT-tiny frozen; trainable(=communicated) = {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)  ->  ~{total/max(trainable,1):.0f}x smaller payload")

    print("\n[1/4] Centralized LoRA (privacy-free upper bound) ...")
    t0 = time.time()
    acc_central = run_centralized(args, pooled_loader, test_loader)
    print(f"      centralized acc = {acc_central:.4f}   ({time.time()-t0:.0f}s)")

    print("\n[2/4] Local-only LoRA (no knowledge transfer) ...")
    acc_local, per_client = run_local_only(args, client_loaders, test_loader)
    print(f"      local-only mean acc = {acc_local:.4f}   per-client={[round(a,3) for a in per_client]}")

    print("\n[3/4] Federated LoRA (FedAvg over adapters) ...")
    hist_fed, _ = run_federated(args, client_loaders, test_loader)
    acc_fed = hist_fed[-1]

    print("\n[4/4] Federated LoRA + user-level DP ...")
    dp_clip = args.dp_clip if args.dp_clip > 0 else 1.0
    dp_noise = args.dp_noise if args.dp_noise > 0 else 0.02
    hist_dp, _ = run_federated(args, client_loaders, test_loader,
                               dp_clip=dp_clip, dp_noise=dp_noise)
    acc_dp = hist_dp[-1]

    # ---- summary + artifacts -------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY  (test accuracy)")
    print("=" * 72)
    rows = [
        ("Centralized LoRA (upper bound)", acc_central, "all data pooled (no privacy)"),
        ("Local-only LoRA (no transfer)", acc_local, "each client alone, non-IID hurts"),
        ("Federated LoRA (FedAvg adapters)", acc_fed, "raw data stays local"),
        (f"Federated LoRA + DP (C={dp_clip},s={dp_noise})", acc_dp, "+ formal privacy knob"),
    ]
    for name, acc, note in rows:
        print(f"  {name:<40} {acc:.4f}   {note}")
    print(f"\n  >> Knowledge-transfer lift (federated - local-only): {acc_fed - acc_local:+.4f}")
    print(f"  >> Privacy cost of DP (federated - DP): {acc_fed - acc_dp:+.4f}")
    print(f"  >> Communication payload: {trainable:,} params/round vs {total:,} full-model"
          f"  (~{total/max(trainable,1):.0f}x cheaper)")

    _write_results_md(rows, hist_fed, hist_dp, trainable, total, source, dirichlet_alpha)
    _maybe_plot(hist_fed, hist_dp, acc_central, acc_local)
    print("\nWrote RESULTS.md" + (" and results.png" if _HAS_MPL else "") + ". Done.")


# --------------------------------------------------------------------------- #
# 5. Artifacts: a markdown results table + optional plot                        #
# --------------------------------------------------------------------------- #
try:
    import matplotlib  # noqa: F401
    _HAS_MPL = True
except Exception:  # noqa: BLE001
    _HAS_MPL = False


def _write_results_md(rows, hist_fed, hist_dp, trainable, total, source, alpha):
    lines = [
        "# Stage 1 results — Federated LoRA fine-tuning of BERT-tiny\n",
        f"_Auto-generated by `fed_lora.py`. Backbone: **BERT-tiny (frozen, pre-trained)**; "
        f"data source: **{source}**; non-IID Dirichlet(alpha={alpha})._\n",
        "## Test accuracy\n",
        "| Setting | Accuracy | Note |",
        "|---|---|---|",
    ]
    for name, acc, note in rows:
        lines.append(f"| {name} | {acc:.4f} | {note} |")
    lines += [
        "\n## Federated accuracy per round\n",
        "| Round | FedAvg-LoRA | FedAvg-LoRA + DP |",
        "|---|---|---|",
    ]
    for i, (a, b) in enumerate(zip(hist_fed, hist_dp), 1):
        lines.append(f"| {i} | {a:.4f} | {b:.4f} |")
    lines += [
        "\n## Communication / privacy footprint\n",
        f"- Communicated per client per round: **{trainable:,}** params "
        f"(LoRA adapters + head).",
        f"- A full-model update would be **{total:,}** params "
        f"(~{total/max(trainable,1):.0f}x larger).",
        "- Raw text never leaves the client; only adapters are shared, and "
        "user-level DP bounds any single client's influence.",
    ]
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _maybe_plot(hist_fed, hist_dp, acc_central, acc_local):
    if not _HAS_MPL:
        return
    import matplotlib.pyplot as plt

    rounds = range(1, len(hist_fed) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(rounds, hist_fed, "o-", label="Federated LoRA")
    plt.plot(rounds, hist_dp, "s--", label="Federated LoRA + DP")
    plt.axhline(acc_central, color="green", ls=":", label="Centralized (upper bound)")
    plt.axhline(acc_local, color="red", ls=":", label="Local-only (no transfer)")
    plt.xlabel("Federated round")
    plt.ylabel("Global test accuracy")
    plt.title("Federated LoRA on BERT-tiny: knowledge transfer vs. privacy cost")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results.png", dpi=120)


if __name__ == "__main__":
    main()
