# TrustFed — Trustworthy Federated Large Language Models

**🇬🇧 English** · [🇻🇳 Tiếng Việt](README.vi.md)

A personal learning project to understand — hands-on — what it takes to make
**federated, privacy-preserving fine-tuning of language models** trustworthy. It
is built in two stages, from "understand plain federated learning" to "run a
working federated LLM adaptation pipeline", so each idea in the problem (privacy,
learning without labels, global↔local knowledge transfer) becomes concrete rather
than abstract.

- **The problem, written out:** [`PROPOSAL.md`](PROPOSAL.md) — a short write-up of
  *TrustFed*: F-LLMs that are private by design, learn from unlabelled data, and
  transfer knowledge efficiently between large global and small local models.
- **The deep-dive (Vietnamese):** [`HUONG_DAN.md`](HUONG_DAN.md) — every concept and
  code section explained from scratch.
- **The experiments:** the two code stages below, and [`RESULTS.md`](RESULTS.md).

---

## Stage 0 — FedAvg from first principles (MNIST)

A minimal Federated Learning demo to internalise the client/server split.

- [`centralized.py`](centralized.py) — trains a small CNN the normal way (baseline).
- [`federated.py`](federated.py) — splits MNIST across 5 virtual clients and trains
  with **FedAvg** using Flower's simulation engine; only model updates are shared.

```bash
python centralized.py    # baseline accuracy
python federated.py      # federated with FedAvg
```

**Takeaways:** how FedAvg aggregates weighted client updates; why non-IID data and
communication cost make FL harder than centralized training; and why *"not sharing
raw data" ≠ privacy* — updates still leak, which motivates Stage 1 and the proposal.

## Stage 1 — Federated LoRA on a pre-trained LLM (the flagship)

[`fed_lora.py`](fed_lora.py) is the artifact that bridges Stage 0 to the proposal.
It fine-tunes a **frozen, pre-trained BERT-tiny** with **LoRA (implemented from
scratch, no `peft`)** across **non-IID clients**, and demonstrates all three
pillars of TrustFed in one script that runs in a few minutes on a CPU:

| Pillar | How the demo shows it |
|---|---|
| **Privacy-by-design** | only tiny LoRA adapters are communicated (~500× smaller than a full-model update); optional **user-level DP-FedAvg** (clip + Gaussian noise) as a formal privacy knob |
| **Efficient knowledge transfer** | FedAvg over adapters transfers knowledge across clients; we measure the **lift of federated over local-only** under domain skew |
| **Practicality** | frozen foundation model + parameter-efficient adapters + a hand-rolled, transparent federated loop (no Ray) that is robust on Windows |

```bash
pip install -r requirements.txt
python fed_lora.py            # full comparison on ag_news (downloads BERT-tiny once)
python fed_lora.py --quick    # tiny + fast smoke test
python fed_lora.py --dp-noise 0.3   # stronger differential privacy
```

The script runs four regimes — **centralized** (upper bound), **local-only** (no
collaboration), **federated LoRA**, and **federated LoRA + DP** — then writes a
results table to [`RESULTS.md`](RESULTS.md) (and `results.png` if `matplotlib` is
installed). See the proposal for how these map to the research questions.

### What makes this a credible F-LLM prototype (and not a toy)
- A **real pre-trained backbone**: LoRA only makes sense on top of a pre-trained
  model, so we freeze real BERT-tiny weights — averaging adapters across a *shared*
  backbone is the subtle correctness condition most naive demos get wrong.
- **Non-IID by design**: a Dirichlet partition gives each client a different domain
  mix, the realistic F-LLM setting where naive FL struggles.
- **Honest metrics**: we report the privacy cost of DP and the communication
  payload explicitly, not just a single accuracy number.

## Repository map

| File | What it is |
|---|---|
| [`PROPOSAL.md`](PROPOSAL.md) | The research proposal (read this first) |
| [`HUONG_DAN.md`](HUONG_DAN.md) | In-depth Vietnamese walkthrough of the whole project (for the author) |
| [`fed_lora.py`](fed_lora.py) | **Flagship**: federated LoRA on a pre-trained LLM |
| [`federated.py`](federated.py) / [`centralized.py`](centralized.py) | Stage 0: FedAvg vs centralized on MNIST |
| [`RESULTS.md`](RESULTS.md) | Auto-generated results from `fed_lora.py` |
| [`requirements.txt`](requirements.txt) | Dependencies (Stage 0 + Stage 1) |

## Roadmap (from the proposal)
Scale the backbone (DistilBERT/GPT-2) → map the DP privacy–utility frontier →
federated **self-supervised** (label-free) adaptation → **bidirectional
distillation** for the global-large ↔ local-small setting.
