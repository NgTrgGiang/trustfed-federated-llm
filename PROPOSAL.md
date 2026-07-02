# TrustFed: Trustworthy Federated Large Language Models

**A short research write-up — a personal project to understand the problem**

*[Your Name], VinUniversity — [date]*

**🇬🇧 English** · [🇻🇳 Tiếng Việt](PROPOSAL.vi.md)

---

## 1. Problem and motivation

Large Language Models (LLMs) are trained on ever-larger centralized corpora. Yet
the data that would make them most useful in practice — clinical notes, financial
records, legal documents, enterprise logs, on-device user text — is exactly the
data that **cannot be centralized**, because of privacy expectations and binding
regulation (GDPR, HIPAA, Vietnam's PDPD 2023, the EU AI Act). The result is a
widening gap: the organizations with the most valuable domain data are the least
able to use it to adapt foundation models.

**Federated Learning (FL)** offers a way out: many parties collaboratively train a
model while their raw data never leaves their premises; only model updates are
shared. But moving from small classifiers to *Federated LLMs (F-LLMs)* is not a
simple substitution. Three obstacles stand in the way, and they map directly onto
the three goals of this project:

1. **Privacy is not free.** "Not sharing raw data" is not the same as privacy —
   model updates (gradients, weights, even adapters) leak information and are
   vulnerable to membership-inference and reconstruction attacks. F-LLMs need
   *privacy-by-design*, with formal guarantees, not just data locality.
2. **Labels are scarce and legally fraught.** In regulated domains, labelled data
   is expensive and its use is often restricted. Practical F-LLMs must learn from
   **unlabelled** text via self-supervision, in a way that is *legally compliant*.
3. **Clients are heterogeneous and small.** A hospital cannot host a 70B model; a
   phone cannot run a 7B one. We need **efficient knowledge transfer** between a
   large global model and smaller, domain-adapted local models — in both
   directions — under a tight communication budget.

**Goal.** This project (working title *TrustFed*) aims to develop methods that
make F-LLMs **practical and trustworthy** in exactly these settings: private by
design, able to learn without labels, and efficient enough to run across
heterogeneous, resource-constrained clients.

## 2. Why now, and where the gap is

The building blocks now exist but have not been unified:

- **FedAvg** (McMahan et al., 2017) made collaborative training practical, but
  degrades badly under **non-IID / domain-skewed** data (Zhao et al., 2018) —
  the norm for real F-LLM clients.
- **Parameter-efficient fine-tuning (LoRA;** Hu et al., 2021) makes adapting a
  frozen foundation model cheap — a natural fit for FL, since only tiny adapters
  need to be communicated (FedIT / OpenFedLLM; Zhang et al., 2023; Ye et al.,
  2024). But naive federation of LoRA adapters interacts poorly with client
  heterogeneity and privacy noise.
- **Differential Privacy** (DP-SGD, Abadi et al., 2016; DP-FedAvg, McMahan et al.,
  2018) and **secure aggregation** (Bonawitz et al., 2017) give formal privacy,
  but the privacy–utility trade-off on LLM adapters is poorly understood.
- **Federated knowledge distillation** (FedDF, Lin et al., 2020; FedKD, Wu et al.,
  2022) enables model-heterogeneous collaboration, but has mostly been studied on
  vision, not on the global-large ↔ local-small LLM setting.

**The gap TrustFed targets:** a single framework where privacy-by-design,
label-free self-supervision, and global↔local knowledge transfer are co-designed
for parameter-efficient F-LLMs — and where the trade-offs between them are
measured, not assumed. Kairouz et al. (2021) explicitly list these as open
problems in FL.

## 3. Research questions

- **RQ1 (Privacy).** How much formal privacy (DP) and how much attack-surface
  reduction (adapter-only, secure aggregation) can we buy for how much utility,
  when fine-tuning a *frozen* foundation model with LoRA in a federated setting?
- **RQ2 (Unsupervised, compliant).** Can federated **self-supervised** adaptation
  (masked / causal language modelling on unlabelled client text) recover most of
  the benefit of labelled fine-tuning, while keeping a clear data-provenance and
  compliance story?
- **RQ3 (Knowledge transfer).** Can we transfer knowledge **bidirectionally** —
  distilling a large global model into small, domain-adapted local models and
  aggregating local expertise back into the global model — better than FedAvg
  under strong domain skew?

## 4. Proposed approach — the TrustFed framework

TrustFed keeps a **frozen pre-trained backbone** on every client and trains only
lightweight, communicable modules. This single decision drives all three pillars:
it shrinks the payload ~100–500×, shrinks the privacy attack surface, and makes
model-heterogeneous distillation tractable.

**Pillar 1 — Privacy-by-design.** Communicate only LoRA adapters, never raw data
or full weights. Layer on (a) *user-level DP-FedAvg* (per-client update clipping +
calibrated Gaussian noise) for a formal `(ε, δ)` guarantee, and (b) *secure
aggregation* so the server sees only the sum of updates. Study the privacy–utility
frontier as a function of rank `r`, clip norm `C`, and noise `σ`.

**Pillar 2 — Legally-compliant unsupervised learning.** Replace the labelled
objective with federated **self-supervised** adaptation (MLM/CLM) on unlabelled
client text, so clients contribute without ever exposing labels or documents.
Pair this with a *data-provenance and consent ledger* per client so the training
process is auditable against regulatory requirements.

**Pillar 3 — Efficient global↔local knowledge transfer.** Use FedAvg of adapters
as the baseline aggregator, then improve it with **bidirectional distillation**:
the large global model teaches small local models on unlabelled public/proxy data
(global→local), and local adapters are aggregated (heterogeneity-aware) back into
the global model (local→global). This supports clients that cannot host the full
model at all — they keep only a distilled student.

## 5. Preliminary work (already done — see this repository)

I have already built a working, end-to-end pipeline that de-risks the core
mechanics and demonstrates I can execute this agenda.

**Stage 0 — FedAvg from first principles** (`federated.py`, Flower + MNIST). A
5-client FedAvg simulation reproducing the centralized baseline, to internalise
the client/server split and the non-IID challenge.

**Stage 1 — Federated LoRA on a pre-trained LLM** (`fed_lora.py`). The flagship
artifact, touching all three pillars in one script that runs in minutes on a CPU:

- A **frozen, pre-trained BERT-tiny** backbone adapted with **LoRA implemented
  from scratch** (no `peft`), on **ag_news** with a **non-IID Dirichlet** split
  across 5 clients — a realistic "different domains per client" setting.
- Only **8,708 adapter parameters** are communicated per round, vs. **4,394,628**
  for a full-model update — a **~505× smaller payload** and a correspondingly
  smaller privacy attack surface.
- **User-level DP-FedAvg** (update clipping + Gaussian noise) as a switchable
  privacy knob, so the privacy–utility trade-off can be measured directly.

**Headline preliminary results** (`RESULTS.md`, test accuracy on ag_news):

| Setting | Accuracy |
|---|---|
| Centralized LoRA (privacy-free upper bound) | **0.862** |
| Local-only LoRA (no collaboration) | 0.560 |
| **Federated LoRA (raw data stays local)** | **0.721** |
| Federated LoRA + user-level DP (C=1.0, σ=0.02) | 0.636 |

The key finding is the **knowledge-transfer lift of +0.161** from local-only
(0.560) to federated (0.721): under domain skew, collaboration measurably helps
the weakest clients, while raw data never moves. Adding user-level DP costs only
~8.5 points (0.721 → 0.636) at this noise level and still converges — a first,
concrete point on the privacy–utility frontier that TrustFed proposes to map.

## 6. Directions to explore next (indicative)

| Phase | Focus | Deliverable |
|---|---|---|
| 1 (m1–2) | Reproduce + scale the harness to DistilBERT/GPT-2; add heterogeneity-aware baselines | Reproducible benchmark suite |
| 2 (m3–5) | **RQ1**: DP + secure aggregation on adapters; map the privacy–utility frontier | Empirical study + `(ε,δ)` accounting |
| 3 (m5–8) | **RQ2**: federated self-supervised (MLM/CLM) adaptation on unlabelled text | Label-free adaptation results |
| 4 (m8–11) | **RQ3**: bidirectional global↔local distillation under domain skew | New aggregation method |
| 5 (m11–12) | Integrate, ablate, write up | Workshop/conference paper draft |

## 7. Expected contributions

1. An **open, reproducible F-LLM benchmark** that jointly varies privacy, label
   availability, and client heterogeneity — most prior work varies one at a time.
2. A **measured privacy–utility–communication frontier** for federated LoRA.
3. A **bidirectional distillation** method for the global-large ↔ local-small
   setting, with clients that need never host the full model.
4. A **compliance-oriented training protocol** (provenance + consent ledger) that
   makes the "legally compliant unsupervised learning" claim concrete.

## 8. What building this taught me

Building this end-to-end turned the abstract problem into concrete understanding.
Starting from a blank repository I built a correct FedAvg simulation, then a
federated LoRA fine-tuner of a real pre-trained transformer with from-scratch LoRA,
non-IID partitioning, user-level DP, and communication accounting — and I debugged
the subtle failure modes along the way (e.g., that averaging adapters only makes
sense on a *shared, pre-trained* backbone; that DP noise must be scaled by the
number of clients). The biggest lesson: each pillar (privacy, label-free learning,
global↔local transfer) is easy to state but full of hidden trade-offs once you
actually implement and *measure* it — which is why building it, rather than only
reading about it, was worth doing.

## References

1. McMahan et al. *Communication-Efficient Learning of Deep Networks from
   Decentralized Data* (FedAvg). AISTATS, 2017.
2. Zhao et al. *Federated Learning with Non-IID Data.* arXiv:1806.00582, 2018.
3. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
4. Zhang et al. *Towards Building the Federated GPT: Federated Instruction Tuning
   (FedIT).* 2023.
5. Ye et al. *OpenFedLLM: Training LLMs on Decentralized Private Data via Federated
   Learning.* KDD, 2024.
6. Abadi et al. *Deep Learning with Differential Privacy* (DP-SGD). CCS, 2016.
7. McMahan et al. *Learning Differentially Private Recurrent Language Models*
   (DP-FedAvg). ICLR, 2018.
8. Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine
   Learning.* CCS, 2017.
9. Lin et al. *Ensemble Distillation for Robust Model Fusion in Federated Learning*
   (FedDF). NeurIPS, 2020.
10. Wu et al. *Communication-Efficient Federated Learning via Knowledge
    Distillation* (FedKD). Nature Communications, 2022.
11. Kairouz et al. *Advances and Open Problems in Federated Learning.* Foundations
    and Trends in ML, 2021.
