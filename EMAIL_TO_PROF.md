# Pitch email to the professor (template)

**🇬🇧 English** · [🇻🇳 Tiếng Việt](EMAIL_TO_PROF.vi.md)

> Short, concrete, and evidence-first. Swap the bracketed parts, keep it under
> ~200 words. The goal of the email is to earn a 20-minute meeting, not to close
> the deal — so lead with what you already *did*, and attach the repo.

---

**Subject:** RA interest — I built a working federated LoRA fine-tuning prototype for F-LLMs

Dear Professor [Wong],

I am [Your Name], a [year / program] student at VinUniversity. I am writing to ask
about a Research Assistant position in your group working on **trustworthy
federated large language models**.

Rather than only expressing interest, I built a small prototype to make sure I
understand the core problem. It is a self-contained pipeline that:

- fine-tunes a **frozen, pre-trained transformer with LoRA** across **5 non-IID
  clients**, communicating only the adapters — a **~500× smaller** payload than a
  full-model update, and a smaller privacy attack surface;
- shows a measurable **knowledge-transfer gain** from federation over local-only
  training under domain skew, while raw data never leaves each client;
- includes a switchable **user-level differential-privacy** mechanism so the
  privacy–utility trade-off can be measured directly.

I wrote up how this connects to privacy-by-design, label-free adaptation, and
global↔local knowledge transfer in a short proposal (attached: `PROPOSAL.md`), and
the code is here: [repo link].

Could I have 15–20 minutes to hear where I could be most useful? I would be happy
to start by reproducing a result from your recent work.

Thank you for your time.

Best regards,
[Your Name]
[email] · [phone] · [GitHub]

---

### Attach / link
- `PROPOSAL.md` — 2-page research proposal (TrustFed).
- Repository — `README.md`, `fed_lora.py` (flagship), `RESULTS.md` (numbers), `results.png`.

### Follow-up etiquette
- If no reply in ~5 working days, send one short, polite follow-up.
- Before any meeting, skim 2–3 of the group's recent papers and have one concrete
  question ready about each.
