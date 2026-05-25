# ETHICS.md — ForgeNet-X Responsible Use Statement

## System Purpose

ForgeNet-X is a **forensic detection system**. Its primary mission is to authenticate handwriting samples and detect signature forgeries. The synthetic handwriting generator is a secondary research sub-feature that serves the detector — not a standalone imitation tool.

---

## Dual-Use Nature

ForgeNet-X involves dual-use technology: the same techniques used to generate synthetic handwriting can theoretically be used to produce deceptive samples. This is a known and acknowledged characteristic of the field. The following mechanisms are in place to mitigate misuse:

| Mechanism | Description |
|-----------|-------------|
| **Consent gate** | A mandatory checkbox requires users to declare research or testing intent before any generation is permitted |
| **Automatic watermark** | Every generated image has `[SYNTHETIC – ForgeNet-X]` baked in via alpha-compositing — it cannot be easily removed without visible degradation |
| **Audit log** | Every generation event is written to `outputs/generation_log.jsonl` with a timestamp and declared purpose |
| **Origin analysis** | Every uploaded handwriting sample is automatically scored for synthetic vs. human origin using EXIF metadata and pixel-level heuristics |

---

## Intended Use Cases

The following are the only authorised uses of this system:

1. **Academic research** — studying handwriting analysis, document forensics, or adversarial ML
2. **Detector training** — generating synthetic samples to augment training datasets
3. **Stress-testing** — probing the authenticity detector to expose weaknesses
4. **Forensic document examination** — supporting (not replacing) expert human review of contested signatures or handwriting samples
5. **Banking and legal document verification** — as a preliminary screening tool, subject to human expert confirmation
6. **Accessibility research** — understanding handwriting variation for assistive technology

---

## Prohibited Use Cases

The following uses are explicitly prohibited:

- Submitting synthetic handwriting as genuine work in any academic, professional, or legal context
- Using the signature comparison tool to forge or replicate another person's signature
- Using the generation feature to produce fraudulent documents, forms, or records
- Deploying this system in any public-facing production environment without appropriate authentication, rate limiting, and human oversight

---

## Legal Framework (India)

Relevant Indian legislation that applies to the misuse of document forgery tools:

- **Indian Penal Code (IPC) § 463–477A** — Forgery and falsification of documents; punishable by up to 7 years imprisonment
- **Information Technology Act 2000, § 66C** — Identity theft; punishable by up to 3 years imprisonment and fine up to ₹1,00,000
- **IT Act § 66D** — Cheating by personation using computer resources; punishable by up to 3 years and fine up to ₹1,00,000
- **Negotiable Instruments Act 1881, § 138** — Cheque fraud; relevant where forged signatures appear on financial instruments
- **Indian Evidence Act 1872, § 45** — Expert opinion on handwriting and documents is admissible; outputs from automated tools like ForgeNet-X must be supported by a qualified examiner's testimony

---

## Limitations & Disclaimers

- All forensic results are **probabilistic**, not definitive. They must be reviewed by a qualified forensic document examiner before any legal, financial, or investigative action is taken.
- The SSIM, Hu Moment, histogram, and pixel metrics used in signature analysis are computer-vision measures of visual similarity, not legal proof of authenticity or forgery.
- The origin analysis classifier uses heuristic rules and is not a trained machine learning model. It can produce false positives (flagging real images as synthetic) and false negatives (failing to detect some digital outputs).
- The system operates entirely offline and does not transmit any images or results to external servers.

---

## Contact & Reporting

If you believe this tool is being used in a manner inconsistent with this ethics statement, raise an issue in the project repository. Suspected criminal misuse should be reported to the appropriate law enforcement authority.

---

*ForgeNet-X — Final Year AI/ML Project. For academic and research use only.*
