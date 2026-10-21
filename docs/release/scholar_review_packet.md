# Scholar review packet (SR-005, gate G6)

**Purpose.** SR-005 requires that user-facing copy describe the *nafs* and *ʿaql* terms as analogies, claim no religious authority, and be reviewed by a scholar of the tradition before release. This packet collects every place the vocabulary appears, the sources the team relied on, and the questions we ask the reviewer. Review is blocking for the analogy text only; the engineering behaviour does not depend on it.

## 1. What the reviewer is asked to judge

1. Is each term used in a way that is accurate as a description of the classical usage, and clearly marked as an analogy?
2. Does any wording imply religious authority, a ruling, or a claim about the soul that the project is not entitled to make?
3. Is anything disrespectful, trivialising or likely to offend? Especially consider:
   - mapping a machine's first draft to *al-nafs al-ammāra*;
   - mapping its abstention to *tawaqquf*.
4. Are the citations to the Qur'an (12:53, 75:2, 89:27, 91:9) and to al-Ghazālī (*Iḥyāʾ*, Book 21) correctly attributed and rendered?
5. Should any term be removed, softened or replaced?

## 2. All user-facing analogy text

### 2.1 Chat interface: the answer's journey (`rust/crates/nafs-web/assets/app.js`, `STATIONS`)

| State | English label | Term shown | Tooltip text |
|---|---|---|---|
| A | Draft | nafs | System 1 answers at once: the first, unexamined response. |
| L | Flagged | lawwāma | Self-reproach (Q 75:2): the gate or a verifier finds the draft wanting. |
| M | Deliberate | ʿaql | Reflection: System 2 reasons within a budget, without seeing the draft. |
| T | Accepted | muṭmaʾinna | At rest (Q 89:27): the answer is accepted with a stated assurance. |
| X | Abstain | tawaqquf | Judgement withheld: no candidate earned acceptance within the budget. |

### 2.2 Design-notes panel (opened from the logo)

> Nafs-Aql borrows the classical vocabulary of the self as a **design analogy**, not as a theological claim. The engineering names are primary; scholar review is a release gate (SR-005).

| Term | Source cited | In Nafs-Aql |
|---|---|---|
| *nafs* (self; first impulse) | Q 12:53 (al-nafs al-ammāra) | System 1: the fast draft |
| *lawwāma* (self-reproaching) | Q 75:2 | State L: the draft is flagged by the gate or a verifier |
| *ʿaql* (intellect, restraint) | al-Ghazālī, *Iḥyāʾ* Book 21: the counsellor of the heart | The Aql Gate and System 2 deliberation |
| *muṭmaʾinna* (at rest) | Q 89:27 | State T: accepted with an assurance level |
| *tawaqquf* (withholding judgement) | usage in jurisprudence and theology | State X: abstention with a notice |
| *muḥāsaba* (self-accounting) | Sufi practice of examining the self | The ledger, audits and capability table |
| *tazkiya* (purification) | Q 91:9 | Distillation: verified System 2 traces improve System 1 |

> Rūḥ, ilhām and the Active Intellect are deliberately not mapped.

### 2.3 Other UI copy

- **Welcome screen:** the A → L → M → T / X strip with the same terms, and the line "the vocabulary of the self is used as an analogy".
- **Lab presets:** "When should ʿaql restrain the first impulse?"; "Does reflection repeat the impulse?"; "How long should reflection last?".
- **Lab trial column headings:** "System 1 · nafs" and "System 2 · ʿaql".
- **Page names:** *Muhasaba* (gate and ledger) and *Tazkiya* (distillation), with the tagline "verified reflection teaches the first impulse".
- **User guide:** `docs/user_guide.md`, section "The design vocabulary".

### 2.4 Documents

- **Technical report:** `docs/report/sections/background.tex`, subsections on the *nafs* and its Qur'anic stages and on al-Ghazālī's account of *qalb*, *rūḥ*, *nafs* and *ʿaql*.
- **Slides:** `docs/slides/nafs_aql_slides.tex`, the analogy slide.
- **Research notes:** `docs/research/islamic_sources.md`, with verified sources and a verification log (375 lines).

## 3. Sources the team relied on

The sources are listed in `docs/research/islamic_sources.md`, where each was checked against a page confirming author, title, year and publisher. The main ones are:

- Abdel Haleem's translation of the Qur'an (2008);
- the *Encyclopaedia of Islam* article "Nafs" (Calverley and Netton);
- Schimmel, *Mystical Dimensions of Islam* (1975);
- al-Ghazālī, *Iḥyāʾ ʿulūm al-dīn*, Book 21 (Skellie translation, 2010);
- Rothman and Coyle (2018) on a model of Islamic psychology.

## 4. Sign-off

The reviewer's decision is recorded in `docs/release/scholar_review.json`, which gate G6 reads:

```json
{"reviewer": "Name, affiliation", "date": "YYYY-MM-DD", "decision": "approved | changes_requested",
 "scope": "UI copy (2.1-2.3), user guide, report background section, slides",
 "notes": "free text: requested changes, or conditions of approval"}
```

Until that file exists with `"decision": "approved"`, gate G6 shows SR-005 as pending and the release is blocked for the analogy text. If changes are requested:

1. Edit the strings above; they are display-only.
2. Update `docs/user_guide.md` and the report.
3. Ask for a re-review.

## 5. Methodology, for the reviewer's context

Nafs-Aql is a dual-process assistant. **System 1** (Qwen3-1.7B) answers at once; the **Aql Gate** decides whether the expected error saved by deliberation outweighs its latency; **System 2** (Qwen3-4B-Thinking-2507) reasons within a token budget when it does. Exact verifiers (a numeric and symbolic checker for arithmetic, unit tests for code, JSON Schema for structured output) decide correctness where they exist; a rubric judge on System 2 scores the rest. The vocabulary in section 2 names the **states of an answer's journey**; nothing in the engineering depends on the names. The analogy is the only thing under review here.

## 6. Evaluation design (what the numbers in the release candidate mean)

- **Benchmarks:** GSM8K (arithmetic word problems), HumanEval+ and MBPP+ (Python functions with hidden tests), twelve JSON Schema tasks, and open or factual prompts. Item sets are fixed and seeded (`scripts/build_item_set.py`); `regression_500` is a held-out fixed suite.
- **Systems compared on the same items:** `fast_only` (System 1 alone), `slow_only` (System 2 alone), `nafs_aql` (the gated system with verifiers), `nafs_blind` (the gate routing without seeing a verifier, which measures the gate itself), `slow_conditioned` (System 2 shown the draft, a coupling ablation).
- **Statistics:** paired comparisons on identical items; exact McNemar tests with Bonferroni correction; bootstrap confidence intervals; Clopper-Pearson bounds on rates. A gate passes only when its stored evidence meets the roadmap's threshold at the required sample size; smaller runs are reported as *scaled*, with n and an interval.
- **Hardware:** one 4-core CPU with no GPU, shared with other work; every timing is labelled with the load average.

## 7. Sample outputs by category, including failures

From stored run `5e87f534` (smoke_mixed, 2026-10-18, shared machine). Answers are shortened. "Correct" is the exact verifier's verdict where one exists.

| Category | Question | System 1 draft | Outcome |
|---|---|---|---|
| Arithmetic | A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total? | "1 bolt of blue ... Total = 1.5 bolts. Answer: 1.5" | **Wrong draft** (verifier: 3). The gated system escalated; System 2 answered 3, verified. State sequence A, L, M, T. |
| Arithmetic | James runs 3 sprints 3 times a week, 60 m each. Total meters per week? | "He runs 3 sprints each week ... 180 meters" | **Wrong draft** (verifier: 540). Escalated; System 2 answered 540, verified. |
| Arithmetic | Janet's ducks lay 16 eggs per day ... how much does she make daily? | step-by-step, answer 18 | Correct, verified on the draft; no deliberation (state A, T). |
| Code | Remove the first and last occurrence of a character from a string. | a function using `index` and `rindex` | Correct on the hidden tests; no deliberation. |
| Structured | A JSON object for a library book (title, author, year 1450-2026, available). | `{"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925, "available": true}` | Valid against the schema; verified. |
| Factual | What is the capital of Indonesia, and on which island? | "Jakarta ... on the island of Java." | No exact verifier; two cheap samples agreed, so the answer is shown as *consensus* (not *verified*). |
| Factual | Who wrote *The Incoherence of the Philosophers*? | "... was written by Michel de Montaigne." | **Wrong draft, no verifier.** The gated system escalated on low confidence; System 2 answered al-Ghazālī (correct) but invented an Arabic title. Shown as *unverified*. This is the kind of error the abstention state (X, *tawaqquf*) exists for and did not catch here. |

The last row is the honest case for the reviewer: a confident-sounding wrong draft about an Islamic source, corrected in substance by deliberation but still carrying a fabricated detail. The interface marks such answers *unverified*; it never marks them *verified*.

## 8. Review checklist

- [ ] Each term in 2.1 and 2.2 is accurate as a description of the classical usage and is marked as an analogy.
- [ ] No wording implies religious authority, a ruling, or a claim about the soul.
- [ ] The mapping of a machine's first draft to *al-nafs al-ammāra* and of abstention to *tawaqquf* is acceptable, or a replacement is named.
- [ ] Qur'an citations (12:53, 75:2, 89:27, 91:9) and the al-Ghazālī reference (*Iḥyāʾ*, Book 21) are correctly attributed and rendered.
- [ ] The failure example in section 7 (a fabricated Arabic title shown as *unverified*) is presented acceptably.
- [ ] Decision recorded in `docs/release/scholar_review.json` (section 4).

Sign-off status on 2026-10-21: **pending; no reviewer assigned.** The team does not sign for the reviewer.
