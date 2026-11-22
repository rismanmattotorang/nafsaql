# The project website: benchmark, patterns, constraints, use cases (Phase 37)

Research of 2026-11-22 for the Nafs-Aql site. Pages were fetched as HTML and read; nothing was rendered in a browser,
so statements about mobile layout and contrast on other sites are inferred from markup.

## 1. The benchmark: vllm-sr.ai

Docusaurus 3.10 on Netlify, custom home page. Navigation: Docs, Research, Blog, Community, a hosted playground, a
"More" menu, versions, two languages, search. Home page, in order: hero ("Intelligence beyond any one model."), a
one-line install with "For humans / For agents" tabs, two architecture flow diagrams, where it fits (two cards), videos,
a sovereignty flow, 19 paper cards (46 % of the page's words), industry quotes, team marquee, news, partner logos,
closing call to action. Docs: 14 sidebar groups, up to three levels; capability pages follow one template; the
configuration reference rests on one JSON schema.

**Worth copying**
- A runnable command directly under the hero, with a copy button.
- The architecture drawn as inline diagrams with real names, not a decorative picture.
- Short parallel headlines with a one-word eyebrow; one rhythm down the page.
- "What it is not" and "who does what" sections; a fixed template per capability page; "next" links everywhere.
- Research listed with year and venue; a citation block.

**Worth beating**
- The hero never says what the product is; the reader needs the diagram two sections later.
- No evidence on the page: the headline figures live inside paper titles; the benchmarking page shows no results,
  hardware or datasets; no comparison with alternatives.
- The research list is nearly half the page and pushes everything else down.
- Quotes are general remarks about the field, not about the product.
- Many invented names before the reader has run anything.
- Time to first success: Docker, about 5 GB of images, 12 GiB of memory, a dashboard account and an external model;
  success is a set of response headers.
- Weight: 878 KB of JavaScript and 471 KB of CSS uncompressed; 179 images; third-party fonts and a measurement script.
- 115 of 179 images have empty alternative text, including logos that carry meaning; 19 animations against two
  reduced-motion rules; a stock 404 page.

## 2. Patterns of credible developer sites

Recurring order: hero (one sentence, one command) → proof strip → how it works → feature groups each with a real
example → evidence with method → trust (privacy, licence) → documentation entry points → closing action → footer.

- A tagline that states category, difference and implementation in one sentence (uv).
- Every number has a workload, a date and a link to the method (uv's BENCHMARKS.md; Qdrant names hardware, datasets,
  publishes raw results, and says "are we biased? Probably, yes").
- A two-command quick start that runs something real (llama.cpp).
- Real product screens with numbered captions, not illustration (Linear).
- Privacy as a plain sentence beside the feature (LM Studio).
- A limit stated before the reader finds it (Supabase).
- Machine-readable entry points (`llms.txt`) for agents (Turso, Qdrant).
- To avoid: unsourced round numbers, logo walls without context, benchmark pictures without method.

## 3. GitHub Pages

- A project site is served at `https://<owner>.github.io/<repository>/`. Branch publishing allows only `/` or `/docs`
  and runs Jekyll; `docs/` here already holds the Markdown documentation, so the site lives in `/site` and is published
  by a workflow (`actions/configure-pages`, `actions/upload-pages-artifact` with `path: site`, `actions/deploy-pages`;
  permissions `pages: write`, `id-token: write`; environment `github-pages`). One manual step: Settings → Pages →
  Source = GitHub Actions.
- Every URL must be relative (`./css/site.css`), including inside CSS and scripts: a root-absolute URL resolves to the
  owner's domain root and fails. `404.html` is served at any depth, so it alone needs links that carry the repository
  prefix. File names are case sensitive.
- Limits: 1 GB site, 100 GB a month, a 10-minute deployment. GitHub logs visitors' IP addresses: the site says so,
  and loads nothing from a third party.

## 4. Use cases

What a 2 to 4 billion parameter model is adequate for: answers grounded in the person's own documents, short drafting,
summaries of a few pages, extraction, and code or maths that a verifier can check. What it is **not** adequate for
without a person's review: open-ended legal or medical reasoning, synthesis across many long documents, large changes
across many files, numerical reasoning with no verifier, low-resource languages. Local processing helps with a duty of
confidentiality; it is not, by itself, compliance with any regulation.

| # | Who and what | Why local, CPU, private | What serves it | Evidence | Limit to state |
|---|---|---|---|---|---|
| 1 | Small law firm: find and cite clauses, dates and facts in a matter file | Confidentiality; ABA Formal Opinion 512 (2024); GDPR | Knowledge base, cited answers, a room per matter, memory that can be deleted | Over 80 % of firms name confidentiality and accuracy as top concerns (ILTA 2025, via Law360) | Not legal research: commercial legal tools were wrong on more than 17 % of queries (Stanford, 2025) |
| 2 | Healthcare administration: policies, payer rules, procedures | HIPAA; GDPR Art. 9 | Document answers with citations; "not sure" without a source | Average breach cost in healthcare USD 7.42 million (IBM, 2025) | Not diagnosis, triage or treatment |
| 3 | Finance and internal audit: check a calculation, query the control library | Bank secrecy, SOX, DORA | Exact arithmetic check; cited policy lookup | Nearly half of privacy professionals admit entering non-public data into public tools (Cisco, 2025) | Not judgement on accounting treatment, not advice |
| 4 | Tax or accounting practice: questions from client folders | IRC §7216; FTC Safeguards Rule | Rooms per client; cited answers | 60 % of tax professionals use AI weekly (Blue J / CPA.com 2026, trade-press figure) | Works only on the documents loaded |
| 5 | Public office: staff questions from regulations and circulars | Sovereignty and residency rules; EU AI Act Art. 4 | Knowledge base and graph; no cloud contract | Skills and data access are the most widespread obstacles (OECD, 2025) | Not decisions about citizens |
| 6 | School or university: maths and programming practice with checked answers | Student data (FERPA, GDPR); no subscription | Sandboxed code and maths checks; course material | 92 % of UK undergraduates use AI; 36 % had institutional support (HEPI, 2025) | Prose explanations are unchecked |
| 7 | Manufacturing and field service: manuals and procedures at the machine | Isolated plant networks; no signal | Offline knowledge base with page references | Up to 3.8 million manufacturing workers needed by 2033 in the US (Deloitte / MI, 2024) | Safety steps are read from the cited source; text only |
| 8 | Software team: explain private code, write and run small functions | Source code is IP; no per-token cost | Code agent with sandboxed tests; rooms | 46 % of developers do not trust AI output (Stack Overflow, 2025) | Not large changes across files |
| 9 | Research lab: unpublished manuscripts and a paper corpus | Embargoes; NIH NOT-OD-23-149 | Graph over a corpus; cited answers | 69 % of researchers use AI tools (Springer Nature, 2025) | Not novel scientific judgement |
| 10 | Small business: contracts, price lists, the handbook | No budget for GPUs or subscriptions | Runs on an existing PC | 17 % of small EU enterprises used AI in 2025 against 55 % of large ones (Eurostat) | Slow for long outputs |
| 11 | Air-gapped environment | No outbound network at all | No GPU, no key, no network after installation | 91 % would choose on-premises or private infrastructure for sensitive data (Cloudian 2026, vendor survey) | Updates need a manual transfer; no certification is claimed |
| 12 | Organisation under a data-residency law | Local-storage and transfer rules | Nothing leaves the machine | Close to 100 localisation measures in 40 countries (OECD, 2023) | Lawful basis and retention are still the organisation's |
| 13 | Low-connectivity setting | No reliable internet | Fully offline on a laptop | 2.2 billion people offline (ITU, 2025) | Weaker in low-resource languages |
| 14 | HR: staff questions from the handbook | Employee data | Policy answers with citations | 46 % had entered employee information into public tools (Cisco 2025, via press) | Not hiring or dismissal decisions |
| 15 | Journalist: connect names and dates across sensitive documents | Source protection | Graph, search, cited passages, offline | Over 80 % of surveyed journalists in the Global South use AI; 13 % have a policy (TRF, 2025) | A lead-finding aid, not evidence |
| 16 | Compliance officer: an approved alternative to unapproved chatbots | "Shadow AI" exposure | Local deployment, inspectable memory | One in five organisations reported a breach linked to shadow AI (IBM, 2025) | Staff will still want larger models for hard tasks |

Several figures come through press summaries of the primary report (rows 1, 4, 14); the site links the primary source
where one could be opened and words the rest as reported.
