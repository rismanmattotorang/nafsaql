#!/usr/bin/env python3
"""Check the Phase 25 retrieval question sets against a snapshot of the knowledge base.

    python3 scripts/kb_p25_check.py SNAPSHOT.json [-v]

SNAPSHOT.json is a list of {chunk_id, doc_id, doc_title, ord, heading, text}, one entry per
passage. The three sets are read from the directory of this script:

    kb_p25_quoted.json    [question, [phrases]]                       exactly 60
    kb_p25_para.json      [question, [phrases]]                       at least 100
    kb_p25_multihop.json  [question, [phrases A], [phrases B]]        at least 40

A passage matches a phrase list when its normalised text contains any normalised phrase.

Sets 1 and 2 fail on a question with 0 or more than 3 matching passages, or with a phrase that
matches nothing. Set 3 fails unless both lists match (1 to 3 passages each) and no passage
matches both. For the reworded set the share of the question's content words (4+ letters, not a
function word) found in the answering passage is printed; a question above 0.5 is flagged and
the set fails when more than 10 % are flagged. Questions must be unique across the three sets and
the four older sets (kb_paper_questions*.json). Every document needs 2 questions in sets 1 and 2;
a question is attributed to the smallest document that has a matching passage.

Standard library only; Python 3.8.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = [
    ("quoted", "kb_p25_quoted.json"),
    ("para", "kb_p25_para.json"),
    ("multihop", "kb_p25_multihop.json"),
]
OLDER = [
    "kb_paper_questions.json",
    "kb_paper_questions_2.json",
    "kb_paper_questions_50.json",
    "kb_paper_questions_para.json",
]
MAX_MATCH = 3
MAX_OVERLAP = 0.5
MAX_FLAGGED_SHARE = 0.10
MIN_PER_DOC = 2

FUNCTION_WORDS = set("""
about above after again against also among another around because been before being below
between both cannot could does doing done down during each either else ever every from further
given have having here hers into itself just least less many more most much must near need
neither never none once only onto other others ought ours over rather same shall should since
some such than that their theirs them then there these they this those through thus under until
upon very were what when whence where whether which while whom whose will with within without
would your yours according instead still
""".split())


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%/-]", " ", s.lower())).strip()


def words(s):
    return re.findall(r"[a-z]+", s.lower())


def content_words(q):
    out = []
    for w in words(q):
        if len(w) >= 4 and w not in FUNCTION_WORDS and w not in out:
            out.append(w)
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv[1:]
    if len(args) != 1:
        sys.stderr.write(__doc__)
        return 2
    passages = load(args[0])
    texts = [norm(p["text"]) for p in passages]
    wordsets = [set(words(p["text"])) for p in passages]
    doc_of = [p["doc_id"] for p in passages]
    doc_size = {}
    doc_title = {}
    doc_order = []
    for p in passages:
        if p["doc_id"] not in doc_size:
            doc_order.append(p["doc_id"])
        doc_size[p["doc_id"]] = doc_size.get(p["doc_id"], 0) + 1
        doc_title[p["doc_id"]] = p["doc_title"]

    def match(phrases):
        """(passages matching any phrase, phrases that match nothing)."""
        hit, dead = set(), []
        for ph in phrases:
            n = norm(ph)
            got = [i for i, t in enumerate(texts) if n and n in t]
            if not got:
                dead.append(ph)
            hit.update(got)
        return sorted(hit), dead

    def attribute(hit):
        return min((doc_of[i] for i in hit), key=lambda d: (doc_size[d], doc_order.index(d)))

    seen = {}
    for name in OLDER:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            for row in load(path):
                seen.setdefault(norm(row[0]), name)

    ok = True
    summary = []
    for label, name in SETS:
        rows = load(os.path.join(HERE, name))
        fails, flagged, overlaps = [], [], []
        per_doc = dict((d, 0) for d in doc_order)
        print("== %s (%s): %d questions" % (label, name, len(rows)))
        for k, row in enumerate(rows):
            want = 3 if label == "multihop" else 2
            shape = (isinstance(row, list) and len(row) == want and isinstance(row[0], str)
                     and all(isinstance(l, list) and l and all(isinstance(x, str) for x in l)
                             for l in row[1:]))
            if not shape:
                fails.append((k, "bad shape", row))
                continue
            q = row[0]
            key = norm(q)
            if key in seen:
                fails.append((k, "duplicate question (also in %s)" % seen[key], q))
            seen.setdefault(key, name)
            problems = []
            hits = []
            for phrases in row[1:]:
                hit, dead = match(phrases)
                hits.append(hit)
                if dead:
                    problems.append("phrase matches nothing: %r" % dead)
                if not hit:
                    problems.append("0 passages for %r" % phrases)
                elif len(hit) > MAX_MATCH:
                    problems.append("%d passages (> %d) for %r" % (len(hit), MAX_MATCH, phrases))
            line = "%3d  %s" % (k + 1, "  ".join("n=%d %s" % (len(h), h) for h in hits))
            if label == "multihop":
                both = sorted(set(hits[0]) & set(hits[1]))
                if both:
                    problems.append("passages %s match both lists" % both)
                elif hits[0] and hits[1] and not any(a != b for a in hits[0] for b in hits[1]):
                    problems.append("no two different passages")
            if label == "para" and hits[0]:
                cw = content_words(q)
                ov = max(sum(1 for w in cw if w in wordsets[i]) / float(len(cw) or 1)
                         for i in hits[0])
                overlaps.append(ov)
                line += "  overlap=%.2f" % ov
                if ov > MAX_OVERLAP:
                    flagged.append(k)
                    line += "  FLAGGED"
            for h in hits:
                if h:
                    per_doc[attribute(h)] += 1 if label != "multihop" else 0.5
            if problems:
                fails.append((k, "; ".join(problems), q))
                line += "  FAIL"
            if verbose or problems or label == "para":
                print(line + "  | " + q)
        for k, why, q in fails:
            print("  FAIL #%d: %s\n        %s" % (k + 1, why, q))
        size_ok = (len(rows) == 60 if label == "quoted" else
                   len(rows) >= (100 if label == "para" else 40))
        if not size_ok:
            print("  FAIL: wrong number of questions (%d)" % len(rows))
        doc_ok = True
        print("  per document (smallest matching document; a multihop list counts 0.5):")
        for d in doc_order:
            short = label != "multihop" and per_doc[d] < MIN_PER_DOC
            doc_ok = doc_ok and not short
            print("    %5.1f  of %3d passages  %s%s" % (per_doc[d], doc_size[d], doc_title[d][:60],
                                                     "   FAIL (< %d)" % MIN_PER_DOC if short else ""))
        flag_ok = True
        extra = ""
        if label == "para":
            share = len(flagged) / float(len(rows) or 1)
            flag_ok = share <= MAX_FLAGGED_SHARE
            extra = ", %d flagged (%.1f %%, limit %.0f %%), mean overlap %.2f" % (
                len(flagged), 100 * share, 100 * MAX_FLAGGED_SHARE,
                sum(overlaps) / float(len(overlaps) or 1))
        good = not fails and size_ok and doc_ok and flag_ok
        ok = ok and good
        summary.append("%-8s %3d questions, %d failing%s: %s" % (
            label, len(rows), len(fails), extra, "PASS" if good else "FAIL"))
    print("== summary (%d passages, %d documents)" % (len(passages), len(doc_order)))
    for s in summary:
        print(s)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
