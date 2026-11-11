#!/usr/bin/env python3
"""Derive scripts/kb_p26_typo.json from scripts/kb_p25_quoted.json.

For each question the longest purely alphabetic word of at least 7 letters is
corrupted with one edit (ties: first occurrence; a word that occurs inside an
accepted phrase is used only when there is no other candidate). The edit is
chosen by `index % 3`: 0 swaps two adjacent inner letters, 1 deletes one inner
letter, 2 replaces one inner letter with the next letter of the alphabet. The
first and last letters are never touched. The position comes from a generator
with a fixed seed, so the output is the same on every run. A question with no
candidate word is copied unchanged and counted.

Usage: python3 scripts/kb_p26_make_typo.py [source.json] [target.json]
"""
import json
import os
import random
import re
import sys

SEED = 26
MIN_LEN = 7
HERE = os.path.dirname(os.path.abspath(__file__))
WORD = re.compile(r"[A-Za-z]+")


def candidates(question):
    """(start, end, word) of every purely alphabetic token of MIN_LEN letters or more."""
    found = []
    for m in re.finditer(r"\S+", question):
        token = m.group()
        core = token.strip("\"'`()[]{}.,;:?!")
        if len(core) >= MIN_LEN and WORD.fullmatch(core):
            start = m.start() + token.index(core)
            found.append((start, start + len(core), core))
    return found


def pick(question, phrases):
    found = candidates(question)
    if not found:
        return None
    joined = " ".join(phrases).lower()
    outside = [c for c in found if c[2].lower() not in joined]
    pool = outside or found
    return max(pool, key=lambda c: (len(c[2]), -c[0]))


def corrupt(word, kind, rng):
    inner = range(1, len(word) - 1)
    if kind == 0:
        spots = [i for i in inner if i + 1 < len(word) - 1 and word[i] != word[i + 1]]
        if spots:
            i = rng.choice(spots)
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        kind = 1
    i = rng.choice(list(inner))
    if kind == 1:
        return word[:i] + word[i + 1:]
    c = word[i]
    base = ord("a") if c.islower() else ord("A")
    return word[:i] + chr(base + (ord(c) - base + 1) % 26) + word[i + 1:]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kb_p25_quoted.json")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "kb_p26_typo.json")
    with open(src, encoding="utf-8") as f:
        items = json.load(f)
    rng = random.Random(SEED)
    out, corrupted, unchanged = [], 0, 0
    for index, (question, phrases) in enumerate(items):
        chosen = pick(question, phrases)
        if chosen is None:
            unchanged += 1
            out.append([question, phrases])
            continue
        start, end, word = chosen
        out.append([question[:start] + corrupt(word, index % 3, rng) + question[end:], phrases])
        corrupted += 1
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{corrupted} corrupted, {unchanged} copied unchanged (no candidate word), {len(out)} written to {dst}")


if __name__ == "__main__":
    main()
