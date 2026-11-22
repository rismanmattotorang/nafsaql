#!/usr/bin/env python3
"""Check the static website in site/ before it is published.

For every HTML page: each relative link and asset resolves to a file in site/ (and each #fragment to an id on the
page it points at); no URL starts with "/" (the site is served under /nafsaql/); nothing is loaded from another
host by a script, a stylesheet or an image; every image has alt, width and height; there is exactly one <h1>, a
<title>, a description and lang; and no page is left without the shared navigation.

Usage: python3 scripts/site_check.py            (exit code 1 when something is wrong)
"""
import os, sys
from html.parser import HTMLParser

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.assets, self.problems = set(), [], [], []
        self.h1 = 0
        self.title = self.desc = self.lang = self.nav = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "html":
            self.lang = bool(a.get("lang"))
        if tag == "h1":
            self.h1 += 1
        if tag == "title":
            self.title = True
        if tag == "nav":
            self.nav = True
        if tag == "meta" and a.get("name") == "description" and a.get("content"):
            self.desc = True
        if tag == "a" and "href" in a:
            self.links.append(a["href"])
        if tag == "img":
            self.assets.append(("img", a.get("src", "")))
            for need in ("alt", "width", "height"):
                if need not in a:
                    self.problems.append(f"<img src={a.get('src')!r}> has no {need}")
        if tag == "script" and a.get("src"):
            self.assets.append(("script", a["src"]))
        if tag == "link" and a.get("href"):
            self.assets.append(("link " + (a.get("rel") or ""), a["href"]))
        if tag in ("source", "video", "audio", "iframe") and a.get("src"):
            self.assets.append((tag, a["src"]))
        if tag == "meta" and a.get("property") in ("og:image", "og:url") and a.get("content", "").startswith("/"):
            self.problems.append(f"{a['property']} starts with /")


def main():
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(__doc__)
        return 0
    pages = {}
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            if f.endswith(".html"):
                p = Page()
                p.feed(open(os.path.join(dirpath, f), encoding="utf-8").read())
                pages[os.path.relpath(os.path.join(dirpath, f), ROOT)] = p
    bad = 0
    for name, p in sorted(pages.items()):
        here = os.path.dirname(name)
        problems = list(p.problems)
        if p.h1 != 1:
            problems.append(f"{p.h1} <h1> elements")
        for ok, what in ((p.title, "<title>"), (p.desc, "meta description"), (p.lang, "lang"), (p.nav, "<nav>")):
            if not ok:
                problems.append(f"no {what}")
        for kind, url in p.assets:
            if url.startswith(("http://", "https://", "//")):
                problems.append(f"{kind} loads from another host: {url}")
            elif url.startswith("/"):
                problems.append(f"{kind} URL starts with /: {url}")
            elif not url.startswith("data:") and not os.path.isfile(os.path.join(ROOT, here, url.split("#")[0].split("?")[0])):
                problems.append(f"{kind} not found: {url}")
        for url in p.links:
            if url.startswith(("http://", "https://", "mailto:")):
                continue
            if url.startswith("/"):
                problems.append(f"link starts with /: {url}")
                continue
            path, _, frag = url.partition("#")
            target = os.path.normpath(os.path.join(here, path)) if path else name
            if path and not os.path.isfile(os.path.join(ROOT, target)):
                problems.append(f"link not found: {url}")
            elif frag and target in pages and frag not in pages[target].ids:
                problems.append(f"no id #{frag} on {target} (from {url})")
        print(("ok   " if not problems else "FAIL ") + name + (f"  ({len(p.links)} links, {len(p.assets)} assets)" if not problems else ""))
        for x in problems:
            print("     " + x)
        bad += len(problems)
    print(f"{len(pages)} pages, {bad} problems")
    return 1 if bad or not pages else 0


if __name__ == "__main__":
    sys.exit(main())
