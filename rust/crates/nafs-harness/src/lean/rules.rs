// Derived from Open Math Harness (interfaces/lean/src/rules.rs, revision 687f2a2), Apache-2.0.
// Modified for Nafs-Aql: module path of MAX_FROZEN_BYTES only.
//! The protocol's textual rules. They decide, without running Lean at all,
//! whether a candidate can be graded (the forbidden tokens, the frozen
//! statement verbatim, the frozen preamble line for line) and whether a
//! frozen file can be registered. They live in the interface because two
//! providers need the same answer: the `lean` service before it elaborates
//! anything, and the discovery harness before it accepts a submission it
//! would otherwise have to record, or a statement a task supplies.

use std::ops::Range;

use super::MAX_FROZEN_BYTES;

/// The frozen statement: from the first line of code that *starts* with
/// `theorem` to the last `:=` of code (the proof assignment), inclusive.
/// Both are found on [`lean_code`], so neither a comment nor a string can
/// move them: a comment line that starts with `theorem` would otherwise end
/// the preamble every candidate must keep before the definitions it holds,
/// and a candidate could define them anew. `None` when the file has no such
/// statement or [`lean_code`] refuses it.
/// Deterministic; the candidate must contain it verbatim.
pub fn extract_statement(frozen: &str) -> Option<&str> {
    frozen.get(statement_span(frozen)?)
}

/// Where [`extract_statement`]'s statement lies in `frozen`; [`lean_code`]
/// keeps every byte where it was.
fn statement_span(frozen: &str) -> Option<Range<usize>> {
    let code = lean_code(frozen).ok()?;
    let start = code
        .match_indices("theorem")
        .map(|(i, _)| i)
        .find(|&i| i == 0 || code.as_bytes().get(i - 1) == Some(&b'\n'))?;
    let end = code.rfind(":=")?;
    (end >= start).then_some(start..end + 2)
}

/// `source` with its comments (`--` to the end of the line, nested `/- … -/`,
/// doc comments included) and string literals blanked to spaces, so nothing
/// moves, or `Err` naming a construct this lexer might read differently from
/// Lean. A textual check must not read a comment ("-- no sorry left" is not an
/// unfinished proof), but it must never blank text Lean executes either, so
/// every place where the two lexers could disagree is refused instead of
/// guessed: a string opened right after an identifier character, `'`, `\` or
/// `#` (raw, interpolated or char-literal strings), a string containing `\`
/// or `{` (escapes, interpolation), an unterminated string, and `'`/`\`
/// right before `«`, `--` or `/-` (a character literal). A text that ends
/// inside a `/-` comment or a `«` name is refused too, rather than read as
/// if it closed at the end. Legitimate proofs need none of these. Escaped
/// identifiers `«…»` are kept visible, so a banned name cannot hide in them.
pub fn lean_code(source: &str) -> Result<String, &'static str> {
    blanked(source, true).map(|(code, _)| code)
}

/// `source` with its comments blanked as [`lean_code`] blanks them, string
/// literals blanked too when `strings`, and the byte span of every string
/// literal, quotes included, in order.
fn blanked(source: &str, strings: bool) -> Result<(String, Vec<Range<usize>>), &'static str> {
    const AMBIGUOUS: &str = "string or character literal the checker cannot read unambiguously";
    const OPEN_COMMENT: &str = "comment opened with /- and never closed with -/";
    const OPEN_NAME: &str = "name opened with « and never closed with »";
    let chars: Vec<(usize, char)> = source.char_indices().collect();
    let mut out = source.to_owned().into_bytes();
    let mut spans = vec![];
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for b in &mut out[from..to] {
            if *b != b'\n' {
                *b = b' ';
            }
        }
    };
    let at = |k: usize| chars.get(k).map(|&(_, c)| c);
    let byte = |k: usize| chars.get(k).map_or(source.len(), |&(i, _)| i);
    let mut k = 0;
    while k < chars.len() {
        let c = chars[k].1;
        let prev = k.checked_sub(1).and_then(at);
        let quote_or_escape = matches!(prev, Some('\'' | '\\'));
        let opens_comment =
            (c == '-' && at(k + 1) == Some('-')) || (c == '/' && at(k + 1) == Some('-'));
        if (opens_comment || c == '«' || c == '"') && quote_or_escape {
            return Err(AMBIGUOUS);
        }
        if c == '-' && at(k + 1) == Some('-') {
            let start = k;
            while k < chars.len() && chars[k].1 != '\n' {
                k += 1;
            }
            blank(&mut out, byte(start), byte(k));
        } else if c == '/' && at(k + 1) == Some('-') {
            let (start, mut depth) = (k, 1usize);
            // Lean reads `/--` and `/-!` as one token, so `/--/` opens a doc
            // comment rather than closing an empty one.
            k += if matches!(at(k + 2), Some('-' | '!')) {
                3
            } else {
                2
            };
            while k < chars.len() && depth > 0 {
                if chars[k].1 == '-' && at(k + 1) == Some('/') {
                    depth -= 1;
                    k += 2;
                } else if chars[k].1 == '/' && at(k + 1) == Some('-') {
                    depth += 1;
                    k += 2;
                } else {
                    k += 1;
                }
            }
            if depth > 0 {
                return Err(OPEN_COMMENT);
            }
            blank(&mut out, byte(start), byte(k));
        } else if c == '"' {
            if prev.is_some_and(|p| is_id_rest(p) || p == '#') {
                return Err(AMBIGUOUS);
            }
            let start = k;
            k += 1;
            loop {
                match at(k) {
                    None | Some('\\' | '{') => return Err(AMBIGUOUS),
                    Some('"') => break,
                    Some(_) => k += 1,
                }
            }
            k += 1;
            spans.push(byte(start)..byte(k));
            if strings {
                blank(&mut out, byte(start), byte(k));
            }
        } else if c == '«' {
            while k < chars.len() && chars[k].1 != '»' {
                k += 1;
            }
            if k == chars.len() {
                return Err(OPEN_NAME);
            }
        } else {
            k += 1;
        }
    }
    // Only ASCII bytes were replaced, and only whole characters, so the text
    // is still valid UTF-8.
    String::from_utf8(out)
        .map(|code| (code, spans))
        .map_err(|_| AMBIGUOUS)
}

/// Lean's `isIdRest`: the characters that continue an identifier.
fn is_id_rest(c: char) -> bool {
    let v = u32::from(c);
    let letter_like = ((0x3b1..=0x3c9).contains(&v) && v != 0x3bb)
        || ((0x391..=0x3a9).contains(&v) && v != 0x3a0 && v != 0x3a3)
        || (0x3ca..=0x3fb).contains(&v)
        || (0x1f00..=0x1ffe).contains(&v)
        || (0x2100..=0x214f).contains(&v)
        || (0x1d49c..=0x1d59f).contains(&v);
    let subscript = (0x2080..=0x2089).contains(&v)
        || (0x2090..=0x209c).contains(&v)
        || (0x1d62..=0x1d6a).contains(&v)
        || v == 0x2c7c;
    c.is_ascii_alphanumeric() || matches!(c, '_' | '\'' | '!' | '?') || letter_like || subscript
}

/// Words that make Lean run code, change the syntax, or reach the
/// meta-programming and IO libraries while it elaborates. A proof needs none
/// of them. They are refused because elaborated code can print anything to
/// the output the verdict is read from (the `verify` report, the REPL's reply
/// stream), add tokens that make the text read differently than
/// [`lean_code`] reads it, or replace how the harness's own commands behave.
#[rustfmt::skip]
const EXEC_WORDS: &[&str] = &[
    "#exit", "#guard", "#html", "#widget", "#sample", "#test",
    "run_cmd", "run_elab", "run_meta", "run_tac", "by_elab",
    "syntax", "macro", "macro_rules", "elab", "elab_rules", "declare_syntax_cat",
    "notation", "notation3", "infix", "infixl", "infixr", "prefix", "postfix", "binder_predicate",
    "initialize", "builtin_initialize", "simproc", "dsimproc", "simproc_decl", "dsimproc_decl",
    "command_elab", "term_elab", "tactic", "delab", "app_unexpander", "implemented_by", "extern",
    "unsafe", "Lean", "Mathlib", "Aesop", "Batteries", "Qq", "IO", "EIO", "BaseIO",
    "println!", "eprintln!", "asTask",
];

/// Options a candidate may not set: `pp.*` would change the `pp.all` type the
/// verdict compares, `debug.*` includes skipping the kernel check.
const BANNED_OPTION_ROOTS: &[&str] = &["pp", "debug"];

/// Words of `code` (the output of [`lean_code`]): maximal runs of identifier
/// characters, a leading `#` kept, with the byte offset after each.
fn words(code: &str) -> Vec<(&str, usize)> {
    let mut out = vec![];
    let mut start = None;
    for (i, c) in code
        .char_indices()
        .chain(std::iter::once((code.len(), ' ')))
    {
        match start {
            Some(s) if !is_id_rest(c) => {
                if !(i == s + 1 && code[s..].starts_with('#')) {
                    out.push((&code[s..i], i));
                }
                start = if c == '#' { Some(i) } else { None };
            }
            None if is_id_rest(c) || c == '#' => start = Some(i),
            _ => {}
        }
    }
    out
}

/// Library roots a header may `import` (the workspace is trusted and
/// read-only); anywhere else they are [`EXEC_WORDS`].
const IMPORTABLE: &[&str] = &["Lean", "Mathlib", "Aesop", "Batteries", "Qq"];

/// The first word in `code` that makes Lean run code or alter what the
/// harness reads (see [`EXEC_WORDS`]).
fn exec_word(code: &str) -> Option<&'static str> {
    let words = words(code);
    // Whether the current word is part of the dotted module name of an
    // `import` on the same line.
    let mut in_import = false;
    for (n, &(word, end)) in words.iter().enumerate() {
        let start = end - word.len();
        in_import = n
            .checked_sub(1)
            .and_then(|p| words.get(p))
            .is_some_and(|&(prev, prev_end)| {
                let gap = &code[prev_end..start];
                (prev == "import" && !gap.is_empty() && gap.chars().all(|c| c == ' ' || c == '\t'))
                    || (in_import && gap == ".")
            });
        if in_import && IMPORTABLE.contains(&word) {
            continue;
        }
        if word.starts_with("#eval") {
            return Some("#eval");
        }
        if word == "eval" && code[end..].starts_with('%') {
            return Some("eval%");
        }
        if let Some(banned) = EXEC_WORDS.iter().find(|w| **w == word) {
            return Some(banned);
        }
    }
    None
}

/// Why `source` may not be elaborated by the `lean` service at all, whatever
/// the action: a construct [`lean_code`] cannot read unambiguously, or a word
/// that runs code at elaboration time ([`EXEC_WORDS`], `#eval`, `eval%`).
/// Comments and string contents do not count.
/// Every entry point that sends text to Lean applies this first, because the
/// verdict is read from output that elaborated code could write into.
pub fn exec_token(source: &str) -> Option<&'static str> {
    match lean_code(source) {
        Err(why) => Some(why),
        Ok(code) => exec_word(&code),
    }
}

/// The first forbidden token in `candidate` under the `verify` protocol, if
/// any: `sorry`, or anything [`check_forbidden_token`] refuses. Comments and
/// string contents do not count ([`lean_code`]).
pub fn forbidden_token(candidate: &str) -> Option<&'static str> {
    match lean_code(candidate) {
        Ok(code) if code.contains("sorry") => Some("sorry"),
        _ => check_forbidden_token(candidate),
    }
}

/// The first token forbidden even in a *partial* proof (`check`): `sorry` is
/// allowed there; `native_decide`, a user `axiom` (after any modifier or
/// attribute), `set_option pp.*`/`debug.*` and everything [`exec_token`]
/// refuses never are.
pub fn check_forbidden_token(candidate: &str) -> Option<&'static str> {
    let code = match lean_code(candidate) {
        Ok(code) => code,
        Err(why) => return Some(why),
    };
    if code.contains("native_decide") {
        return Some("native_decide");
    }
    let words = words(&code);
    if words.iter().any(|&(w, _)| w == "axiom") {
        return Some("axiom");
    }
    let option = words
        .windows(2)
        .find(|p| p[0].0 == "set_option" && BANNED_OPTION_ROOTS.contains(&p[1].0));
    if let Some(pair) = option {
        return Some(if pair[1].0 == "pp" {
            "set_option pp"
        } else {
            "set_option debug"
        });
    }
    exec_word(&code)
}

/// Why `source` cannot be registered as the frozen file of `problem`
/// (`register`, stage `source`), or `None`: at most
/// [`MAX_FROZEN_BYTES`]; read by [`lean_code`], so it does not end inside a
/// comment or a `«` name; no token [`check_forbidden_token`] refuses (the
/// same rule as a partial proof, so nothing in it runs code while Lean
/// elaborates it); its statement ([`extract_statement`]) is the theorem
/// named `problem`; the proof after that statement's `:=` is `sorry` or
/// `by sorry` alone, since a frozen file is a theorem still to be proved,
/// and `sorry` stands nowhere in the code before it, since every
/// candidate keeps that text and `verify` refuses a file with `sorry` in
/// its code ([`forbidden_token`]); the last line of code before the
/// statement does not end in `in`, whose scope would be the command
/// `verify` puts after the preamble ([`preamble_copy`]) instead of the
/// theorem; and [`preamble_copy`] reads the file as a candidate of itself,
/// since `verify` builds the file Lean checks from what it finds.
#[must_use]
pub fn frozen_source_objection(problem: &str, source: &str) -> Option<String> {
    if source.len() > MAX_FROZEN_BYTES {
        return Some(format!(
            "the frozen file is {} bytes; at most {MAX_FROZEN_BYTES}",
            source.len()
        ));
    }
    let code = match lean_code(source) {
        Ok(code) => code,
        Err(why) => {
            return Some(format!(
                "the frozen file has a {why}; the checks that judge its proofs must read its code as Lean \
                 does, so close every comment and `«` name and write string literals plainly"
            ));
        }
    };
    if let Some(token) = check_forbidden_token(source) {
        return Some(format!("the frozen file has the forbidden token `{token}`"));
    }
    let Some(span) = statement_span(source) else {
        return Some(
            "the frozen file has no `theorem … :=` statement at the start of a line".to_owned(),
        );
    };
    let named = source
        .get(span.clone())
        .and_then(|statement| statement.strip_prefix("theorem"))
        .and_then(|rest| rest.trim_start().strip_prefix(problem))
        .is_some_and(|rest| rest.starts_with(|c: char| c.is_whitespace() || ":({[".contains(c)));
    if !named {
        return Some(format!(
            "the frozen file's theorem must be named `{problem}`, the problem id"
        ));
    }
    let commented = blanked(source, false)
        .map(|(text, _)| text)
        .unwrap_or_default();
    if !commented.get(span.end..).is_some_and(proved_by_sorry_alone) {
        return Some(
            "the frozen file's theorem must be proved by `sorry` alone (`:= sorry` or `:= by sorry`); put \
             no proof steps in the frozen file"
                .to_owned(),
        );
    }
    if code
        .get(..span.end)
        .is_none_or(|before| before.contains("sorry"))
    {
        return Some(
            "the definitions and statement before the proof contain `sorry`; every candidate must copy \
             them, so no proof could ever be accepted. Write the answer into the file in place of `sorry`"
                .to_owned(),
        );
    }
    let head = source.get(..span.start).unwrap_or_default();
    let last = code_lines(head)
        .ok()
        .and_then(|lines| lines.last().map(|l| l.text.clone()));
    if last.is_some_and(|line| line.rsplit(|c: char| !is_id_rest(c)).next() == Some("in")) {
        return Some(
            "the frozen file's last line before the theorem ends in `in` (`open … in`, `set_option … in`), \
             which scopes it to the next command; verify puts a command of its own after the preamble, so \
             drop the `in` and let the line apply to the rest of the file"
                .to_owned(),
        );
    }
    if let Err(why) = preamble_copy(source, source) {
        return Some(format!(
            "verify reads the frozen file's own preamble as it reads a candidate's, and cannot: {why}"
        ));
    }
    None
}

/// Whether `proof`, a frozen file's text after its statement's `:=` with
/// comments blanked, is `sorry` or `by sorry` and nothing more, up to a new
/// command on a later line (`end` of a namespace, say). A frozen file is a
/// statement to prove, not a proof skeleton: a step there, a `have … :=`
/// above all, would end the statement inside the proof.
fn proved_by_sorry_alone(proof: &str) -> bool {
    let proof = proof.trim_start();
    let proof = proof
        .strip_prefix("by")
        .filter(|rest| rest.starts_with(char::is_whitespace))
        .unwrap_or(proof);
    let Some(rest) = proof
        .trim_start()
        .strip_prefix("sorry")
        .filter(|rest| !rest.starts_with(is_id_rest))
    else {
        return false;
    };
    let next = rest.trim_start();
    next.is_empty() || (rest[..rest.len() - next.len()].contains('\n') && starts_command(next))
}

/// Why `frozen`'s header is not exactly `import Mathlib`, or `None`: its
/// first line of code must be that import and no other line an `import`
/// (comments, blank lines and string contents do not count). `check`,
/// `analyze` and `complete` elaborate in the resident REPL's `import Mathlib`
/// environment while `verify` runs the file's own header, so any other header
/// would let the two disagree about the same text.
#[must_use]
pub fn header_objection(frozen: &str) -> Option<String> {
    let code = match lean_code(frozen) {
        Ok(code) => code,
        Err(why) => return Some(why.to_owned()),
    };
    let mut lines = code.lines().map(str::trim).filter(|l| !l.is_empty());
    let is_import = |line: &str| {
        line.strip_prefix("import")
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
    };
    if lines.next() != Some("import Mathlib") {
        return Some("the frozen file must begin with exactly `import Mathlib`".to_owned());
    }
    if lines.any(is_import) {
        return Some(
            "the frozen file may import nothing but Mathlib, with one `import Mathlib` line"
                .to_owned(),
        );
    }
    None
}

/// Why `candidate` cannot be graded as a partial proof of `statement`, or
/// `None` when nothing textual stands in the way. `sorry` is allowed here:
/// this is the `check` rule, not `verify`'s.
///
/// The reason quotes the frozen statement back, because a caller that reflowed
/// it while editing (one space is enough) cannot see what differs from a
/// description alone.
#[must_use]
pub fn protocol_objection(statement: &str, candidate: &str) -> Option<String> {
    if let Some(token) = check_forbidden_token(candidate) {
        return Some(format!("forbidden token `{token}`"));
    }
    if !candidate.contains(statement) {
        return Some(format!(
            "the frozen theorem statement must appear verbatim; it is, byte for byte:\n{statement}"
        ));
    }
    None
}

/// Keywords that start a command and can never continue a term. Lean reads a
/// line that starts with anything else (an operator, a name, a bracket) as
/// more of the term before it, even at column 0.
#[rustfmt::skip]
const COMMAND_WORDS: &[&str] = &[
    "theorem", "lemma", "def", "abbrev", "instance", "example", "opaque", "structure", "class", "inductive",
    "namespace", "section", "end", "variable", "universe", "noncomputable", "private", "protected", "partial",
    "open", "set_option", "attribute", "local", "scoped", "mutual", "export", "import", "omit", "include",
    "irreducible_def",
];

/// Whether `text` starts a new command: a [`COMMAND_WORDS`] word or `@[`.
fn starts_command(text: &str) -> bool {
    let word = text
        .split(|c: char| !is_id_rest(c))
        .next()
        .unwrap_or_default();
    text.starts_with("@[") || COMMAND_WORDS.contains(&word)
}

/// One line of code as the preamble rule compares it ([`code_lines`]).
struct CodeLine {
    text: String,
    /// The byte offset just after its last character.
    end: usize,
}

/// The lines of code of `text`: comments blanked ([`lean_code`]'s lexer),
/// each line's end trimmed, blank lines dropped, except that a line break
/// inside a string literal does not end a line, so a literal is compared as
/// written, its blank lines and trailing spaces included.
fn code_lines(text: &str) -> Result<Vec<CodeLine>, &'static str> {
    let (code, strings) = blanked(text, false)?;
    let mut strings = strings.into_iter().peekable();
    let mut lines = vec![];
    let mut start = 0;
    let breaks = code
        .match_indices('\n')
        .map(|(i, _)| i)
        .chain(std::iter::once(code.len()));
    for at in breaks {
        while strings.next_if(|s| s.end <= at).is_some() {}
        if strings.peek().is_some_and(|s| s.start < at) {
            continue;
        }
        let line = code[start..at].trim_end();
        if !line.is_empty() {
            lines.push(CodeLine {
                text: line.to_owned(),
                end: start + line.len(),
            });
        }
        start = at + 1;
    }
    Ok(lines)
}

/// Where a candidate's copy of the frozen preamble ends ([`preamble_copy`]):
/// `frozen[..frozen_end]` is the preamble through its last character of code,
/// and `candidate[..candidate_end]` the candidate's copy of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreambleCopy {
    pub frozen_end: usize,
    pub candidate_end: usize,
}

/// Whether `candidate` keeps the preamble of `frozen` (everything before its
/// statement: imports, `open`s and the definitions the statement uses), and
/// where its copy ends; `Ok(None)` when the preamble has no code. The
/// candidate's first lines of code must be the preamble's, each byte for
/// byte, and the line after them must start a command ([`COMMAND_WORDS`],
/// `@[`). Blank lines and comments do not count on either side, so helper
/// lemmas may follow the preamble, but inside a string literal every
/// character counts.
///
/// `verify` compares the `pp.all` type, which names constants but does not
/// show their bodies, so this rule is what keeps a candidate from redefining
/// a `def` the statement uses. The command rule refuses a line that would
/// continue the preamble's last declaration only in the common cases (a
/// `protected` structure field starts with a command word, for one); what
/// guarantees it is that `verify` elaborates the frozen preamble itself up to
/// `frozen_end`, followed by a complete command of its own and then
/// `candidate[candidate_end..]`.
pub fn preamble_copy(frozen: &str, candidate: &str) -> Result<Option<PreambleCopy>, String> {
    let Some(head) = statement_span(frozen).and_then(|span| frozen.get(..span.start)) else {
        return Err(
            "the frozen file has no `theorem … :=` statement at the start of a line".to_owned(),
        );
    };
    let (preamble, lines) = match (code_lines(head), code_lines(candidate)) {
        (Ok(preamble), Ok(lines)) => (preamble, lines),
        (Err(why), _) | (_, Err(why)) => return Err(why.to_owned()),
    };
    let differs =
        |(k, want): &(usize, &CodeLine)| lines.get(*k).is_none_or(|got| got.text != want.text);
    if let Some((_, want)) = preamble.iter().enumerate().find(differs) {
        return Err(format!(
            "the candidate must begin with the frozen file's preamble (its lines before the theorem, blank \
             lines and comments aside), unchanged; the first line that differs is, byte for byte:\n{}",
            want.text
        ));
    }
    // Without a preamble there is no declaration to extend.
    let (Some(last), Some(copy)) = (
        preamble.last(),
        preamble.len().checked_sub(1).and_then(|k| lines.get(k)),
    ) else {
        return Ok(None);
    };
    let copy = PreambleCopy {
        frozen_end: last.end,
        candidate_end: copy.end,
    };
    let Some(next) = lines.get(preamble.len()).map(|l| l.text.trim_start()) else {
        return Ok(Some(copy));
    };
    if starts_command(next) {
        return Ok(Some(copy));
    }
    Err(format!(
        "the line after the frozen preamble must start a new command (`lemma`, `theorem`, `def`, `open`, `@[`, \
         …) so that it adds nothing to the preamble's last declaration; it starts: {}",
        next.chars().take(200).collect::<String>()
    ))
}

/// Why `candidate` does not keep the preamble of `frozen`, or `None`
/// ([`preamble_copy`]'s rule).
#[must_use]
pub fn preamble_objection(frozen: &str, candidate: &str) -> Option<String> {
    preamble_copy(frozen, candidate).err()
}

#[cfg(test)]
mod protocol_tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const FROZEN: &str = "import Mathlib\n\ntheorem demo (n : Nat) : n + 0 = n :=\nsorry\n";

    #[test]
    fn a_candidate_is_refused_for_the_two_things_lean_would_not_notice() {
        let statement = extract_statement(FROZEN).unwrap();
        assert_eq!(statement, "theorem demo (n : Nat) : n + 0 = n :=");
        // A partial proof that keeps the statement is gradeable, `sorry` and all.
        let partial =
            format!("import Mathlib\n\n{statement} by\n  have h : True := trivial\n  sorry\n");
        assert_eq!(protocol_objection(statement, &partial), None);
        // Reflowing the statement while editing is refused, though Lean
        // itself would not object to it.
        let reflowed = partial.replace("(n : Nat)", "(n : Nat )");
        assert!(
            protocol_objection(statement, &reflowed)
                .unwrap()
                .contains("byte for byte")
        );
        // The banned tactic, and a user axiom, in code rather than in a comment.
        assert!(
            protocol_objection(statement, &partial.replace("trivial", "by native_decide"))
                .is_some()
        );
        assert!(
            protocol_objection(statement, &format!("axiom cheat : False\n{partial}")).is_some()
        );
        assert_eq!(
            protocol_objection(
                statement,
                &format!("-- native_decide, in a comment\n{partial}")
            ),
            None,
            "a comment is not code"
        );
    }

    /// Blank lines and trailing spaces do not count between lines of code,
    /// but inside a string literal they are its value: dropping the blank
    /// line of `"a\n\nb"` makes `s.length = 3` true, and Lean accepts that
    /// proof with the pristine `pp.all` type.
    #[test]
    fn a_string_literal_in_the_preamble_is_kept_character_for_character() {
        let frozen = "import Mathlib\n\ndef s : String := \"a\n\nb  \"\n\ntheorem demo : s.length = 3 :=\nsorry\n";
        let proof =
            |preamble: &str| format!("{preamble}\n\ntheorem demo : s.length = 3 :=\n by decide\n");
        for (preamble, kept) in [
            ("import Mathlib\n\ndef s : String := \"a\n\nb  \"", true),
            (
                "-- a comment\nimport Mathlib\ndef s : String := \"a\n\nb  \"   ",
                true,
            ),
            ("import Mathlib\n\ndef s : String := \"a\nb  \"", false),
            ("import Mathlib\n\ndef s : String := \"a\n\nb\"", false),
            ("import Mathlib\n\ndef s : String := \"a\n  \nb  \"", false),
        ] {
            assert_eq!(
                preamble_objection(frozen, &proof(preamble)).is_none(),
                kept,
                "{preamble:?}"
            );
        }
    }

    /// A frozen file is read on its code, so a comment moves neither end of
    /// its statement, nor with it the preamble every candidate must keep;
    /// and it is refused when it ends inside a comment or a `«` name, has
    /// `sorry` before its proof, since verify could then accept no proof, or
    /// has a proof other than `sorry` alone, whose steps (a `have … :=`
    /// above all) would end the statement inside the proof.
    #[test]
    fn a_frozen_file_is_read_on_its_code_and_refused_when_no_proof_could_pass() {
        let moved = "import Mathlib\n\n/-\ntheorem kmp is about the answer below.\n-/\ndef answer : ℕ := 5\n\n\
                     theorem kmp : answer = 6 := by\n  sorry\n";
        // (the frozen file, its statement, a word of the refusal to register it)
        let rows: &[(&str, Option<&str>, Option<&str>)] = &[
            (moved, Some("theorem kmp : answer = 6 :="), None),
            (
                "import Mathlib\n\ntheorem kmp : 1 + 1 = 2 := by\n  sorry\n-- note: x := sorry\n",
                Some("theorem kmp : 1 + 1 = 2 :="),
                None,
            ),
            (
                "import Mathlib\n\nabbrev answer : ℕ := sorry\n\ntheorem kmp : answer = 6 := by\n  sorry\n",
                Some("theorem kmp : answer = 6 :="),
                Some("before the proof"),
            ),
            (
                "import Mathlib\n\ntheorem kmp : answer = 6 := by\n  sorry\n/- a note",
                None,
                Some("-/"),
            ),
            (
                "import Mathlib\n\ndef «answer : ℕ := 6\n\ntheorem kmp : answer = 6 := by\n  sorry\n",
                None,
                Some("»"),
            ),
            (
                "import Mathlib\n\ntheorem kmp : 1 + 1 = 2 := by\n  have h : 2 = 2 := sorry\n  sorry\n",
                Some("theorem kmp : 1 + 1 = 2 := by\n  have h : 2 = 2 :="),
                Some("alone"),
            ),
            (
                "import Mathlib\n\ntheorem kmp : 1 + 1 = 2 := by\n  norm_num\n  sorry\n",
                Some("theorem kmp : 1 + 1 = 2 :="),
                Some("alone"),
            ),
        ];
        for &(frozen, statement, refusal) in rows {
            assert_eq!(extract_statement(frozen), statement, "{frozen}");
            let objection = frozen_source_objection("kmp", frozen);
            assert_eq!(
                objection
                    .as_deref()
                    .map(|why| refusal.is_some_and(|word| why.contains(word))),
                refusal.map(|_| true),
                "{frozen}: {objection:?}"
            );
        }
        // The preamble ends after `def answer`, so a candidate that gives it
        // another value is refused, the statement copied into a comment or not.
        let forged = "import Mathlib\n\ndef answer : ℕ := 6\n\n/- /-\ntheorem kmp is about the answer below.\n-/\n\
                      def answer : ℕ := 5\n\ntheorem kmp : answer = 6 :=\n-/\n\ntheorem kmp : answer = 6 := by\n  rfl\n";
        assert!(
            preamble_objection(moved, forged)
                .is_some_and(|why| why.contains("def answer : ℕ := 5"))
        );
    }

    #[test]
    fn a_registered_header_is_exactly_import_mathlib() {
        let body = "\n\ntheorem demo : True :=\nsorry\n";
        for (header, ok) in [
            ("import Mathlib", true),
            ("-- the statement\n/- a comment -/\nimport Mathlib", true),
            ("import Mathlib.Tactic", false),
            ("import Mathlib\nimport Aesop", false),
            ("import Mathlib Aesop", false),
            ("open Nat", false),
            ("", false),
        ] {
            let frozen = format!("{header}{body}");
            assert_eq!(
                header_objection(&frozen).is_none(),
                ok,
                "{header:?}: {:?}",
                header_objection(&frozen)
            );
        }
        // A word that merely starts with `import` is not an import.
        assert_eq!(
            header_objection(&format!("import Mathlib\ndef imported := 1{body}")),
            None
        );
    }
}
