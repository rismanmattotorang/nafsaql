//! Chat templates, rendered in Rust from the shape the GGUF's own Jinja template
//! describes.
//!
//! The engine does not run Jinja. It recognises the two templates the catalogue's
//! models use — ChatML (SmolLM2, Qwen2.5) and Llama 3 — from the markers in
//! `tokenizer.chat_template`, and renders them so that the token stream is the one
//! llama.cpp's `--jinja` path produces for the same messages. That equality is what
//! makes a greedy answer from this engine comparable with llama.cpp's on the same
//! file: a prompt that differs by one newline is a different prompt.
//!
//! Anything else is refused at load (`detect`), never approximated with ChatML:
//! a model prompted in a template it was not trained on answers fluently and wrongly.

use crate::gguf::Gguf;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>role\n…<|im_end|>\n`
    ChatMl,
    /// `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`
    Llama3,
    /// Qwen3's ChatML: no default system message, and a reasoning block the model
    /// opens itself — or, with thinking switched off, the template writes an empty
    /// `<think>\n\n</think>\n\n` at the start of the assistant turn.
    Qwen3,
}

/// Template kinds this crate renders, for the compatibility probe's message.
pub const TEMPLATES: &[&str] = &["chatml", "llama3", "qwen3"];

impl Template {
    pub fn name(self) -> &'static str {
        match self {
            Template::ChatMl => "chatml",
            Template::Llama3 => "llama3",
            Template::Qwen3 => "qwen3",
        }
    }

    /// Which template the file's own `tokenizer.chat_template` is, and whether the
    /// vocabulary carries the control tokens it needs. A missing template string is
    /// treated as ChatML only when the ChatML tokens are present.
    pub fn detect(g: &Gguf) -> Result<Template, String> {
        let toks = g.arr_str("tokenizer.ggml.tokens")?;
        let has = |t: &str| toks.iter().any(|x| x == t);
        let tpl = g.str("tokenizer.chat_template").unwrap_or("");
        let kind = if tpl.contains("<|start_header_id|>") {
            Template::Llama3
        } else if tpl.contains("<|im_start|>") && tpl.contains("enable_thinking") {
            Template::Qwen3
        } else if tpl.contains("<|im_start|>") || (tpl.is_empty() && has("<|im_start|>")) {
            Template::ChatMl
        } else {
            return Err(format!(
                "chat template is neither ChatML nor Llama 3 — Smiley Core renders {}",
                TEMPLATES.join(", ")
            ));
        };
        for t in kind.required_tokens() {
            if !has(t) {
                return Err(format!(
                    "no {t} token — Smiley Core renders the {} template",
                    kind.name()
                ));
            }
        }
        Ok(kind)
    }

    pub fn required_tokens(self) -> &'static [&'static str] {
        match self {
            Template::ChatMl => &["<|im_start|>", "<|im_end|>"],
            Template::Qwen3 => &["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
            Template::Llama3 => &[
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
        }
    }

    /// Token ids that end an assistant turn under this template, besides the
    /// tokenizer's own `eos`. Llama 3 instruct models stop on `<|eot_id|>` and may
    /// emit `<|eom_id|>`; `<|end_of_text|>` ends the document.
    pub fn stop_ids(self, tk: &Tokenizer) -> Vec<u32> {
        let names: &[&str] = match self {
            Template::ChatMl | Template::Qwen3 => &["<|im_end|>", "<|endoftext|>"],
            Template::Llama3 => &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"],
        };
        let mut ids: Vec<u32> = names.iter().filter_map(|n| tk.id_of(n)).collect();
        if !ids.contains(&tk.eos) {
            ids.push(tk.eos);
        }
        ids
    }

    /// The system message the model's own template inserts when the caller gives
    /// none. SmolLM2 and Qwen2.5 each name themselves; Llama 3.2 inserts only its
    /// date header (rendered in `system_block`).
    pub fn default_system(self, model_name: &str) -> Option<&'static str> {
        match self {
            Template::ChatMl if model_name.to_ascii_lowercase().contains("qwen") => {
                Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
            }
            Template::ChatMl => {
                Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
            }
            Template::Llama3 | Template::Qwen3 => None,
        }
    }

    /// What the template writes at the start of the assistant's reply before the
    /// model continues: Qwen3 with thinking off gets an empty reasoning block, which
    /// is how its own template (`enable_thinking=false`) switches reasoning off.
    pub fn assistant_open(self, thinking: bool) -> &'static str {
        match self {
            Template::Qwen3 if !thinking => "<think>\n\n</think>\n\n",
            _ => "",
        }
    }

    /// The text that opens a conversation: the system block, before any user turn.
    pub fn system_block(self, system: Option<&str>, today: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => match system {
                Some(sys) => format!("<|im_start|>system\n{sys}<|im_end|>\n"),
                None => String::new(),
            },
            // Llama 3.2's template always writes the header with the knowledge
            // cut-off and today's date (`strftime_now("%d %b %Y")`), then the system
            // message if any — so an absent system message still costs these lines.
            Template::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                 Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n{}<|eot_id|>",
                system.unwrap_or("")
            ),
        }
    }

    /// One user message followed by the assistant header the model continues from.
    pub fn user_turn(self, user: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => {
                format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
            }
            Template::Llama3 => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        }
    }

    /// A completed assistant message, as it appears in a replayed transcript.
    pub fn assistant_turn(self, text: &str) -> String {
        match self {
            Template::ChatMl | Template::Qwen3 => format!("{text}<|im_end|>\n"),
            Template::Llama3 => format!("{text}<|eot_id|>"),
        }
    }

    /// What closes the previous assistant message when a conversation continues
    /// incrementally: the end-of-turn token was sampled but never fed, so the cache
    /// holds the answer without its terminator.
    pub fn close_previous(self) -> &'static str {
        match self {
            Template::ChatMl | Template::Qwen3 => "<|im_end|>\n",
            Template::Llama3 => "<|eot_id|>",
        }
    }
}

/// Today's date as Llama 3's template formats it (`%d %b %Y`, e.g. `14 Oct 2026`),
/// from the system clock, with no dependency: the civil-from-days algorithm.
pub fn today_llama3() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    const MON: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!("{d:02} {} {y}", MON[(m - 1) as usize])
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_renders_the_smollm_transcript_shape() {
        let t = Template::ChatMl;
        let s = t.system_block(Some("S"), "x") + &t.user_turn("U");
        assert_eq!(s, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(t.assistant_turn("A"), "A<|im_end|>\n");
        assert_eq!(t.close_previous(), "<|im_end|>\n");
        assert_eq!(t.system_block(None, "x"), "");
    }

    #[test]
    fn llama3_always_writes_the_date_header() {
        let t = Template::Llama3;
        let s = t.system_block(None, "14 Oct 2026");
        assert!(s.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"));
        assert!(s.contains("Cutting Knowledge Date: December 2023\nToday Date: 14 Oct 2026\n\n"));
        assert!(s.ends_with("\n\n<|eot_id|>"), "{s:?}");
        let s = t.system_block(Some("Be terse."), "14 Oct 2026");
        assert!(
            s.ends_with("Today Date: 14 Oct 2026\n\nBe terse.<|eot_id|>"),
            "{s:?}"
        );
        assert_eq!(t.user_turn("U"),
            "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        assert_eq!(t.close_previous(), "<|eot_id|>");
    }

    #[test]
    fn default_system_follows_the_model_family() {
        assert!(Template::ChatMl
            .default_system("Qwen2.5 0.5B Instruct")
            .unwrap()
            .contains("Qwen"));
        assert!(Template::ChatMl
            .default_system("SmolLM2 360M Instruct")
            .unwrap()
            .contains("SmolLM"));
        assert!(Template::Llama3
            .default_system("Llama 3.2 1B Instruct")
            .is_none());
    }

    #[test]
    fn civil_dates_are_right_across_leap_years_and_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(20_740), (2026, 10, 14));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        let today = today_llama3();
        assert_eq!(today.len(), 11, "{today:?}");
        assert!(
            today.ends_with("202")
                || today.ends_with("203")
                || today.chars().rev().take(4).all(|c| c.is_ascii_digit())
        );
    }
}
