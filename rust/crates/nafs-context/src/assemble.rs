//! The budgeted context assembler (spec §4, "summary before source"): every hit first enters at its cheapest level,
//! so the prompt shows more sources; the budget left is then spent showing the best hits in full. A level that does
//! not fit falls back to a cheaper one instead of being cut mid-sentence. Category quotas keep one kind of source from
//! taking the whole prompt; a quota a category does not use passes to the others.

use std::collections::{HashMap, HashSet};

/// One candidate for the prompt.
#[derive(Debug, Clone)]
pub struct Item {
    pub uri: String,
    /// "passage", "memory", "skill", ...
    pub category: String,
    pub score: f64,
    /// The levels it can be shown at, cheapest first (an abstract or best sentence, an overview, the full text); a
    /// level that is `None` is skipped.
    pub levels: Vec<Option<String>>,
}

/// What the assembler chose: item index and level index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen {
    pub item: usize,
    pub level: usize,
}

#[derive(Debug, Clone)]
pub struct Budget {
    /// Characters the block may hold.
    pub chars: usize,
    /// At most this many items per category; categories not named are unlimited.
    pub quotas: HashMap<String, usize>,
    /// At most this many items in all.
    pub max_items: usize,
}

fn norm_key(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Choose items and levels (spec §4). Items are taken in score order; each enters at its cheapest level within a
/// per-entry cap of `2 · budget / n`; then, best first, items are raised to their next level while the budget lasts.
pub fn assemble(items: &[Item], budget: &Budget) -> Vec<Chosen> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&a, &b| items[b].score.total_cmp(&items[a].score).then(a.cmp(&b)));
    // Quotas, with the slots a category leaves unused passed on in score order.
    let mut taken: HashMap<&str, usize> = HashMap::new();
    let mut picked: Vec<usize> = Vec::new();
    let mut deferred: Vec<usize> = Vec::new();
    for &i in &order {
        if picked.len() >= budget.max_items {
            break;
        }
        let cat = items[i].category.as_str();
        let used = taken.entry(cat).or_default();
        if budget.quotas.get(cat).is_some_and(|q| *used >= *q) {
            deferred.push(i);
            continue;
        }
        *used += 1;
        picked.push(i);
    }
    let total_quota: usize = budget.quotas.values().sum();
    for i in deferred {
        if picked.len() >= budget.max_items.min(total_quota.max(picked.len())) {
            break;
        }
        picked.push(i);
    }
    if picked.is_empty() {
        return Vec::new();
    }
    let cap = (2 * budget.chars / picked.len()).max(1);
    let mut seen: HashSet<String> = HashSet::new();
    let mut chosen: Vec<Chosen> = Vec::new();
    let mut used = 0usize;
    // Breadth: every item at its cheapest level that fits the cap and what is left.
    for &i in &picked {
        let fits = items[i].levels.iter().enumerate().find(|(_, l)| {
            l.as_ref()
                .is_some_and(|t| t.len() <= cap && used + t.len() <= budget.chars)
        });
        let Some((level, Some(text))) = fits else {
            continue;
        };
        // The same text twice (two hits on one passage) is shown once.
        if !seen.insert(norm_key(text)) {
            continue;
        }
        used += text.len();
        chosen.push(Chosen { item: i, level });
    }
    // Depth: best first, raise to the richest level that still fits.
    for c in chosen.iter_mut() {
        let levels = &items[c.item].levels;
        let current = levels[c.level].as_ref().map_or(0, String::len);
        for next in (c.level + 1..levels.len()).rev() {
            let Some(t) = &levels[next] else {
                continue;
            };
            if used - current + t.len() <= budget.chars && !seen.contains(&norm_key(t)) {
                used = used - current + t.len();
                seen.insert(norm_key(t));
                c.level = next;
                break;
            }
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(uri: &str, cat: &str, score: f64, sizes: &[usize]) -> Item {
        Item {
            uri: uri.into(),
            category: cat.into(),
            score,
            levels: sizes
                .iter()
                .map(|&n| Some(format!("{uri}:{}", "x".repeat(n))))
                .collect(),
        }
    }

    #[test]
    fn breadth_first_then_the_best_hits_in_full() {
        let items = vec![
            item("a", "passage", 0.9, &[50, 1000]),
            item("b", "passage", 0.8, &[50, 1000]),
            item("c", "passage", 0.7, &[50, 1000]),
            item("d", "passage", 0.6, &[50, 1000]),
        ];
        let b = Budget {
            chars: 2300,
            quotas: HashMap::new(),
            max_items: 8,
        };
        let got = assemble(&items, &b);
        assert_eq!(got.len(), 4, "every hit is shown");
        // The two best get their full text; the rest stay at their abstract.
        assert_eq!(
            got.iter().map(|c| c.level).collect::<Vec<_>>(),
            [1, 1, 0, 0]
        );
        // Four whole passages would not have fit.
        assert!(4 * 1002 > b.chars);
    }

    #[test]
    fn quotas_pass_unused_slots_on_and_duplicates_are_dropped() {
        let mut items = vec![
            item("m1", "memory", 0.95, &[40]),
            item("m2", "memory", 0.94, &[40]),
            item("m3", "memory", 0.93, &[40]),
            item("p1", "passage", 0.5, &[40]),
        ];
        items.push(Item {
            uri: "p1-again".into(),
            ..items[3].clone()
        });
        let b = Budget {
            chars: 10_000,
            quotas: HashMap::from([("memory".to_string(), 1), ("passage".to_string(), 3)]),
            max_items: 4,
        };
        let got = assemble(&items, &b);
        let uris: Vec<&str> = got.iter().map(|c| items[c.item].uri.as_str()).collect();
        assert!(uris.contains(&"m1") && uris.contains(&"p1"), "{uris:?}");
        assert!(!uris.contains(&"p1-again"), "same text once: {uris:?}");
        // Memory's quota of one is filled first; slots the passages left go to the next memories.
        assert!(uris.contains(&"m2"), "{uris:?}");
    }

    #[test]
    fn a_level_that_does_not_fit_falls_back_and_nothing_is_cut() {
        let items = vec![
            item("a", "passage", 0.9, &[100, 5000]),
            item("b", "passage", 0.8, &[3000]),
        ];
        let b = Budget {
            chars: 1000,
            quotas: HashMap::new(),
            max_items: 4,
        };
        let got = assemble(&items, &b);
        assert_eq!(got, [Chosen { item: 0, level: 0 }]);
    }
}
