//! Charts and diagrams in answers. When a question asks for one, the model is told the exact block the chat page
//! draws (a ```chart block of JSON for Nivo, or a ```mermaid block) instead of the usual domain instruction.
//!
//! The wording was chosen by a sweep of the models (scripts/visual_sweep.py, docs/decisions/025): without it a small
//! model draws neither, with it most answers draw, and the short wording beat the long one. `scripts/visual_hints.py`
//! (variant C) holds the same text for the sweep.
use nafs_types::Domain;

/// Words that mean the user wants a diagram (checked first: "a pie chart in Mermaid" is a diagram).
const DIAGRAM_WORDS: [&str; 10] = [
    "diagram",
    "flowchart",
    "flow chart",
    "mermaid",
    "state machine",
    "state chart",
    "process map",
    "sequence of messages",
    "swimlane",
    "uml",
];
/// Words that mean the user wants a chart of data.
const CHART_WORDS: [&str; 20] = [
    "chart",
    "bar graph",
    "line graph",
    "pie graph",
    "histogram",
    "heatmap",
    "heat map",
    "treemap",
    "tree map",
    "sankey",
    "funnel",
    "scatter",
    "radar",
    "plot a ",
    "plot the ",
    "plot of",
    "visualize",
    "visualise",
    "visualization",
    "visualisation",
];
/// Asking for code or a file is not asking for a drawing, and the code answer keeps its own instruction.
const CODE_WORDS: [&str; 12] = [
    "python",
    "matplotlib",
    "javascript",
    "code",
    "script",
    "function",
    "ggplot",
    "d3.js",
    "excel",
    "pandas",
    "sql",
    "program",
];

const CHART_HEAD: &str = "The user wants a chart. Reply with one short sentence, then a ```chart block with only JSON, using exactly the numbers given. Never draw with text characters. Layout:\n```chart\n";
const DIAGRAM_HEAD: &str = "The user wants a diagram. Reply with one short sentence, then a ```mermaid block. Replace every <...> with the user's own words and draw every step or part they named, one node each, with short letter ids and the text in brackets. No styling. Layout:\n```mermaid\n";

/// One example block per kind of chart: a small model copies the shape it is shown, so it is shown the right one.
const CHART_EXAMPLES: [(&[&str], &str); 11] = [
    (
        &["sankey"],
        r#"{"type":"sankey","title":"<title>","nodes":[{"id":"<name>"}],"links":[{"source":"<name>","target":"<name>","value":<number>}]}"#,
    ),
    (
        &["treemap", "tree map"],
        r#"{"type":"treemap","title":"<title>","data":{"name":"all","children":[{"name":"<name>","value":<number>}]}}"#,
    ),
    (
        &["funnel"],
        r#"{"type":"funnel","title":"<title>","data":[{"id":"<name>","label":"<name>","value":<number>}]}"#,
    ),
    (
        &["heatmap", "heat map"],
        r#"{"type":"heatmap","title":"<title>","data":[{"id":"<row>","data":[{"x":"<column>","y":<number>}]}]}"#,
    ),
    (
        &["radar", "spider"],
        r#"{"type":"radar","title":"<title>","indexBy":"metric","keys":["<series 1>","<series 2>"],"data":[{"metric":"<name>","<series 1>":<number>,"<series 2>":<number>}]}"#,
    ),
    (
        &["scatter"],
        r#"{"type":"scatter","title":"<title>","series":[{"id":"<name>","data":[{"x":<number>,"y":<number>}]}]}"#,
    ),
    (
        &["donut", "doughnut"],
        r#"{"type":"donut","title":"<title>","data":[{"label":"<name>","value":<number>}]}"#,
    ),
    (
        &["pie"],
        r#"{"type":"pie","title":"<title>","data":[{"label":"<name>","value":<number>}]}"#,
    ),
    (
        &["line", "area", "trend", "over time"],
        r#"{"type":"line","title":"<title>","series":[{"id":"<name>","data":[{"x":"<label>","y":<number>}]}]}"#,
    ),
    (
        &["compare", "stacked", "grouped", "versus", " vs "],
        r#"{"type":"bar","title":"<title>","indexBy":"label","keys":["<series 1>","<series 2>"],"data":[{"label":"<name>","<series 1>":<number>,"<series 2>":<number>}]}"#,
    ),
    (
        &["bar", "column", "histogram"],
        r#"{"type":"bar","title":"<title>","indexBy":"label","keys":["<series>"],"data":[{"label":"<name>","<series>":<number>}]}"#,
    ),
];
const CHART_TYPES: &str = "Replace every <...> with the user's own names and numbers, one entry for each item given, and keep the JSON complete and closed. Other types: bar, line, pie, donut, radar, scatter, heatmap, treemap, funnel, sankey.";

const DIAGRAM_FLOW: &str = "flowchart TD\n  A[<first step>] --> B[<second step>]\n  B --> C{<question?>}\n  C -->|yes| D[<step>]\n  C -->|no| A";
const DIAGRAM_SEQ: &str = "sequenceDiagram\n  participant A as <name>\n  participant B as <name>\n  A->>B: <message>\n  B-->>A: <reply>";
const DIAGRAM_STATE: &str =
    "stateDiagram-v2\n  [*] --> Idle\n  Idle --> Busy: start\n  Busy --> Idle: stop\n  Busy --> [*]";
const DIAGRAM_PIE: &str = "pie title <title>\n  \"<name>\" : <number>";

/// The first example whose words appear in the (lower-case) question.
fn example_for(q: &str, table: &[(&[&str], &'static str)], default: &'static str) -> &'static str {
    table
        .iter()
        .find(|(words, _)| words.iter().any(|w| q.contains(w)))
        .map_or(default, |(_, ex)| ex)
}

/// The instruction for a question that asks for a chart or a diagram, or `None` for any other question.
pub fn instruction(question: &str, dom: Domain) -> Option<String> {
    if matches!(dom, Domain::Code | Domain::Structured) {
        return None;
    }
    let q = question.to_lowercase();
    if CODE_WORDS.iter().any(|w| contains_word(&q, w)) {
        return None;
    }
    if DIAGRAM_WORDS.iter().any(|w| q.contains(w)) {
        let ex = if q.contains("sequence") || q.contains("between the") {
            DIAGRAM_SEQ
        } else if q.contains("state") {
            DIAGRAM_STATE
        } else if contains_word(&q, "pie") {
            DIAGRAM_PIE
        } else {
            DIAGRAM_FLOW
        };
        return Some(format!("{DIAGRAM_HEAD}{ex}\n```\nOther kinds: flowchart, sequenceDiagram, stateDiagram-v2, pie."));
    }
    if CHART_WORDS.iter().any(|w| q.contains(w)) {
        let ex = example_for(&q, &CHART_EXAMPLES, CHART_EXAMPLES[10].1);
        return Some(format!("{CHART_HEAD}{ex}\n```\n{CHART_TYPES}"));
    }
    None
}

/// `word` as a whole word (so "code" does not match "barcode" or "decode").
fn contains_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + word.len()..].chars().next();
        !before.is_some_and(|c| c.is_alphanumeric()) && !after.is_some_and(|c| c.is_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(q: &str) -> &'static str {
        match instruction(q, Domain::Other) {
            Some(i) if i.starts_with(CHART_HEAD) => "chart",
            Some(_) => "diagram",
            None => "none",
        }
    }

    #[test]
    fn charts_and_diagrams_are_recognised() {
        for q in [
            "Make a bar chart of monthly sales: Jan 120, Feb 150.",
            "Show a pie chart of browser share",
            "Draw a donut chart of a budget",
            "Make a radar chart for two phones",
            "Visualize the results as a heatmap",
            "plot the temperature by hour",
            "Draw a sankey of energy flows",
            "Make a funnel chart for signups",
        ] {
            assert_eq!(kind(q), "chart", "{q}");
        }
        for q in [
            "Draw a flowchart of how to make tea",
            "Make a sequence diagram of a login",
            "Draw a state diagram of a traffic light",
            "Draw a Mermaid pie chart of how a day is spent",
            "Make a diagram of the water cycle.",
        ] {
            assert_eq!(kind(q), "diagram", "{q}");
        }
    }

    #[test]
    fn other_questions_and_code_are_left_alone() {
        for q in [
            "What is the capital of France?",
            "Write a python function that plots a bar chart",
            "Write JavaScript code for a pie chart with d3.js",
            "How do I read a barcode?",
            "What is 12 * 12 + 6?",
            "Explain what a chart of accounts is for in code",
        ] {
            assert_eq!(kind(q), "none", "{q}");
        }
        assert!(instruction("Make a bar chart", Domain::Code).is_none());
        assert!(instruction("Make a bar chart", Domain::Structured).is_none());
    }

    #[test]
    fn the_example_matches_the_kind_asked_for() {
        let i = |q: &str| instruction(q, Domain::Other).unwrap();
        assert!(i("Draw a donut chart of a budget").contains(r#""type":"donut""#));
        assert!(i("Make a pie chart of browser share").contains(r#""type":"pie""#));
        assert!(i("Plot a line chart of temperature").contains(r#""type":"line""#));
        assert!(i("Make a stacked bar chart of votes").contains(r#""type":"bar""#));
        assert!(i("Show a sankey chart of energy").contains(r#""type":"sankey""#));
        assert!(i("Make a heatmap of sales").contains(r#""type":"heatmap""#));
        assert!(i("Make a chart of sales").contains(r#""type":"bar""#));
        assert!(i("Draw a sequence diagram of a login").contains("sequenceDiagram\n  participant"));
        assert!(i("Draw a state diagram of a traffic light").contains("stateDiagram-v2\n  [*]"));
        assert!(i("Draw a Mermaid pie chart of a day").contains("pie title"));
        assert!(i("Draw a flowchart of making tea").contains("flowchart TD"));
        assert!(
            i("Make a chart of sales").contains("```chart")
                && i("Draw a flowchart").contains("```mermaid")
        );
    }
}
