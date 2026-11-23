use super::*;

const CODE: &str = r#"
@semantic("Total the cost of some items bought at one price each")
def item_cost(ctx):
    count = load(ctx, "number of items bought", int)
    price = load(ctx, "price of one item in dollars", int)
    narrate(ctx, f"{count} x ${price} = ${count * price}")
    return {"the cost": count * price}

@semantic("Work out the change from paying for pencils and notebooks", root=True)
def change(ctx):
    a = do(ctx, "Total the cost of items at one price", {"number of items bought": load(ctx, "number of pencils bought", int), "price of one item in dollars": load(ctx, "price per pencil in dollars", int)})
    b = do(ctx, "Total the cost of items at one price", {"number of items bought": load(ctx, "number of notebooks bought", int), "price of one item in dollars": load(ctx, "price per notebook in dollars", int)})
    paid = load(ctx, "amount paid in dollars", int)
    total = load(a, "cost", int) + load(b, "cost", int)
    narrate(ctx, f"Change = ${paid} - ${total} = ${paid - total}")
    return {"answer": paid - total, "summary": f"The change is ${paid - total}."}
"#;

const Q: &str = "A store sells pencils at 3 dollars each and notebooks at 7 dollars each. Maria buys 4 pencils and 5 notebooks and pays with a 100 dollar bill. How much change does she receive?";

fn params() -> Vec<Param> {
    let p = |n: &str, d: &str, v: &str| Param {
        name: n.into(),
        ty: "int".into(),
        description: d.into(),
        value: v.into(),
    };
    vec![
        p("pencil_price", "price per pencil in dollars", "3"),
        p("notebook_price", "price per notebook in dollars", "7"),
        p("pencils", "number of pencils bought", "4"),
        p("notebooks", "number of notebooks bought", "5"),
        p("paid", "amount paid in dollars", "100"),
    ]
}

fn entry() -> Entry {
    Entry {
        id: "e1".into(),
        created_at: now(),
        domain: "math".into(),
        question: normalize(Q),
        template: synthesize_template(Q, &params()).unwrap(),
        params: params(),
        expected: "53".into(),
        code: CODE.into(),
        functions: vec![],
        root: "change".into(),
        links: BTreeMap::new(),
        status: Status::Candidate,
        stats: Stats::default(),
        build: Value::Null,
        narration: vec![],
        source_reply: None,
        vectors: vec![],
        fn_vectors: vec![],
    }
}

struct TempDir(PathBuf);
impl TempDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn store() -> (ArcStore, TempDir) {
    let dir =
        TempDir(std::env::temp_dir().join(format!("nafs-arc-{}", uuid::Uuid::new_v4().simple())));
    std::fs::create_dir_all(dir.path()).unwrap();
    let limits = SandboxLimits {
        cpu_time_s: 10,
        wall_clock_s: 30,
        ..Default::default()
    };
    (ArcStore::open(dir.path().to_str().unwrap(), limits), dir)
}

#[test]
fn template_is_synthesized_from_the_question() {
    let t = synthesize_template(Q, &params()).unwrap();
    assert!(t.contains("pencils at {pencil_price} dollars each"), "{t}");
    assert!(t.contains("a {paid} dollar bill"), "{t}");
    // A value written twice is ambiguous.
    let mut p = params();
    p[0].value = "7".into();
    assert!(synthesize_template(Q, &p).is_none());
}

#[test]
fn template_matches_new_numbers_and_rejects_other_questions() {
    let e = entry();
    let q2 = Q
        .replace("4 pencils", "12 pencils")
        .replace("100 dollar", "250 dollar");
    let got = match_template(&e, &q2).unwrap();
    assert_eq!(got["number of pencils bought"], "12");
    assert_eq!(got["amount paid in dollars"], "250");
    // Another question with the same numbers is not this entry.
    let other = Q.replace(
        "How much change does she receive?",
        "How much does she spend?",
    );
    assert!(match_template(&e, &other).is_none());
    assert!(match_shape(&e, &other).is_none());
}

#[test]
fn shape_tier_maps_numbers_by_position() {
    let mut e = entry();
    e.template = "unrelated {pencils}".into();
    let q2 = Q.replace('3', "2").replace("5 notebooks", "6 notebooks");
    let got = match_shape(&e, &q2).unwrap();
    assert_eq!(got["price per pencil in dollars"], "2");
    assert_eq!(got["number of notebooks bought"], "6");
}

#[test]
fn guards_on_values() {
    let e = entry();
    assert!(verbatim(Q, "100"));
    assert!(verbatim(Q, "100.0"));
    assert!(!verbatim(Q, "101"));
    let mut vals: BTreeMap<String, String> = e
        .params
        .iter()
        .map(|p| (p.description.clone(), p.value.clone()))
        .collect();
    assert!(unused_numbers(&e, Q, &vals).is_empty());
    let q2 = format!("{Q} She also buys an eraser for 2 dollars.");
    assert_eq!(unused_numbers(&e, &q2, &vals), vec!["2".to_string()]);
    vals.remove("amount paid in dollars");
    assert_eq!(unused_numbers(&e, Q, &vals), vec!["100".to_string()]);
}

#[test]
fn answers_compare_to_the_written_precision() {
    assert!(same_answer("53", "Maria receives $\\boxed{53}$ dollars."));
    assert!(same_answer("5.1321", "The pressure is \\boxed{5.13} MPa"));
    assert!(!same_answer("5.2", "\\boxed{5.13}"));
    assert!(!same_answer("54", "\\boxed{53}"));
    assert!(same_answer("3/4", "\\boxed{3/4}"));
    assert!(same_answer("0.75", "\\boxed{0.75}"));
}

#[test]
fn draft_sections_parse() {
    let reply = format!(
        "TEMPLATE:\nHow many divisors does {{n}} have?\n\nPARAMS:\nn | integer | the integer | 196\n\nCODE:\n```python\n{CODE}\n```"
    );
    let d = parse_draft(&reply).unwrap();
    assert_eq!(d.template, "How many divisors does {n} have?");
    assert_eq!(d.params[0].ty, "int");
    assert!(d.code.contains("@semantic"));
    assert!(check_template("How many divisors does 196 have?", &d).is_ok());
    assert!(check_template("How many divisors does 195 have?", &d).is_err());
}

#[test]
fn extraction_reply_parses_and_refuses_other_questions() {
    let e = entry();
    let ok = r#"{"same": true, "pencil_price": "3", "notebook_price": 7, "pencils": "4", "notebooks": "5", "paid": "100"}"#;
    let v = parse_extraction(&e, ok).unwrap();
    assert_eq!(v["price per notebook in dollars"], "7");
    assert!(parse_extraction(&e, r#"{"same": false}"#).is_err());
    assert!(parse_extraction(&e, "no json").is_err());
}

#[test]
fn entry_floor_learns_from_outcomes() {
    let cfg = ArcCfg::default();
    let mut e = entry();
    assert!((entry_floor(&e, &cfg) - 0.85).abs() < 1e-6);
    e.stats.min_agreed = Some(0.83);
    assert!((entry_floor(&e, &cfg) - 0.83).abs() < 1e-6);
    e.stats.max_disagreed = Some(0.88);
    assert!(entry_floor(&e, &cfg) > 0.88);
    e.stats.min_agreed = Some(0.5);
    e.stats.max_disagreed = None;
    assert!((entry_floor(&e, &cfg) - 0.80).abs() < 1e-6);
}

#[test]
fn shadow_comparisons_promote_and_quarantine() {
    let (s, _d) = store();
    let cfg = ArcCfg::default();
    s.insert(entry(), 10);
    s.record_comparison("e1", true, false, Tier::Template, 1.0, &cfg);
    assert_eq!(s.get("e1").unwrap().status, Status::Candidate);
    s.record_comparison("e1", true, true, Tier::Template, 1.0, &cfg);
    assert_eq!(s.get("e1").unwrap().status, Status::Trusted);
    s.record_comparison("e1", false, true, Tier::Template, 1.0, &cfg);
    assert_eq!(s.get("e1").unwrap().status, Status::Quarantined);
    assert!(s.lookup_text(Q).is_none());
}

#[test]
fn eviction_keeps_entries_others_call() {
    let (s, _d) = store();
    let mut a = entry();
    a.id = "a".into();
    a.stats.hits = 0;
    let mut b = entry();
    b.id = "b".into();
    b.stats.hits = 5;
    b.links.insert("x".into(), "a:item_cost".into());
    let mut c = entry();
    c.id = "c".into();
    c.stats.hits = 1;
    s.insert(a, 2);
    s.insert(b, 2);
    s.insert(c, 2);
    let ids: Vec<String> = s.list().into_iter().map(|e| e.id).collect();
    assert!(
        ids.contains(&"a".to_string()) && ids.contains(&"b".to_string()),
        "{ids:?}"
    );
}

#[tokio::test]
async fn runs_in_the_sandbox_with_semantic_calls() {
    let (s, _d) = store();
    let e = entry();
    let (fns, problems) = s.describe(CODE).await.unwrap();
    assert!(problems.is_empty(), "{problems:?}");
    assert_eq!(fns.iter().filter(|f| f.root).count(), 1);
    let q2 = Q.replace("4 pencils", "10 pencils");
    let vals = match_template(&e, &q2).unwrap();
    let out = s.invoke(&e, &vals).await.unwrap();
    assert_eq!(out.answer, "35");
    assert!(
        out.narration.iter().any(|n| n.contains("10 x $3")),
        "{:?}",
        out.narration
    );
    // A value that cannot be converted is an invocation error, not a wrong answer.
    let mut bad = vals.clone();
    bad.insert("amount paid in dollars".into(), "lots".into());
    assert!(s.invoke(&e, &bad).await.is_err());
}

#[tokio::test]
async fn forbidden_code_is_refused() {
    let (s, _d) = store();
    let (_f, problems) = s
        .describe("import os\n@semantic('x', root=True)\ndef f(ctx):\n    narrate(ctx, 'x')\n    return {'answer': open('/etc/passwd').read()}\n")
        .await
        .unwrap();
    assert!(problems.iter().any(|p| p.contains("os")), "{problems:?}");
    assert!(problems.iter().any(|p| p.contains("open")), "{problems:?}");
    let mut e = entry();
    e.code = "import os\n@semantic('x', root=True)\ndef f(ctx):\n    narrate(ctx, 'x')\n    return {'answer': 1}\n".into();
    e.root = "f".into();
    assert!(s.invoke(&e, &BTreeMap::new()).await.is_err());
}

#[tokio::test]
async fn compiles_to_a_module_that_answers_without_a_model() {
    let (s, d) = store();
    let mut e = entry();
    e.status = Status::Trusted;
    s.insert(e, 10);
    let (module, n, skipped) = s.compile(false).await.unwrap();
    assert!(
        skipped.as_object().is_none_or(|o| o.is_empty()),
        "{skipped}"
    );
    assert_eq!(n, 1);
    assert!(!module.contains("do(ctx"), "do() calls must be resolved");
    let path = d.path().join("arc_cache.py");
    std::fs::write(&path, &module).unwrap();
    let q2 = Q.replace("5 notebooks", "2 notebooks");
    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(&path)
        .arg(&q2)
        .output()
        .unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    assert_eq!(
        v["answer"],
        json!(74),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A callee given no payload reads its caller's inputs (the context stack); a near description among the given ones must
/// not be taken for an exact one elsewhere. The compiled form must forward those inputs and give the same answer.
const FORWARD: &str = r#"
@semantic("Calculate total cost of pencils and notebooks")
def cost(ctx):
    pp = load(ctx, "price per pencil in dollars", float)
    n = load(ctx, "number of pencils purchased", int)
    np_ = load(ctx, "price per notebook in dollars", float)
    m = load(ctx, "number of notebooks purchased", int)
    narrate(ctx, f"{n} x {pp} + {m} x {np_}")
    return {"total_cost": n * pp + m * np_}

@semantic("Compute change received after purchase", root=True)
def change(ctx):
    paid = load(ctx, "amount paid in dollars", int)
    c = do(ctx, "Calculate total cost of pencils and notebooks", {})
    total = load(c, "total_cost", float)
    narrate(ctx, f"{paid} - {total}")
    return {"answer": paid - total}
"#;

#[tokio::test]
async fn empty_payload_reads_the_callers_inputs_in_both_forms() {
    let (s, d) = store();
    let mut e = entry();
    e.code = FORWARD.into();
    e.root = "change".into();
    e.status = Status::Trusted;
    e.params = vec![
        Param {
            name: "pp".into(),
            ty: "float".into(),
            description: "price per pencil in dollars".into(),
            value: "3".into(),
        },
        Param {
            name: "np".into(),
            ty: "float".into(),
            description: "price per notebook in dollars".into(),
            value: "7".into(),
        },
        Param {
            name: "n".into(),
            ty: "int".into(),
            description: "number of pencils purchased".into(),
            value: "4".into(),
        },
        Param {
            name: "m".into(),
            ty: "int".into(),
            description: "number of notebooks purchased".into(),
            value: "5".into(),
        },
        Param {
            name: "paid".into(),
            ty: "int".into(),
            description: "amount paid in dollars".into(),
            value: "100".into(),
        },
    ];
    e.template = synthesize_template(Q, &e.params).unwrap();
    s.insert(e.clone(), 10);
    let q2 = Q
        .replace("at 3 dollars", "at 2 dollars")
        .replace("at 7 dollars", "at 9 dollars")
        .replace("4 pencils", "6 pencils")
        .replace("5 notebooks", "3 notebooks")
        .replace("100 dollar", "50 dollar");
    let vals = match_template(&e, &q2).unwrap();
    assert_eq!(s.invoke(&e, &vals).await.unwrap().answer, "11");
    let (module, n, _) = s.compile(false).await.unwrap();
    assert_eq!(n, 1);
    let path = d.path().join("arc_fwd.py");
    std::fs::write(&path, &module).unwrap();
    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(&path)
        .arg(&q2)
        .output()
        .unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    assert_eq!(
        v["answer"],
        json!(11.0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn contrast_words_separate_near_questions() {
    let p = "A rectangle is 12 cm long and 7 cm wide. What is its perimeter in centimetres?";
    assert!(contrast_conflict(
        p,
        "A rectangle is 12 cm long and 7 cm wide. What is its area in square centimetres?"
    )
    .is_some());
    assert!(contrast_conflict(
        p,
        "What is the perimeter, in centimetres, of a rectangle 12 cm long and 7 cm wide?"
    )
    .is_none());
    let o = "What is the sum of the first 50 positive odd integers?";
    assert!(
        contrast_conflict(o, "What is the sum of the first 50 positive even integers?").is_some()
    );
    assert!(contrast_conflict(
        o,
        "Add up the first 50 odd positive whole numbers. What do you get?"
    )
    .is_none());
}

/// A local variable named like a semantic function (seen in a built entry) must not be renamed with it.
#[tokio::test]
async fn a_local_named_like_a_function_compiles() {
    let (s, d) = store();
    let mut e = entry();
    e.code = CODE
        .replace("    a = do(", "    item_cost = do(")
        .replace("load(a, \"cost\", int)", "load(item_cost, \"cost\", int)");
    e.status = Status::Trusted;
    s.insert(e, 10);
    let (module, n, skipped) = s.compile(false).await.unwrap();
    assert_eq!(n, 1, "{skipped}");
    let path = d.path().join("arc_local.py");
    std::fs::write(&path, &module).unwrap();
    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(&path)
        .arg(Q)
        .output()
        .unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    assert_eq!(
        v["answer"],
        json!(53),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_template_of_placeholders_matches_nothing() {
    assert!(template_is_specific(&entry().template));
    assert!(!template_is_specific("{question}"));
    assert!(!template_is_specific("Solve {problem} now"));
    let mut e = entry();
    e.template = "{q}".into();
    e.params = vec![Param {
        name: "q".into(),
        ty: "str".into(),
        description: "the question".into(),
        value: Q.into(),
    }];
    assert!(match_template(&e, "What is the capital of France?").is_none());
}
