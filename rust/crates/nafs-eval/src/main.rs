//! nafs-eval CLI: start an evaluation run on a running gateway and print the summary.
//! Example: nafs-eval --gateway http://127.0.0.1:8080 --benchmark smoke_mixed --max-items 6
use clap::Parser;
use serde_json::{json, Value};

#[derive(Parser)]
#[command(about = "Run a Nafs-Aql evaluation through the gateway API")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    gateway: String,
    #[arg(long)]
    benchmark: String,
    #[arg(long, default_value_t = 10)]
    max_items: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "nafs_aql,fast_only,slow_only"
    )]
    systems: Vec<String>,
    #[arg(long)]
    token: Option<String>,
    #[arg(long, default_value_t = 10)]
    poll_s: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let c = reqwest::Client::new();
    let auth = |r: reqwest::RequestBuilder| match &a.token {
        Some(t) => r.bearer_auth(t),
        None => r,
    };
    let run: Value =
        auth(c.post(format!("{}/v1/eval/runs", a.gateway)).json(
            &json!({"benchmark": a.benchmark, "max_items": a.max_items, "systems": a.systems}),
        ))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let id = run["run_id"].as_str().unwrap_or_default().to_string();
    println!("started eval run {id}");
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(a.poll_s)).await;
        let r: Value = auth(c.get(format!("{}/v1/eval/runs/{id}", a.gateway)))
            .send()
            .await?
            .json()
            .await?;
        println!(
            "progress {}/{}",
            r["progress"]["done"], r["progress"]["total"]
        );
        if r["status"] != "running" && r["status"] != "queued" {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"systems": r["systems"], "paired": r["paired"], "routing": r["routing"]})
                )?
            );
            break;
        }
    }
    Ok(())
}
