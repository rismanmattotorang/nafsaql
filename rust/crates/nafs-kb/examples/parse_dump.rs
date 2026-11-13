//! Print what the reader makes of a file (Phase 28 aid): `parse_dump <file> [start_byte] [len]`, or
//! `parse_dump <file> chunks` for the passages with their heading paths and pages.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&a[0]).expect("read");
    let kind = nafs_kb::Kind::from_filename(&a[0]).expect("kind");
    let p = nafs_kb::parse(kind, &bytes).expect("parse");
    if a.get(1).map(String::as_str) == Some("chunks") {
        for c in nafs_kb::chunk(&p, nafs_kb::ChunkOpts::default()) {
            println!(
                "=== {} p{:?} [{}] {} chars\n{}",
                c.ord, c.page, c.heading, c.chars, c.text
            );
        }
        return;
    }
    let t = &p.text;
    let mut s: usize = a
        .get(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0)
        .min(t.len());
    while !t.is_char_boundary(s) {
        s += 1;
    }
    let n: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(4000);
    let mut e = (s + n).min(t.len());
    while !t.is_char_boundary(e) {
        e += 1;
    }
    println!("title: {:?}\n{}", p.title, &t[s..e]);
}
