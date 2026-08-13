fn main() {
    let engine = fearless_md5::Md5Many::new();
    println!("native u32 lanes: {}", engine.lanes());
}
