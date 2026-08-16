fn main() {
    let engine = md5_many::Md5Many::new();
    println!("native u32 lanes: {}", engine.lanes());
}
