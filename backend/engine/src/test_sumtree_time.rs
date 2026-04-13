use tricked_engine::sumtree::SegmentTree;
use std::time::Instant;

fn main() {
    let tree = SegmentTree::new(2_000_000);
    
    // Simulate updating 2048 priorities
    let mut updates = Vec::new();
    for i in 0..2048 {
        updates.push((i * 100, 1.0));
    }

    let start = Instant::now();
    tree.update_batch(&updates);
    let elapsed = start.elapsed().as_nanos();
    
    println!("2048 elements took {} nanos ({} microseconds)", elapsed, elapsed / 1000);
}
