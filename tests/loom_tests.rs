#[cfg(loom)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tricked_engine::sumtree::SegmentTree;

    #[test]
    fn test_segment_tree_loom() {
        loom::model(|| {
            let tree = Arc::new(SegmentTree::new(4));

            let t1 = tree.clone();
            let handle1 = loom::thread::spawn(move || {
                t1.update(0, 5.0);
            });

            let t2 = tree.clone();
            let handle2 = loom::thread::spawn(move || {
                t2.update(1, 10.0);
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            let total = tree.get_total_priority();
            assert!(
                (total - 15.0).abs() < 1e-5,
                "Race condition detected, sumtree mathematically incorrect! total={}",
                total
            );
        });
    }
}
