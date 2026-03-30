use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tricked_engine::sumtree::ShardedPrioritizedReplay;

pub fn bench_per_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_sampling");
    group.sample_size(100);

    group.bench_function("sharded_per_sample_1024", |b| {
        let per = ShardedPrioritizedReplay::new(1_000_000, 1.0, 1.0, 8);

        let indices: Vec<usize> = (0..10_000).collect();
        let priorities: Vec<f64> = (0..10_000).map(|i| (i % 100) as f64).collect();
        per.add_batch(&indices, &priorities);

        b.iter(|| {
            let sample = per.sample(1024, 1_000_000, 1.0);
            black_box(sample);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_per_sampling);
criterion_main!(benches);
