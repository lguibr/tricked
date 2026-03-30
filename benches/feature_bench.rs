use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tricked_engine::board::GameStateExt;
use tricked_engine::features::extract_feature_native;

pub fn criterion_benchmark(c: &mut Criterion) {
    let state = GameStateExt::new(Some([0, 1, 2]), 0u128, 0, 5, 0);
    let history = Some(vec![1u128, 2u128, 3u128, 4u128, 5u128]);

    c.bench_function("extract_feature_native_copies", |b| {
        b.iter(|| extract_feature_native(black_box(&state), black_box(history.clone()), None, 5))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
