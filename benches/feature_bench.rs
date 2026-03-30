use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tricked_engine::board::GameStateExt;
use tricked_engine::features::extract_feature_native;

pub fn criterion_benchmark(c: &mut Criterion) {
    let state = GameStateExt::new(Some([0, 1, 2]), 0u128, 0, 5, 0);
    let history = vec![1u128, 2u128, 3u128, 4u128, 5u128];

    c.bench_function("extract_feature_native_copies", |b| {
        let mut slice = vec![0.0; 20 * 128];
        b.iter(|| {
            extract_feature_native(
                black_box(&mut slice),
                black_box(state.board_bitmask_u128),
                black_box(&state.available),
                black_box(&history),
                black_box(&[]),
                black_box(5),
            );
            black_box(&slice);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
