use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tricked_engine::board::GameStateExt;
use tricked_engine::features::extract_feature_native;

pub fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");
    // Ensure we capture the `< 5 microseconds` requirement visually
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    let state = GameStateExt::new(Some([0, 1, 2]), 0b101010101010101010101010, 50, 6, 0);
    let history_boards = vec![
        0b0101010, 0b1010101, 0b0011001, 0b1100110, 0b0000111, 0b1110000, 0b0101010,
    ];
    let action_history = vec![10, 45, 90, 15];

    group.bench_function("extract_feature_native_current", |b| {
        b.iter(|| {
            let features = extract_feature_native(
                black_box(&state),
                black_box(Some(history_boards.clone())),
                black_box(Some(action_history.clone())),
                black_box(6),
            );
            black_box(features);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_feature_extraction);
criterion_main!(benches);
