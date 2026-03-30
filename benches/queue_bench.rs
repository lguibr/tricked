use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use std::thread;
use tricked_engine::mcts::EvalReq;
use tricked_engine::queue::FixedInferenceQueue;

pub fn bench_queue_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossbeam_contention");
    group.sample_size(50);

    group.bench_function("fixed_inference_queue_32_threads", |b| {
        b.iter(|| {
            let queue = FixedInferenceQueue::new(16384, 32);
            let mut handles = vec![];

            // Simulating 32 Self-Play Workers hammering the queue simultaneously
            for worker_id in 0..32 {
                let q = Arc::clone(&queue);
                let (tx, _) = crossbeam_channel::unbounded();
                handles.push(thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = q.push(
                            worker_id,
                            EvalReq {
                                is_initial: true,
                                state_feat: None,
                                piece_action: 0,
                                piece_id: 0,
                                node_index: 0,
                                worker_id,
                                parent_cache_index: 0,
                                leaf_cache_index: 0,
                                evaluation_request_transmitter: tx.clone(),
                            },
                        );
                    }
                }));
            }

            // Simulating the Inference Thread popping
            let mut popped = 0;
            while popped < 3200 {
                if let Ok(batch) =
                    queue.pop_batch_timeout(1024, std::time::Duration::from_millis(10))
                {
                    popped += batch.len();
                }
            }

            for handle in handles {
                handle.join().unwrap();
            }
            black_box(popped);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_queue_contention);
criterion_main!(benches);
