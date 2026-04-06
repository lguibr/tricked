use arc_swap::ArcSwap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tch::{nn, Device};
use tricked_engine::net::MuZeroNet;

#[test]
fn test_arcswap_double_buffering_latency() {
    let device = Device::Cpu;
    let vs_a = nn::VarStore::new(device);
    let vs_b = nn::VarStore::new(device);

    // Create tiny MuZeroNet instances to minimize allocation overhead during setup,
    // we only care about the ArcSwap pointer swap latency for this test.
    let net_a = Arc::new(MuZeroNet::new(&vs_a.root(), 16, 1, 288, 288, 20, 64));
    let net_b = Arc::new(MuZeroNet::new(&vs_b.root(), 16, 1, 288, 288, 20, 64));

    let shared_arc = Arc::new(ArcSwap::from(net_a));

    // Simulate 8 inference threads doing wait-free loads continuously
    let mut handles = vec![];
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    for _ in 0..8 {
        let arc_clone = Arc::clone(&shared_arc);
        let flag = Arc::clone(&stop_flag);
        handles.push(thread::spawn(move || {
            let mut loads = 0;
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                let _loaded = arc_clone.load();
                loads += 1;
            }
            std::hint::black_box(loads);
        }));
    }

    // Give readers a moment to spin up
    thread::sleep(std::time::Duration::from_millis(50));

    // Measure the latency of 100 swaps (simulating 100 training steps' weight syncs)
    let start = Instant::now();
    for _ in 0..100 {
        shared_arc.store(Arc::clone(&net_b));
    }
    let duration = start.elapsed();

    // Signal readers to stop
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    // The total time for 100 stores while 8 readers are hammering it should be minimal.
    let avg_latency_ms = duration.as_secs_f64() / 100.0 * 1000.0;
    println!(
        "Average ArcSwap Store Latency (with 8 active readers): {:.4}ms",
        avg_latency_ms
    );

    // Test that the swap is wait-free (< 1.0ms on any modern CPU even under high contention)
    assert!(
        avg_latency_ms < 1.0,
        "ArcSwap block time is too high: {:.4}ms. Double buffering should be virtually wait-free.",
        avg_latency_ms
    );
}
