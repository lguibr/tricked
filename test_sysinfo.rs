use sysinfo::Networks;
fn main() {
    let mut networks = Networks::new_with_refreshed_list();
    std::thread::sleep(std::time::Duration::from_millis(500));
    networks.refresh_list();
    let mut rx = 0;
    for (_, net) in &networks {
        rx += net.received();
    }
    println!("rx: {}", rx);
}
