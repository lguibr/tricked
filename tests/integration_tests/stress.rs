use tricked_engine::mcts::EvaluationRequest;
use crossbeam_channel::unbounded;

#[test]
fn test_transmission_stress_test() {
    let (evaluation_request_transmitter, evaluation_response_receiver) =
        unbounded::<EvaluationRequest>();

    let mut handlers = vec![];
    let num_workers = 10;
    let num_reqs = 100;

    for _w in 0..num_workers {
        let thread_tx = evaluation_request_transmitter.clone();
        handlers.push(std::thread::spawn(move || {
            for _i in 0..num_reqs {
                let mailbox = std::sync::Arc::new(tricked_engine::mcts::mailbox::AtomicMailbox::new());
                let req = EvaluationRequest {
                    is_initial: true,
                    board_bitmask: 0,
                    available_pieces: [-1; 3],
                    recent_board_history: [0; 8],
                    history_len: 0,
                    recent_action_history: [0; 4],
                    action_history_len: 0,
                    difficulty: 6,
                    piece_action: 0,
                    piece_id: 0,
                    node_index: 0,
                    generation: 0,
                    worker_id: 0,
                    parent_cache_index: 0,
                    leaf_cache_index: 0,
                    mailbox: mailbox.clone(),
                };
                thread_tx.send(req).unwrap();
                let active_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                let _ = tricked_engine::mcts::mailbox::spin_wait(&mailbox, &active_flag).unwrap();
            }
        }));
    }

    let total_reqs = num_workers * num_reqs;
    for _ in 0..total_reqs {
        let req = evaluation_response_receiver.recv().unwrap();
        req.mailbox
            .write_and_notify(tricked_engine::mcts::EvaluationResponse {
                child_prior_probabilities_tensor: [0.0; 288],
                value: 0.0,
                value_prefix: 0.0,
                node_index: 0,
                generation: 0,
            });
    }

    for h in handlers {
        h.join().unwrap();
    }
    assert!(
        evaluation_response_receiver.is_empty(),
        "Channel should be thoroughly processed"
    );
}
