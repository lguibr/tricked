pub fn serialize_trajectory(
    difficulty: i32,
    score: f32,
    step: u64,
    ep_boards: &[u128],
    ep_available: &[i32],
    ep_actions: &[i64],
    ep_p_ids: &[i64],
    ep_rewards: &[f32],
    ep_policies: &[f32],
    ep_values: &[f32],
) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&difficulty.to_le_bytes());
    payload.extend_from_slice(&score.to_le_bytes());
    payload.extend_from_slice(&step.to_le_bytes());

    let b_bytes = unsafe {
        std::slice::from_raw_parts(ep_boards.as_ptr() as *const u8, ep_boards.len() * 16)
    };
    let av_bytes: &[u8] = bytemuck::cast_slice(ep_available);
    let a_bytes: &[u8] = bytemuck::cast_slice(ep_actions);
    let pid_bytes: &[u8] = bytemuck::cast_slice(ep_p_ids);
    let r_bytes: &[u8] = bytemuck::cast_slice(ep_rewards);
    let pol_bytes: &[u8] = bytemuck::cast_slice(ep_policies);
    let v_bytes: &[u8] = bytemuck::cast_slice(ep_values);

    payload.extend_from_slice(&(b_bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(av_bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(a_bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(pid_bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(r_bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(pol_bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(v_bytes.len() as u64).to_le_bytes());

    payload.extend_from_slice(b_bytes);
    payload.extend_from_slice(av_bytes);
    payload.extend_from_slice(a_bytes);
    payload.extend_from_slice(pid_bytes);
    payload.extend_from_slice(r_bytes);
    payload.extend_from_slice(pol_bytes);
    payload.extend_from_slice(v_bytes);

    payload
}
