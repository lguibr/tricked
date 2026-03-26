pub struct TrajectoryData<'a> {
    pub difficulty: i32,
    pub score: f32,
    pub step: u64,
    pub boards: &'a [u128],
    pub available: &'a [i32],
    pub actions: &'a [i64],
    pub piece_ids: &'a [i64],
    pub rewards: &'a [f32],
    pub policies: &'a [f32],
    pub values: &'a [f32],
}

#[allow(dead_code)]
pub fn serialize_trajectory(data: TrajectoryData) -> Vec<u8> {
    let mut payload_buffer = Vec::new();
    payload_buffer.extend_from_slice(&data.difficulty.to_le_bytes());
    payload_buffer.extend_from_slice(&data.score.to_le_bytes());
    payload_buffer.extend_from_slice(&data.step.to_le_bytes());

    let boards_bytes: &[u8] = bytemuck::cast_slice(data.boards);
    let available_bytes: &[u8] = bytemuck::cast_slice(data.available);
    let actions_bytes: &[u8] = bytemuck::cast_slice(data.actions);
    let piece_identifier_bytes: &[u8] = bytemuck::cast_slice(data.piece_ids);
    let rewards_bytes: &[u8] = bytemuck::cast_slice(data.rewards);
    let policies_bytes: &[u8] = bytemuck::cast_slice(data.policies);
    let values_bytes: &[u8] = bytemuck::cast_slice(data.values);

    payload_buffer.extend_from_slice(&(boards_bytes.len() as u64).to_le_bytes());
    payload_buffer.extend_from_slice(&(available_bytes.len() as u64).to_le_bytes());
    payload_buffer.extend_from_slice(&(actions_bytes.len() as u64).to_le_bytes());
    payload_buffer.extend_from_slice(&(piece_identifier_bytes.len() as u64).to_le_bytes());
    payload_buffer.extend_from_slice(&(rewards_bytes.len() as u64).to_le_bytes());
    payload_buffer.extend_from_slice(&(policies_bytes.len() as u64).to_le_bytes());
    payload_buffer.extend_from_slice(&(values_bytes.len() as u64).to_le_bytes());

    payload_buffer.extend_from_slice(boards_bytes);
    payload_buffer.extend_from_slice(available_bytes);
    payload_buffer.extend_from_slice(actions_bytes);
    payload_buffer.extend_from_slice(piece_identifier_bytes);
    payload_buffer.extend_from_slice(rewards_bytes);
    payload_buffer.extend_from_slice(policies_bytes);
    payload_buffer.extend_from_slice(values_bytes);

    payload_buffer
}
