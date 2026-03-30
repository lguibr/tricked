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

#[allow(dead_code)]
pub mod u128_string {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(val: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&val.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<u128>().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_flex_compression_ratio() {
        let boards = vec![0u128; 100];
        let available = vec![0i32; 100];
        let actions = vec![0i64; 100];
        let piece_ids = vec![0i64; 100];
        let rewards = vec![0.5f32; 100];
        let policies = vec![0.0f32; 100 * 288];
        let values = vec![0.1f32; 100];

        let traj = TrajectoryData {
            difficulty: 6,
            score: 100.0,
            step: 50,
            boards: &boards,
            available: &available,
            actions: &actions,
            piece_ids: &piece_ids,
            rewards: &rewards,
            policies: &policies,
            values: &values,
        };

        let raw_bytes = serialize_trajectory(traj);
        let compressed = lz4_flex::compress_prepend_size(&raw_bytes);

        let ratio = raw_bytes.len() as f64 / compressed.len() as f64;
        println!("Raw payload size: {} bytes", raw_bytes.len());
        println!("LZ4 compressed size: {} bytes", compressed.len());
        println!("Compression Ratio: {:.2}x", ratio);

        // Expect at least 3x compression for sparse policy/board arrays
        assert!(ratio > 3.0, "LZ4 did not compress efficiently");
    }
}
