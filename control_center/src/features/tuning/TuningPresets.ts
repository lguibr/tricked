export const applyPresetToGroup = (
  conf: Record<string, any>,
  idx: number,
  level: number,
) => {
  const lIdx = level - 1;
  if (idx === 0) {
    // Neural Architecture (Single values in tuning)
    const blocksMap = [2, 4, 10, 15, 20];
    const channelsMap = [32, 64, 128, 256, 512];
    conf.resnetBlocks = blocksMap[lIdx];
    conf.resnetChannels = channelsMap[lIdx];
  } else if (idx === 1) {
    // MDP & Value Estimation (Bounds)
    const maxDiscount = [0.95, 0.99, 0.999, 0.999, 0.999];
    const minDiscount = [0.9, 0.9, 0.9, 0.95, 0.98];
    conf.discount_factor = { min: minDiscount[lIdx], max: maxDiscount[lIdx] };
    const maxLambda = [0.9, 0.95, 0.99, 1.0, 1.0];
    const minLambda = [0.5, 0.8, 0.9, 0.95, 0.95];
    conf.td_lambda = { min: minLambda[lIdx], max: maxLambda[lIdx] };
  } else if (idx === 2) {
    // Search Dynamics (Bounds)
    const maxSims = [100, 400, 1000, 1500, 2000];
    const minSims = [10, 50, 100, 400, 800];
    conf.simulations = { min: minSims[lIdx], max: maxSims[lIdx] };
    const maxGumbel = [8, 16, 32, 48, 64];
    const minGumbel = [4, 4, 8, 16, 24];
    conf.max_gumbel_k = { min: minGumbel[lIdx], max: maxGumbel[lIdx] };
  } else if (idx === 3) {
    // Optimization (Bounds)
    const maxLr = [0.1, 0.05, 0.01, 0.005, 0.001];
    const minLr = [0.01, 0.005, 0.001, 0.0005, 0.0001];
    conf.lr_init = { min: minLr[lIdx], max: maxLr[lIdx] };

    const maxDecay = [0.1, 0.05, 0.01, 0.005, 0.001];
    const minDecay = [0.0, 0.0, 0.0, 0.0, 0.0];
    conf.weight_decay = { min: minDecay[lIdx], max: maxDecay[lIdx] };

    const maxBatch = [256, 1024, 2048, 4096, 4096];
    const minBatch = [64, 128, 512, 1024, 2048];
    conf.train_batch_size = { min: minBatch[lIdx], max: maxBatch[lIdx] };
  } else if (idx === 4) {
    // Systems (Bounds)
    const maxWorkers = [4, 8, 32, 64, 128];
    const minWorkers = [1, 2, 4, 16, 32];
    conf.num_processes = { min: minWorkers[lIdx], max: maxWorkers[lIdx] };
  }
};
