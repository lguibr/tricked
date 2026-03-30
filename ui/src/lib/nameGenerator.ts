export type TraitCategory = 'low' | 'med_low' | 'med' | 'med_high' | 'high';

const TRAITS: Record<TraitCategory, string[]> = {
    low: ['Hasty', 'Reckless', 'Impulsive', 'Rash', 'Instinctive'],
    med_low: ['Cautious', 'Steady', 'Vigilant', 'Alert', 'Watchful'],
    med: ['Brave', 'Adventurous', 'Bold', 'Daring', 'Courageous'],
    med_high: ['Wise', 'Calculating', 'Tactical', 'Strategic', 'Methodical'],
    high: ['Omniscient', 'Prophetic', 'Prescient', 'Clairvoyant', 'All-Seeing']
};

export type CreatureCategory = 'tiny' | 'small' | 'medium' | 'large' | 'huge' | 'gargantuan';

const CREATURES: Record<CreatureCategory, string[]> = {
    tiny: ['Pixies', 'Sprites', 'Imps', 'Familiars', 'Homunculi', 'Pseudodragons'],
    small: ['Goblins', 'Kobolds', 'Gnomes', 'Halflings', 'Gremlins'],
    medium: ['Orcs', 'Humans', 'Elves', 'Dwarves', 'Skeletons', 'Zombies'],
    large: ['Ogres', 'Trolls', 'Griffins', 'Minotaurs', 'Manticores', 'Owlbears'],
    huge: ['Giants', 'Treants', 'Behemoths', 'Cyclopes', 'Hydras', 'Wyverns'],
    gargantuan: ['Dragons', 'Krakens', 'Leviathans', 'Tarrasques', 'Rocs', 'Titans']
};

export type ScaleCategory = 'solo' | 'pair' | 'squad' | 'swarm' | 'horde' | 'legion';

const SCALES: Record<ScaleCategory, string> = {
    solo: 'Solo',
    pair: 'Pair',
    squad: 'Squad',
    swarm: 'Swarm',
    horde: 'Horde',
    legion: 'Legion'
};

export function getExplorationTrait(simulations: number): string {
    let category: TraitCategory = 'med';
    if (simulations < 64) category = 'low';
    else if (simulations <= 128) category = 'med_low';
    else if (simulations <= 256) category = 'med';
    else if (simulations <= 1024) category = 'med_high';
    else category = 'high';

    const arr = TRAITS[category];
    const hash = (simulations * 13) % arr.length;
    return arr[hash];
}

export function getCreatureName(hidden_dimension_size: number): string {
    let category: CreatureCategory = 'medium';
    if (hidden_dimension_size <= 32) category = 'tiny';
    else if (hidden_dimension_size <= 64) category = 'small';
    else if (hidden_dimension_size <= 128) category = 'medium';
    else if (hidden_dimension_size <= 256) category = 'large';
    else if (hidden_dimension_size <= 512) category = 'huge';
    else category = 'gargantuan';

    const arr = CREATURES[category];
    const hash = (hidden_dimension_size * 17) % arr.length;
    return arr[hash];
}

export function getScalePrefix(num_processes: number): string {
    let category: ScaleCategory = 'squad';
    if (num_processes <= 1) category = 'solo';
    else if (num_processes <= 2) category = 'pair';
    else if (num_processes <= 8) category = 'squad';
    else if (num_processes <= 32) category = 'swarm';
    else if (num_processes <= 128) category = 'horde';
    else category = 'legion';

    return SCALES[category];
}

export function generateExperimentName(num_processes: number, simulations: number, hidden_dimension_size: number): string {
    return `${getScalePrefix(num_processes)} ${getExplorationTrait(simulations)} ${getCreatureName(hidden_dimension_size)}`;
}
