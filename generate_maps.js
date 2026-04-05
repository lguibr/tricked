const fs = require('fs');
const gridCoords = JSON.parse(fs.readFileSync('./control_center/src/lib/game/gridCoords.json'));

const h = 17.320508;
const sin60 = Math.sqrt(3) / 2;
const cos60 = 0.5;

const getTrueCentroid = (c) => ({
    x: c.x,
    y: c.up ? c.y - h / 6 : c.y + h / 6,
});

let rot_map = new Array(96).fill(0);
let flip_map = new Array(96).fill(0);

gridCoords.forEach((c) => {
    const p = getTrueCentroid(c);

    // 60 degree rotation clockwise
    const rx = p.x * cos60 - p.y * sin60;
    const ry = p.x * sin60 + p.y * cos60;

    // Reflection across Y axis (x -> -x)
    const fx = -p.x;
    const fy = p.y;

    let bestRotId = -1, bestRotDist = Infinity;
    let bestFlipId = -1, bestFlipDist = Infinity;

    gridCoords.forEach((c2) => {
        const p2 = getTrueCentroid(c2);

        let dRot = (rx - p2.x) ** 2 + (ry - p2.y) ** 2;
        if (dRot < bestRotDist) { bestRotDist = dRot; bestRotId = c2.id; }

        let dFlip = (fx - p2.x) ** 2 + (fy - p2.y) ** 2;
        if (dFlip < bestFlipDist) { bestFlipDist = dFlip; bestFlipId = c2.id; }
    });

    rot_map[c.id] = bestRotId;
    flip_map[c.id] = bestFlipId;
});

console.log("pub const ROT_MAP_60: [usize; 96] = [");
console.log("    " + rot_map.join(", "));
console.log("];\n");

console.log("pub const FLIP_MAP_Y: [usize; 96] = [");
console.log("    " + flip_map.join(", "));
console.log("];");
