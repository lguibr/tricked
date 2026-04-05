const fs = require('fs');

const data = JSON.parse(fs.readFileSync('./control_center/src/lib/game/masks.json'));
const gridCoords = JSON.parse(fs.readFileSync('./control_center/src/lib/game/gridCoords.json'));

const rotationMap = (() => {
    const map = new Array(96).fill(0);
    const h = 17.320508;
    const sin60 = Math.sqrt(3) / 2;
    const cos60 = 0.5;

    const getTrueCentroid = (c) => ({
        x: c.x,
        y: c.up ? c.y - h / 6 : c.y + h / 6,
    });

    const coords = gridCoords;
    coords.forEach((c) => {
        const p = getTrueCentroid(c);
        const rx = p.x * cos60 - p.y * sin60;
        const ry = p.x * sin60 + p.y * cos60;

        let bestId = -1;
        let bestDist = Infinity;
        coords.forEach((c2) => {
            const p2 = getTrueCentroid(c2);
            const d = (rx - p2.x) ** 2 + (ry - p2.y) ** 2;
            if (d < bestDist) {
                bestDist = d;
                bestId = c2.id;
            }
        });
        map[c.id] = bestId;
    });
    return map;
})();

function rotateMask60(mask, rotations) {
    let result = BigInt(mask);
    rotations = ((Number(rotations) % 6) + 6) % 6;
    for (let r = 0; r < rotations; r++) {
        let next = 0n;
        for (let i = 0n; i < 96n; i++) {
            let bitTarget = rotationMap[Number(i)];
            if (bitTarget === undefined) bitTarget = 0;
            let v = BigInt(bitTarget);
            if ((result & (1n << i)) !== 0n) {
                next |= (1n << v);
            }
        }
        result = next;
    }
    return result;
}

for (let rot = 1; rot < 6; rot++) {
    let outOfSetCount = 0;
    let mappingArray = new Array(48).fill(-1);

    for (let p = 0; p < data.standard.length; p++) {
        let foundInPiece = -1;

        for (let c = 0; c < 96; c++) {
            let [m0, m1] = data.standard[p][c];
            if (m0 !== 0 || m1 !== 0) {
                let baseMask = BigInt.asUintN(64, BigInt(m0)) | (BigInt.asUintN(64, BigInt(m1)) << 64n);
                let rotMask = rotateMask60(baseMask, rot);

                for (let p2 = 0; p2 < data.standard.length; p2++) {
                    for (let c2 = 0; c2 < 96; c2++) {
                        let [sm0, sm1] = data.standard[p2][c2];
                        let searchMask = BigInt.asUintN(64, BigInt(sm0)) | (BigInt.asUintN(64, BigInt(sm1)) << 64n);
                        if (searchMask === rotMask) {
                            foundInPiece = p2;
                            break;
                        }
                    }
                    if (foundInPiece !== -1) break;
                }
                if (foundInPiece !== -1) break;
            }
        }

        if (foundInPiece === -1) {
            outOfSetCount++;
        } else {
            mappingArray[p] = foundInPiece;
        }
    }

    console.log(`Rotation ${rot * 60} degrees: Total out of set: ${outOfSetCount}`);
}
