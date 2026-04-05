import json

# Grid structure
rows = [9, 11, 13, 15, 15, 13, 11, 9]

cells = []
idx = 0

s = 20  # side length multiplier
h = 17.3205  # height (sqrt(3)/2 * s)

for r, count in enumerate(rows):
    # center of the row in terms of triangles
    # the x-offset of the first triangle depends on the row
    # To keep it symmetrical, the center of each row is x = 0.
    start_x = -(count * (s / 2)) / 2 + (s / 4)

    # In the top half (rows 0,1,2,3), the left boundary moves left.
    # A DOWN triangle (\/) has a flat top. An UP triangle (/\) has a flat bottom.
    for c in range(count):
        # Is it pointing up or down?
        # Let's see: row 0 has length 9. If it starts with DOWN, ends with DOWN.
        # Actually, looking at typical hex hexboards, the top edge has DOWN triangles.
        points_up = c % 2 != 0

        # In the bottom half (rows 4,5,6,7), row 4 has 15 triangles.
        # Does it start with DOWN or UP? If it's a perfect hexagon, row 3 ends with DOWN, row 4 should start with UP to form the left straight edge.
        # Let's adjust UP/DOWN based on a simple uniform grid:
        # A triangle at col `col` and row `row` points UP if (col + row) is odd.
        # Let's test this:
        # row 0, col 0 (adjusted): if it needs to be DOWN (0), then even sum = DOWN.

        # Center x
        x = start_x + c * (s / 2)
        y = (r - 4 + 0.5) * h

        cells.append(
            {
                "id": idx,
                "row": r,
                "col": c,
                "x": round(x, 2),
                "y": round(y, 2),
                "up": points_up if r < 4 else not points_up,
                # wait, if we explicitly define up:
            }
        )
        idx += 1

print(json.dumps(cells))
