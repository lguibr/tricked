# Tricked: Game Mechanics & Engine Implementation

## 1. Overview
**Tricked** is a topological board puzzle game played on a **96-triangle hexagonal grid**. The objective is to place generated poly-triangle pieces onto the board to complete continuous lines, which then clear and grant points.

## 2. The Grid and Coordinate System
The board is a regular hexagon composed of exactly 96 smaller equilateral triangles (equivalent to a side length of 4 triangle units). 

![Grid Symmetry & Rhombus Axes](grid.png)

**Rhombus Coordinate Cube System:**
To elegantly handle geometric operations, connectivity, and line-clearing validation, the grid uses a **Rhombus Coordinate Cube** system. 
- Adjacent pairs of triangles are treated conceptually as rhombuses, representing the visible faces of 3D cubes in an isometric projection.
- This results in a 3-axis ($X, Y, Z$) coordinate addressing system.
- It defines three distinct axes for straight "lines" across the board, radically simplifying the math required to detect continuous occupied lines from edge to edge.

## 3. The 3-Piece Buffer
The piece selection and turn system is managed via a **3-Piece Buffer**:
- **Generation:** The game populates the buffer with up to 3 randomly generated pieces (various shapes made of connected triangles).
- **Placement:** The player (or agent) selects and places pieces into unoccupied spaces on the grid. In our AI-training version, this revolves around absolute coordinate placement rather than relying on human UI drag-and-drop mechanics.
- **Refresh:** Once all 3 pieces from the buffer have been successfully placed on the board, the game generates a fresh batch of 3 pieces.
- **Monochrome Constraint:** In the actual training environment we will play and optimize for, pieces **do not have colors**. They function purely as binary topological obstacles (occupied vs. empty), forcing the AI to rely entirely on shape and spatial reasoning.

## 4. Line Clearing and Scoring Math
The primary method of survival and generating reward signals is line clearing.

- **Line Definition:** A "line" is a continuous sequence of triangles stretching across the grid from one edge to the opposite edge along one of the three coordinate axes.
- **Clear Execution:** When an entire line is occupied, the triangles in that line are cleared (reset to empty).
- **Base Scoring:** You receive **2 points for every triangle** present in the cleared line.
- **Combo Intersection Multiplier:** The scoring system highly rewards multi-line combos. If a single piece placement completes *multiple overlapping lines* simultaneously, any triangle that belongs to more than one of the cleared lines is **scored independently for each line it belongs to**. For instance, an overlapping intersection triangle in a 2-line clear will grant $2 \times 2 = 4$ points (or $2 \times 3 = 6$ points for a 3-line cross), drastically amplifying the reward signal for strategic placements.

## 5. Terminal State (Game Over)
The episode ends (Game Over) when the board is too cluttered to legally accommodate *any* of the pieces currently residing in the 3-piece buffer. Survival relies on maximizing buffer throughput and prioritizing multi-line clears to free up topological real estate.
