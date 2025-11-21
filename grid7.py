import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def in_bounds(p, H, W): return 0 <= p[0] < H and 0 <= p[1] < W

def neighbors(p, H, W, grid):
    for dr, dc in DIRS:
        nr, nc = p[0] + dr, p[1] + dc
        if in_bounds((nr, nc), H, W) and grid[nr, nc] == 0: yield (nr, nc)

def bfs(start, goals, H, W, grid):
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur in goals:
            path = []
            p = cur
            while p:
                path.append(p)
                p = parent[p]
            return path[::-1]
        for nb in neighbors(cur, H, W, grid):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    return None

class GridWorld:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=int)
    def add_walls(self, prob=0.05, seed=0):
        random.seed(seed)
        for r in range(self.H):
            for c in range(self.W):
                if random.random() < prob: self.grid[r, c] = 1

class Agent:
    def __init__(self, id, start):
        self.id = id
        self.pos = start
        self.path = [start]
        self.history = [start]
    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

# --- Main Execution ---
H, W = 12, 12
world = GridWorld(H, W)
world.add_walls(prob=0.05, seed=8)

painters = [Agent(0, (0, 0)), Agent(1, (H-1, W-1))]
to_paint = {(r, c) for r in range(H) for c in range(W) if world.grid[r, c] == 0 and random.random() < 0.35}

# Checkerboard partition to avoid conflict
region1 = {p for p in to_paint if (p[0] + p[1]) % 2 == 0}
region2 = to_paint - region1
rem1, rem2 = set(region1), set(region2)

t = 0
while (rem1 or rem2) and t < 500:
    t += 1
    for idx, (a, rem) in enumerate([(painters[0], rem1), (painters[1], rem2)]):
        if not rem: continue
        
        # Find nearest unpainted cell in own region
        target = min(rem, key=lambda x: abs(x[0] - a.pos[0]) + abs(x[1] - a.pos[1]))
        p = bfs(a.pos, {target}, H, W, world.grid)
        if p: a.path = p
        
        a.step()
        if a.pos in rem:
            rem.remove(a.pos)

print(f"Painted approx {len(to_paint) - len(rem1) - len(rem2)} cells.")

# Visualization
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for p in to_paint: canvas[p] = np.array([0.95, 0.9, 0.7]) # base paint color
for a in painters:
    for h in a.history: canvas[h] = np.array([0.8, 0.9, 0.9]) # path
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Grid Painting Final')
plt.axis('off')
plt.show()