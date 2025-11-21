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

def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
    def add_walls(self, prob=0.03, seed=0):
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
        self.task = None
    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

# --- Main Execution ---
H, W = 16, 16
world = GridWorld(H, W)
world.add_walls(prob=0.03, seed=21)

ffs = [Agent(0, (0, 0)), Agent(1, (0, W-1)), Agent(2, (H-1, W-1))]
fires = {(random.randint(3, H-4), random.randint(3, W-4)) for _ in range(6)}

t = 0
while fires and t < 300:
    # Greedy assignment
    for f in ffs:
        if f.task is None and fires:
            f.task = min(fires, key=lambda x: manhattan(f.pos, x))
            
    for f in ffs:
        if f.task:
            p = bfs(f.pos, {f.task}, H, W, world.grid)
            if p: f.path = p
        f.step()
        
        if f.pos == f.task and f.pos in fires:
            fires.remove(f.pos)
            f.task = None

    # Fire Spread
    newfires = set(fires)
    for fire in list(fires):
        for nb in neighbors(fire, H, W, world.grid):
            if random.random() < 0.08: # Spread probability
                newfires.add(nb)
    fires = newfires
    t += 1

print("Remaining fires approx:", len(fires))

# Visualization
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for fr in fires: canvas[fr] = np.array([0.9, 0.3, 0.2])
for f in ffs:
    for h in f.history: canvas[h] = np.array([0.8, 0.9, 0.9])
    r, c = f.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (f.id % 3), 0.2 + 0.4 * ((f.id + 1) % 3), 0.3 + 0.3 * ((f.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Cooperative Firefighters Final')
plt.axis('off')
plt.show()