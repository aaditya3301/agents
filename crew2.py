import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def in_bounds(p, H, W): return 0 <= p[0] < H and 0 <= p[1] < W

def neighbors(p, H, W, grid):
    for dr, dc in DIRS:
        nr, nc = p[0] + dr, p[1] + dc
        if in_bounds((nr, nc), H, W) and grid[nr, nc] == 0:
            yield (nr, nc)

def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, H, W, grid):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start, None))
    came = {}
    g = {start: 0}
    
    while open_set:
        f, dist, cur, parent = heapq.heappop(open_set)
        if cur in came: continue
        came[cur] = parent
        
        if cur == goal:
            path = []
            p = cur
            while p:
                path.append(p)
                p = came[p] if p in came else None
            return path[::-1]
            
        for nb in neighbors(cur, H, W, grid):
            nd = dist + 1
            if nb not in g or nd < g[nb]:
                g[nb] = nd
                heapq.heappush(open_set, (nd + manhattan(nb, goal), nd, nb, cur))
    return None

class GridWorld:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=int)
        
    def add_walls(self, prob=0.08, seed=0):
        random.seed(seed)
        for r in range(self.H):
            for c in range(self.W):
                if random.random() < prob: self.grid[r, c] = 1

class Agent:
    def __init__(self, id, start):
        self.id = id
        self.pos = start
        self.history = [start]
        self.path = [start]
        
    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

H, W = 12, 12
world = GridWorld(H, W)
world.add_walls(prob=0.08, seed=7)

dirty = [(r, c) for r in range(H) for c in range(W) if world.grid[r, c] == 0 and random.random() < 0.25]
dirty_set = set(dirty)

a1 = Agent(0, (0, 0))
a2 = Agent(1, (H-1, W-1))
agents = [a1, a2]

# Divide regions
region1 = {(r, c) for r in range(H) for c in range(W) if c < W // 2}
region2 = {(r, c) for r in range(H) for c in range(W) if c >= W // 2}

t = 0
while dirty_set and t < 500:
    t += 1
    for a, region in [(a1, region1), (a2, region2)]:
        region_targets = [d for d in dirty_set if d in region]
        if not region_targets:
            region_targets = list(dirty_set) # Help other region if done
        if not region_targets: continue
        
        target = min(region_targets, key=lambda x: manhattan(a.pos, x))
        p = astar(a.pos, target, H, W, world.grid)
        if p: a.path = p
        
        a.step()
        if a.pos in dirty_set: dirty_set.remove(a.pos)

# Visualize
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for d in dirty: canvas[d] = np.array([0.6, 0.3, 0.1])
for a in agents:
    for p in a.history: canvas[p] = np.array([0.8, 0.9, 0.9])
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Cleaning Crew Final')
plt.axis('off')
plt.show()