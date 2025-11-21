import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Logic Core ---
def astar(start, goal, H, W, grid):
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start, goal), 0, start, None)]
    came_from = {}
    g = {start: 0}
    while open_set:
        _, dist, cur, parent = heapq.heappop(open_set)
        if cur in came_from: continue
        came_from[cur] = parent
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = came_from[cur] if cur in came_from else None
            return path[::-1]
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<H and 0<=nc<W and grid[nr, nc]==0:
                if dist+1 < g.get((nr, nc), float('inf')):
                    g[(nr, nc)] = dist+1
                    heapq.heappush(open_set, (dist+1+h((nr, nc), goal), dist+1, (nr, nc), cur))
    return [start]

# --- 2. Setup ---
H, W = 12, 12
grid = np.zeros((H, W), dtype=int)
random.seed(5)
for r in range(H):
    for c in range(W):
        if random.random() < 0.12: grid[r, c] = 1

start1, start2 = (1, 1), (1, W-2)
goal1, goal2 = (H-2, 1), (H-2, W-2)

# Plan
p1 = astar(start1, goal1, H, W, grid)
p2 = astar(start2, goal2, H, W, grid)
hist = {0: list(p1), 1: list(p2)}

# Resolve Collisions
t = 0
max_len = max(len(p1), len(p2)) + 20
while t < max_len:
    pos0 = hist[0][t] if t < len(hist[0]) else hist[0][-1]
    pos1 = hist[1][t] if t < len(hist[1]) else hist[1][-1]
    
    if pos0 == pos1 and t > 0: # Collision detected
        # Agent 1 waits
        hist[1].insert(t, hist[1][t-1])
        max_len += 1
    t += 1

# Normalize lengths
final_len = max(len(hist[0]), len(hist[1]))

# --- 3. Animation ---
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#121212')

def render(frame):
    ax.clear()
    ax.axis('off')
    
    # Grid
    canvas = np.zeros((H, W, 3)) + 0.1
    canvas[grid == 1] = [0.05, 0.05, 0.1]
    ax.imshow(canvas, extent=[0, W, H, 0])
    
    # Goals
    ax.scatter(goal1[1], goal1[0], c='#00FF00', marker='x', s=100, linewidth=3, label='Goal 1')
    ax.scatter(goal2[1], goal2[0], c='#FF00FF', marker='x', s=100, linewidth=3, label='Goal 2')

    # Agents
    positions = []
    for i, color in [(0, '#00FFFF'), (1, '#FF00FF')]:
        h = hist[i]
        pos = h[frame] if frame < len(h) else h[-1]
        positions.append(pos)
        
        # Trail
        if frame > 0:
            past = h[:frame+1][-10:]
            py, px = zip(*past)
            ax.plot(px, py, c=color, linewidth=2, alpha=0.5)
            
        ax.scatter(pos[1], pos[0], c=color, s=200, edgecolors='white')

    status = "MOVING"
    if frame >= final_len - 1: status = "ARRIVED"
    if positions[0] == positions[1]: status = "COLLISION (Error)" # Should not happen
    
    ax.set_title(f"PATH PLANNERS | Step: {frame} | {status}", color='white')
    ax.set_ylim(H-0.5, -0.5); ax.set_xlim(-0.5, W-0.5)

anim = FuncAnimation(fig, render, frames=final_len, interval=200, repeat=False)
plt.show()