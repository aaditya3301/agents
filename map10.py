import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# (Include BFS helper)
def bfs(start, goals, H, W, grid):
    q = deque([start]); parent = {start: None}; visited = {start}
    while q:
        cur = q.popleft()
        if cur in goals:
            path = []; 
            while cur: path.append(cur); cur = parent[cur]
            return path[::-1]
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<H and 0<=nc<W and grid[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc)); parent[(nr,nc)]=cur; q.append((nr,nc))
    return None

H, W = 18, 18
real_grid = np.zeros((H, W), dtype=int)
random.seed(31)
for r in range(H):
    for c in range(W): 
        if random.random() < 0.06: real_grid[r, c] = 1

agents = [{'pos':p, 'path':[], 'id':i} for i, p in enumerate([(1,1), (1,W-2), (H-2,1)])]
explored = set(a['pos'] for a in agents)
unexplored = {(r, c) for r in range(H) for c in range(W) if real_grid[r, c] == 0} - explored

# Partition columns
cols_per = W // 3
regions = [set() for _ in range(3)]
for (r, c) in unexplored:
    idx = min(c // cols_per, 2)
    regions[idx].add((r, c))

frames = []
for _ in range(600):
    if not any(regions): break
    
    for i, a in enumerate(agents):
        reg = regions[i]
        if not a['path'] and reg:
            target = min(reg, key=lambda x: abs(x[0]-a['pos'][0]) + abs(x[1]-a['pos'][1]))
            path = bfs(a['pos'], {target}, H, W, real_grid)
            if path: a['path'] = path
        
        if a['path']:
            if len(a['path']) > 1: a['path'].pop(0); a['pos'] = a['path'][0]
        
        if a['pos'] in reg:
            reg.remove(a['pos'])
            explored.add(a['pos'])

    frames.append({'agents': [a['pos'] for a in agents], 'explored': list(explored)})

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('black')

def render(frame):
    ax.clear(); ax.axis('off')
    data = frames[frame]
    
    # Fog of War: Start Black (0), Explored areas become Grey (1)
    canvas = np.zeros((H, W, 3)) 
    for (r, c) in data['explored']:
        canvas[r, c] = [0.4, 0.4, 0.4] # Revealed floor
    
    # Walls are visible if explored? Let's assume we just see explored floor.
    ax.imshow(canvas, extent=[0, W, H, 0])

    for i, pos in enumerate(data['agents']):
        color = ['#FF1744', '#00E5FF', '#76FF03'][i]
        ax.scatter(pos[1], pos[0], c=color, s=150, edgecolors='white')
        # Vision radius visual
        ax.add_patch(plt.Circle((pos[1], pos[0]), 2, color=color, alpha=0.1))

    ax.set_title(f"MAP EXPLORATION | Explored: {len(data['explored'])}", color='white')
    ax.set_ylim(H-0.5, -0.5); ax.set_xlim(-0.5, W-0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=50, repeat=False)
plt.show()