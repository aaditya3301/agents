import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

H, W = 16, 16
grid = np.zeros((H, W), dtype=int)
random.seed(21)
for r in range(H):
    for c in range(W): 
        if random.random() < 0.03: grid[r, c] = 1

agents = [{'pos':p, 'path':[], 'task':None} for p in [(0,0), (0,W-1), (H-1,W-1)]]
fires = {(random.randint(3,H-4), random.randint(3,W-4)) for _ in range(5)}

frames = []
for _ in range(400):
    if not fires: break
    
    # Assign
    for a in agents:
        if not a['task'] and fires:
            a['task'] = min(fires, key=lambda x: abs(x[0]-a['pos'][0]) + abs(x[1]-a['pos'][1]))
            path = bfs(a['pos'], {a['task']}, H, W, grid)
            if path: a['path'] = path
    
    # Move & Extinguish
    for a in agents:
        if a['path']:
            if len(a['path']) > 1: a['path'].pop(0); a['pos'] = a['path'][0]
        
        if a['pos'] in fires:
            fires.remove(a['pos'])
            a['task'] = None
            a['path'] = []

    # Spread
    new_fires = set(fires)
    for f in fires:
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = f[0]+dr, f[1]+dc
            if 0<=nr<H and 0<=nc<W and grid[nr,nc]==0:
                if random.random() < 0.05: # 5% spread chance
                    new_fires.add((nr, nc))
    fires = new_fires
    
    frames.append({'agents': [a['pos'] for a in agents], 'fires': list(fires)})

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('black')

def render(frame):
    ax.clear(); ax.axis('off')
    data = frames[frame]
    
    canvas = np.zeros((H, W, 3)) + 0.1
    canvas[grid==1] = [0.3, 0.3, 0.3]
    ax.imshow(canvas, extent=[0, W, H, 0])

    if data['fires']:
        fy, fx = zip(*data['fires'])
        ax.scatter(fx, fy, c='#FF3D00', s=120, marker='^', alpha=0.8, label='Fire')

    for pos in data['agents']:
        ax.scatter(pos[1], pos[0], c='#2962FF', s=180, marker='o', edgecolors='white')

    ax.set_title(f"FIREFIGHTERS | Active Fires: {len(data['fires'])}", color='white')
    ax.set_ylim(H-0.5, -0.5); ax.set_xlim(-0.5, W-0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=100, repeat=False)
plt.show()