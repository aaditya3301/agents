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
            path = []
            while cur: path.append(cur); cur = parent[cur]
            return path[::-1]
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<H and 0<=nc<W and grid[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc)); parent[(nr,nc)]=cur; q.append((nr,nc))
    return None

H, W = 12, 12
grid = np.zeros((H, W), dtype=int)
random.seed(8)
for r in range(H):
    for c in range(W): 
        if random.random() < 0.05: grid[r, c] = 1

to_paint = {(r, c) for r in range(H) for c in range(W) if grid[r, c] == 0 and random.random() < 0.4}

painters = [
    {'pos':(0,0), 'path':[], 'color':'#E040FB', 'rem':{p for p in to_paint if (p[0]+p[1])%2==0}},
    {'pos':(H-1,W-1), 'path':[], 'color':'#00E5FF', 'rem':{p for p in to_paint if (p[0]+p[1])%2!=0}}
]

curr_painted = set()
frames = []

for _ in range(400):
    if all(not p['rem'] for p in painters): break
    
    for p in painters:
        if not p['path'] and p['rem']:
            target = min(p['rem'], key=lambda x: abs(x[0]-p['pos'][0])+abs(x[1]-p['pos'][1]))
            path = bfs(p['pos'], {target}, H, W, grid)
            if path: p['path'] = path
        
        if p['path']:
            if len(p['path'])>1: p['path'].pop(0); p['pos'] = p['path'][0]
        
        if p['pos'] in p['rem']:
            p['rem'].remove(p['pos'])
            curr_painted.add((p['pos'], p['color']))
    
    frames.append({'agents': [p['pos'] for p in painters], 'painted': list(curr_painted)})

print(f"Painting finished. Frames: {len(frames)}")
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#212121')

def render(frame):
    ax.clear(); ax.axis('off')
    data = frames[frame]
    
    canvas = np.zeros((H, W, 3)) + 0.2
    canvas[grid==1] = [0.1, 0.1, 0.1]
    ax.imshow(canvas, extent=[0, W, H, 0])
    
    for (pos, col) in data['painted']:
        ax.scatter(pos[1], pos[0], c=col, s=180, marker='s', alpha=0.9)

    for i, pos in enumerate(data['agents']):
        ax.scatter(pos[1], pos[0], c=painters[i]['color'], s=200, edgecolors='white', linewidth=2)

    ax.set_title(f"GRID PAINTERS | Painted: {len(data['painted'])}", color='white')
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=80, repeat=False)
plt.show()