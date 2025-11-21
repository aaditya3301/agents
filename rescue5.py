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

H, W = 15, 15
grid = np.zeros((H, W), dtype=int)
random.seed(3)
for r in range(H):
    for c in range(W): 
        if random.random() < 0.1: grid[r, c] = 1

agents = [{'pos':s, 'path':[], 'hist':[s]} for s in [(0,0), (0,W-1), (H-1,0)]]
victims = set()
while len(victims) < 6:
    p = (random.randrange(H), random.randrange(W))
    if grid[p] == 0: victims.add(p)

frames = []
for _ in range(300):
    if not victims: break
    
    for a in agents:
        if not a['path'] and victims:
            path = bfs(a['pos'], victims, H, W, grid)
            if path: a['path'] = path
        
        if a['path']:
            if len(a['path']) > 1: a['path'].pop(0); a['pos'] = a['path'][0]
            a['hist'].append(a['pos'])
            
        if a['pos'] in victims: victims.remove(a['pos']); a['path'] = []

    frames.append({'agents': [a['pos'] for a in agents], 'victims': list(victims), 'trails': [list(a['hist']) for a in agents]})

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('black')

def render(frame):
    ax.clear(); ax.axis('off')
    data = frames[frame]
    
    canvas = np.zeros((H, W, 3)) + 0.1
    canvas[grid==1] = [0.3, 0.3, 0.3]
    ax.imshow(canvas, extent=[0, W, H, 0])
    
    # Victims
    if data['victims']:
        vy, vx = zip(*data['victims'])
        ax.scatter(vx, vy, c='red', marker='P', s=150, edgecolors='white')

    # Trails
    for t in data['trails']:
        if not t: continue
        ty, tx = zip(*t[-10:])
        ax.plot(tx, ty, c='#00E676', alpha=0.5, linewidth=2)

    # Agents
    for pos in data['agents']:
        ax.scatter(pos[1], pos[0], c='#00E676', s=180, edgecolors='black')

    ax.set_title(f"RESCUE SQUAD | Victims Left: {len(data['victims'])}", color='white')
    ax.set_ylim(H-0.5, -0.5); ax.set_xlim(-0.5, W-0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=100, repeat=False)
plt.show()