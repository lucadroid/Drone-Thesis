import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# --- 1. SETUP ---
np.random.seed(42)
n_nodi = 20
n_droni = 5
base_coords = np.array([50, 50]) 
nodi_coords = np.random.rand(n_nodi, 2) * 100
velocita_drone = 15.0 
AUTONOMIA_MAX = 1000.0 # 1 km

# --- 2. CLUSTERIZZAZIONE ---
kmeans = KMeans(n_clusters=n_droni, n_init=10).fit(nodi_coords)
gruppi = kmeans.labels_

# --- 3. OTTIMIZZAZIONE TSP ---
def ottimizza_percorso(coords_cluster, base):
    tutti_i_punti = np.vstack([base, coords_cluster])
    n = len(tutti_i_punti)
    dist_matrix = cdist(tutti_i_punti, tutti_i_punti)
    
    m = gp.Model(); m.Params.OutputFlag = 0
    # Variabile binaria: x[i,j] = 1 se il drone va da i a j
    x = m.addVars(n, n, vtype=GRB.BINARY)
    # Variabile continua u[i]: serve per l'ordine (Eliminazione Sottocicli MTZ)
    u = m.addVars(n, vtype=GRB.CONTINUOUS)
    
    for i in range(n): x[i,i].ub = 0
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(x.sum('*', i) == 1 for i in range(n))
    # VINCOLO MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j: m.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1)
            
    m.setObjective(gp.quicksum(x[i,j] * dist_matrix[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    m.optimize()
    
    ordine = [0]
    while len(ordine) < n:
        curr = ordine[-1]
        next_node = [j for j in range(n) if x[curr, j].X > 0.5][0]
        ordine.append(next_node)
    
    # Restituiamo le coordinate nell'ordine ottimo
    return [tutti_i_punti[idx] for idx in ordine]

# --- 4. PIANIFICAZIONE MISSIONE MULTI-GIRO ---
missioni_complete = []                              # Lista delle missioni per ogni drone
cronologia_tempi = []                               # Tempi di arrivo per ogni punto in ogni missione

for i in range(n_droni):
    percorso_base = ottimizza_percorso(nodi_coords[gruppi == i], base_coords)       # Percorso ottimizzato per il cluster del drone i
    
    # Calcolo lunghezza di un singolo giro completo (Base -> Nodi -> Base)
    dist_giro = sum(np.linalg.norm(percorso_base[(j+1)%len(percorso_base)] - percorso_base[j]) for j in range(len(percorso_base)))
    
    # Quanti giri completi può fare con 1000m?
    num_giri = int(AUTONOMIA_MAX // dist_giro)
    
    # Costruiamo la missione: Base -> (Nodi -> Nodi) x N -> Base
    # Per semplicità facciamo: Base -> Nodi -> Nodi (ripetuto) -> Base
    nodi_pure = percorso_base[1:] # Togliamo la base dall'inizio
    missione = [base_coords]
    for _ in range(num_giri):
        missione.extend(nodi_pure)
    missione.append(base_coords)
    
    missioni_complete.append(missione)
    
    # Calcolo tempi per l'animazione
    tempi = [0.0]
    for k in range(len(missione)-1):
        d = np.linalg.norm(np.array(missione[k+1]) - np.array(missione[k]))
        tempi.append(tempi[-1] + d / velocita_drone)
    cronologia_tempi.append(tempi)
    
    print(f"Drone {i}: Lunghezza giro {dist_giro:.1f}m. Eseguirà {num_giri} giri consecutivi prima di ricaricare.")

# --- 5. ANIMAZIONE ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-10, 110); ax.set_ylim(-10, 110)
ax.plot(base_coords[0], base_coords[1], 'kP', markersize=15, label="Base")

colori = ['blue', 'green', 'red', 'purple', 'orange']
for i in range(n_nodi):
    ax.plot(nodi_coords[i,0], nodi_coords[i,1], 'o', color=colori[gruppi[i]], alpha=0.3)

disegni_droni = [ax.plot([], [], 'X', color=colori[i], markersize=10)[0] for i in range(n_droni)]

def update(frame):
    t_sim = frame * 0.06 # Velocità simulazione
    for i in range(n_droni):
        m = missioni_complete[i]
        t = cronologia_tempi[i]
        t_tot = t[-1]
        t_att = t_sim % t_tot
        
        idx = 0
        for j in range(len(t)-1):
            if t[j] <= t_att < t[j+1]:
                idx = j
                break
        
        prog = (t_att - t[idx]) / (t[idx+1] - t[idx])
        pos = np.array(m[idx]) + (np.array(m[idx+1]) - np.array(m[idx])) * prog
        disegni_droni[i].set_data([pos[0]], [pos[1]])
    return disegni_droni

ani = animation.FuncAnimation(fig, update, frames=2000, interval=20, blit=True)
plt.legend(); plt.show()