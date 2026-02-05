import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ==========================================
# 1. SETUP DATI E NODI
# ==========================================
np.random.seed(42)                                                  # Uso il seed 42 per riproducibilità
n_nodi = 20
n_droni = 5
nodi_coords = np.random.rand(n_nodi, 2) * 100                       # Coordinate casuali tra 0 e 100 per i nodi

# ==========================================
# 2. CLUSTERIZZAZIONE (K-MEANS)
# ==========================================
kmeans = KMeans(n_clusters=n_droni, n_init=10).fit(nodi_coords)     # Clustering K-Means per assegnare i nodi ai droni, n_clusters = numero di droni, n_init = numero di inizializzazioni per trovare la soluzione migliore
gruppi = kmeans.labels_                                             # Etichette dei cluster per ogni nodo (0, 1, 2, 3, 4)

# ==========================================
# 3. OTTIMIZZAZIONE TSP (CON ELIMINAZIONE SOTTOCICLI)
# ==========================================
def ottimizza_percorso_tsp(coords_cluster):
    n = len(coords_cluster)
    if n <= 1: return [0]
    
    dist_matrix = cdist(coords_cluster, coords_cluster)             #Crea una tabella che contiene le distanze tra ogni coppia di punti. Gurobi deve sapere quanto "costa" andare dal punto A al punto B.
    m = gp.Model()
    m.Params.OutputFlag = 0                                         # Disabilita output console Gurobi
    
    # Variabile binaria: x[i,j] = 1 se il drone va da i a j
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")                 #Questi sono i "binari" su cui viaggia il drone. Se x[A,B]=1, il drone va da A a B. Se è 0, non ci va
    # Variabile continua u[i]: serve per l'ordine (Eliminazione Sottocicli MTZ)
    u = m.addVars(n, vtype=GRB.CONTINUOUS, name="u")                #Questa è una variabile di supporto. Numero d'ordine assegnato a ogni nodo per decidere chi viene visitato prima e chi dopo. Numero delle tappe f
    for i in range(n): x[i,i].ub = 0                                # Impedisce i cicli su se stessi    
    
    # Vincoli standard: entra uno, esce uno
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n))              # Ogni nodo deve essere lasciato esattamente una volta, freccia out
    m.addConstrs(x.sum('*', i) == 1 for i in range(n))              # Ogni nodo deve essere raggiunto esattamente una volta, freccia in
    
    # VINCOLO MTZ (Il pezzo mancante): Impedisce i mini-giri tra soli 2 nodi
    '''Questo è il pezzo più tecnico ma geniale (scoperto da Miller, Tucker e Zemlin). Senza questa riga, Gurobi potrebbe creare due mini-giri separati (es. 1-2 e 3-4). 
    Questa formula assegna un valore crescente alla variabile u lungo il percorso. Poiché i valori devono crescere, il drone è obbligato a visitare tutti i nodi in una 
    sequenza unica (1 -> 2 -> 3 -> 4) prima di poter tornare alla base. Impedisce matematicamente i "cortocircuiti.
    Praticamente se il drone ha fatto nodo 1 -> nodo 2 (quindi x[1,2]=1), dopo non può tornare indietro (2 -> 1) perché u[2] deve essere maggiore di u[1]. Così via per tutti i nodi.'''
    
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1)

    m.setObjective(gp.quicksum(x[i,j] * dist_matrix[i,j] 
                               for i in range(n) for j in range(n)), GRB.MINIMIZE) #Chiede a Gurobi di sommare le distanze di tutte le frecce scelte (x=1) e trovare la combinazione che dà il totale più basso
    m.optimize()
    
    # Estrazione dell'ordine corretto
    percorso_ordinato = [0]
    while len(percorso_ordinato) < n:                              # Costruisce il percorso seguendo le frecce scelte
        attuale = percorso_ordinato[-1]
        next_node = [j for j in range(n) if x[attuale, j].X > 0.5][0]
        percorso_ordinato.append(next_node)
    
    return percorso_ordinato

# Calcolo percorsi per ogni drone
percorsi_finali = []
for i in range(n_droni):
    indici = np.where(gruppi == i)[0]
    c_cluster = nodi_coords[indici]
    ordine = ottimizza_percorso_tsp(c_cluster)
    percorsi_finali.append([c_cluster[idx] for idx in ordine])

# ==========================================
# 4. ANIMAZIONE                                                 # Con velocità fissa per semplicità 
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-10, 110); ax.set_ylim(-10, 110)
ax.set_title("Pattugliamento TSP Corretto (Senza Sottocicli)")

colori = ['blue', 'green', 'red', 'purple', 'orange']
for i in range(n_nodi):
    ax.plot(nodi_coords[i,0], nodi_coords[i,1], 'o', color=colori[gruppi[i]], alpha=0.5)

disegni_droni = [ax.plot([], [], 'X', color=colori[i], markersize=12)[0] for i in range(n_droni)]

def update(frame):
    tot_f = 800 # Aumentato per vedere bene tutti i nodi
    for i in range(n_droni):
        p = percorsi_finali[i]
        n_t = len(p)
        f_per_t = tot_f / n_t
        t_att = int((frame % tot_f) / f_per_t)
        prog = (frame % f_per_t) / f_per_t
        pos = np.array(p[t_att]) + (np.array(p[(t_att+1)%n_t]) - np.array(p[t_att])) * prog
        disegni_droni[i].set_data([pos[0]], [pos[1]])
    return disegni_droni

ani = animation.FuncAnimation(fig, update, frames=800, interval=20, blit=True)
plt.grid(True, alpha=0.3)
plt.show()