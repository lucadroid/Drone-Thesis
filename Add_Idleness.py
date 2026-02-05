import gurobipy as gp  # Libreria per l'ottimizzazione matematica
from gurobipy import GRB  # Costanti per definire i problemi Gurobi
import numpy as np  # Per la gestione di array e calcoli matematici
import matplotlib.pyplot as plt  # Per la creazione dei grafici
import matplotlib.animation as animation  # Per creare l'animazione dinamica
from sklearn.cluster import KMeans  # Algoritmo per raggruppare i nodi in zone
from scipy.spatial.distance import cdist  # Per calcolare le distanze tra le coordinate

# --- 1. CONFIGURAZIONE SCENARIO ---
np.random.seed(42)  # Blocchiamo la casualità per avere risultati ripetibili
n_nodi = 20  # Numero totale di punti di interesse sulla mappa
base_coords = np.array([50, 50])  # Posizione della base (stazione di ricarica) al centro
nodi_coords = np.random.rand(n_nodi, 2) * 100  # Generiamo 20 nodi casuali in un'area 100x100
velocita_drone = 5.0  # Velocità di crociera costante del drone (metri/secondo)
AUTONOMIA_MAX = 1000.0  # Distanza massima percorribile con una batteria (metri)

# Generiamo tempi di scadenza (deadlines) casuali tra 30 e 100 secondi per ogni nodo
deadlines = np.random.randint(30, 100, size=n_nodi)

# --- 2. STAMPA TABELLA RIASSUNTIVA (OUTPUT TESTUALE) ---
print("-" * 65)
print(f"{'NODO':<8} | {'COORD X':<12} | {'COORD Y':<12} | {'DEADLINE (s)':<15}")
print("-" * 65)
for i in range(n_nodi):
    # Stampiamo i dettagli di ogni nodo per il report della tesi
    print(f"{i:<8} | {nodi_coords[i,0]:<12.2f} | {nodi_coords[i,1]:<12.2f} | {deadlines[i]:<15}")
print("-" * 65)

def ottimizza_percorso(coords_cluster, base):
    """Funzione che usa Gurobi per trovare il giro più breve in un cluster"""
    tutti_i_punti = np.vstack([base, coords_cluster])  # Mettiamo insieme base e nodi del gruppo
    n = len(tutti_i_punti)  # Numero di punti totali per questo drone
    dist_matrix = cdist(tutti_i_punti, tutti_i_punti)  # Calcoliamo la matrice delle distanze
    
    m = gp.Model()  # Creiamo il modello Gurobi
    m.Params.OutputFlag = 0  # Disattiviamo i messaggi tecnici di Gurobi
    
    # x[i,j] = 1 se il drone percorre il tratto dal punto i al punto j
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    # u[i] = variabile ausiliaria per l'ordine di visita (evita sottocicli)
    u = m.addVars(n, vtype=GRB.CONTINUOUS, name="u")
    
    for i in range(n): x[i,i].ub = 0  # Un nodo non può connettersi a se stesso
    
    # Vincoli di flusso: ogni nodo deve avere un'entrata e un'uscita
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(x.sum('*', i) == 1 for i in range(n))
    
    # Vincoli MTZ (Miller-Tucker-Zemlin) per eliminare i sottocicli
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1)
                
    # Obiettivo: minimizzare la distanza totale percorsa
    m.setObjective(gp.quicksum(x[i,j] * dist_matrix[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    m.optimize()  # Risolviamo il problema
    
    if m.status != GRB.OPTIMAL: return None  # Se non c'è soluzione ottimale, restituisci nulla
    
    # Ricostruiamo la sequenza dei punti partendo dalla base
    ordine = [0]
    while len(ordine) < n:
        curr = ordine[-1]
        next_node = [j for j in range(n) if x[curr, j].X > 0.5][0]
        ordine.append(next_node)
    
    # Restituiamo le coordinate ordinate e la lunghezza totale in metri
    return [tutti_i_punti[idx] for idx in ordine], m.ObjVal

# --- 3. RICERCA ITERATIVA DEL NUMERO MINIMO DI DRONI ---
n_droni = 1  # Iniziamo provando con un solo drone
successo = False  # Interruttore per fermare la ricerca
percorsi_finali = []  # Lista che conterrà i tragitti ottimi
gruppi_finali = []  # Lista che conterrà l'appartenenza dei nodi ai droni

print(f"Ricerca numero minimo di droni per rispettare le scadenze...")

while not successo:
    print(f"Testando configurazione con {n_droni} droni...")
    # Dividiamo i nodi in cluster (uno per ogni drone attuale)
    kmeans = KMeans(n_clusters=n_droni, n_init=10).fit(nodi_coords)
    gruppi = kmeans.labels_
    temp_percorsi = []
    configurazione_valida = True
    
    for i in range(n_droni):
        indici_cluster = np.where(gruppi == i)[0]  # Nodi assegnati al drone i
        ris = ottimizza_percorso(nodi_coords[indici_cluster], base_coords)
        
        if ris is None: # Se Gurobi fallisce
            configurazione_valida = False; break
            
        percorso, dist = ris
        tempo_giro = dist / velocita_drone  # Calcoliamo il tempo necessario per il giro
        
        # VERIFICA SCADENZA: il tempo del giro deve essere < della deadline minima del cluster
        if tempo_giro > min(deadlines[indici_cluster]):
            print(f"  -> Fallito: Drone {i} impiega {tempo_giro:.1f}s (Scadenza max: {min(deadlines[indici_cluster])}s)")
            configurazione_valida = False; break
            
        temp_percorsi.append(percorso)
        
    if configurazione_valida:
        successo = True  # Abbiamo trovato il numero minimo di droni!
        percorsi_finali = temp_percorsi
        gruppi_finali = gruppi
        print(f"CONFIGURAZIONE TROVATA: Servono almeno {n_droni} droni.\n")
    else:
        n_droni += 1  # Se non basta, incrementiamo il numero di droni e riproviamo

# --- 4. PREPARAZIONE ANIMAZIONE ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-10, 110); ax.set_ylim(-10, 110)  # Impostiamo i limiti della mappa
ax.set_title(f"Simulazione: {n_droni} Droni per {n_nodi} Nodi (Scadenze Rispettate)")

# Definiamo i colori per ogni cluster
colori_mappa = plt.cm.get_cmap('tab10', n_droni)
ax.plot(base_coords[0], base_coords[1], 'kP', markersize=15, label="Base Centrale") # Base

# Disegnamo i nodi sulla mappa con il testo della loro scadenza
for i in range(n_nodi):
    ax.plot(nodi_coords[i,0], nodi_coords[i,1], 'o', color=colori_mappa(gruppi_finali[i]), markersize=8)
    ax.text(nodi_coords[i,0]+1, nodi_coords[i,1]+1, f"D:{deadlines[i]}s", fontsize=8)

# Creiamo gli oggetti grafici per i droni (segnalati con una X)
disegni_droni = [ax.plot([], [], 'X', color=colori_mappa(i), markersize=12)[0] for i in range(n_droni)]

# Calcoliamo i tempi di arrivo precisi per ogni tappa di ogni drone
cronologie_tempi = []
for p in percorsi_finali:
    lista_tempi = [0.0]
    for i in range(len(p)):
        dist_tratto = np.linalg.norm(np.array(p[(i+1)%len(p)]) - np.array(p[i]))
        lista_tempi.append(lista_tempi[-1] + dist_tratto / velocita_drone)
    cronologie_tempi.append(lista_tempi)

def update(frame):
    """Funzione chiamata ad ogni fotogramma dell'animazione"""
    tempo_simulazione = frame * 1.0  # Ogni frame equivale a 1 secondo di realtà
    for i in range(n_droni):
        p = percorsi_finali[i]
        t = cronologie_tempi[i]
        t_giro_completo = t[-1]
        t_attuale = tempo_simulazione % t_giro_completo  # Fa ripartire il tempo al termine del giro
        
        # Identifichiamo in quale tratta si trova il drone al tempo attuale
        idx_tratta = 0
        for j in range(len(t)-1):
            if t[j] <= t_attuale < t[j+1]:
                idx_tratta = j; break
        
        # Calcoliamo la posizione esatta tra l'inizio e la fine della tratta attuale
        inizio, fine = np.array(p[idx_tratta]), np.array(p[(idx_tratta+1)%len(p)])
        progresso_tratta = (t_attuale - t[idx_tratta]) / (t[idx_tratta+1] - t[idx_tratta])
        posizione_attuale = inizio + (fine - inizio) * progresso_tratta
        
        # Aggiorniamo la posizione della X sulla mappa
        disegni_droni[i].set_data([posizione_attuale[0]], [posizione_attuale[1]])
    return disegni_droni

# Avviamo l'animazione: 500 frame, aggiornati ogni 50 millisecondi
ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)
plt.legend(loc='upper right')  # Aggiungiamo la legenda
plt.grid(True, alpha=0.3)  # Aggiungiamo una griglia leggera
plt.show()  # Mostriamo il risultato finale