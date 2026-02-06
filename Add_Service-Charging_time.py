# Importazione della libreria Gurobi per risolvere il problema matematico di ottimizzazione TSP
import gurobipy as gp 

# Importazione delle costanti di Gurobi (come GRB.BINARY o GRB.MINIMIZE)
from gurobipy import GRB 

# Importazione di NumPy per la gestione efficiente di array e calcoli sulle coordinate
import numpy as np 

# Importazione di Matplotlib per la creazione della parte grafica e dei grafici 2D
import matplotlib.pyplot as plt 

# Importazione del modulo animation per gestire il movimento dei droni nel tempo
import matplotlib.animation as animation 

# Importazione di KMeans per suddividere i nodi in gruppi (uno per ogni drone)
from sklearn.cluster import KMeans 

# Importazione di cdist per calcolare velocemente la distanza tra tutte le coppie di punti
from scipy.spatial.distance import cdist 

# Fissaggio del seme casuale per garantire che i nodi siano generati sempre nelle stesse posizioni
np.random.seed(42) 

# Definizione del numero totale di sensori da monitorare nella simulazione
n_nodi = 20 

# Definizione della posizione centrale della stazione di ricarica [X, Y]
base_coords = np.array([50, 50]) 

# Generazione di coordinate casuali per i 20 nodi all'interno di un'area 100x100 metri
nodi_coords = np.random.rand(n_nodi, 2) * 100 

# Impostazione della velocità costante di volo dei droni espressa in metri al secondo
velocita_drone = 15.0 

# Definizione dell'autonomia massima della batteria espressa come distanza percorribile in metri
AUTONOMIA_MAX = 1000.0 

# Variabile impostabile per decidere quanti secondi il drone deve restare fermo sul nodo per lo scan
tempo_scan_nodo = 5.0 

# Variabile impostabile per decidere quanti secondi il drone deve restare in base per la ricarica
tempo_ricarica_base = 10.0 

# Generazione di scadenze casuali per ogni sensore comprese tra 60 e 150 secondi
deadlines = np.random.randint(30, 70, size=n_nodi) 

# Stampa di una linea decorativa per la tabella dei nodi nel terminale
print("-" * 65) 

# Stampa dell'intestazione della tabella con le informazioni sui nodi
print(f"{'NODO':<8} | {'COORD X':<12} | {'COORD Y':<12} | {'DEADLINE (s)':<15}") 

# Stampa di una linea decorativa di separazione
print("-" * 65) 

# Ciclo per stampare i dati di posizione e scadenza di ogni singolo nodo
for i in range(n_nodi): 
    print(f"{i:<8} | {nodi_coords[i,0]:<12.2f} | {nodi_coords[i,1]:<12.2f} | {deadlines[i]:<15}") 

# Chiusura estetica della tabella nel terminale
print("-" * 65) 

# Definizione della funzione per ottimizzare il percorso TSP di un singolo drone
def ottimizza_percorso(coords_cluster, base): 
    # Unione delle coordinate della base con quelle dei nodi assegnati al cluster
    punti = np.vstack([base, coords_cluster]) 
    
    # Calcolo del numero totale di fermate previste per questo drone
    n = len(punti) 
    
    # Creazione della matrice delle distanze euclidee tra tutti i punti del cluster
    dist_matrix = cdist(punti, punti) 
    
    # Inizializzazione del modello di ottimizzazione Gurobi in modalità silenziosa
    m = gp.Model(); m.Params.OutputFlag = 0 
    
    # Creazione delle variabili decisionali binarie per ogni possibile arco tra i punti
    x = m.addVars(n, n, vtype=GRB.BINARY) 
    
    # Creazione delle variabili continue per l'eliminazione dei sottocicli (metodo MTZ)
    u = m.addVars(n, vtype=GRB.CONTINUOUS) 
    
    # Vincolo per impedire al drone di creare un arco che parta e torni nello stesso punto
    for i in range(n): x[i,i].ub = 0 
    
    # Vincolo: da ogni punto deve uscire esattamente un solo arco verso un altro punto
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n)) 
    
    # Vincolo: in ogni punto deve entrare esattamente un solo arco proveniente da un altro punto
    m.addConstrs(x.sum('*', i) == 1 for i in range(n)) 
    
    # Ciclo per impostare i vincoli MTZ che garantiscono la creazione di un unico tour chiuso
    for i in range(1, n): 
        for j in range(1, n): 
            if i != j: m.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1) 
            
    # Definizione dell'obiettivo: minimizzare la somma totale delle distanze degli archi scelti
    m.setObjective(gp.quicksum(x[i,j] * dist_matrix[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE) 
    
    # Esecuzione dell'algoritmo di ottimizzazione
    m.optimize() 
    
    # Inizializzazione della lista che conterrà l'ordine di visita partendo dalla base
    ordine = [0] 
    
    # Ricostruzione della sequenza di visita leggendo i valori delle variabili x ottimizzate
    while len(ordine) < n: 
        curr = ordine[-1] 
        next_node = [j for j in range(n) if x[curr, j].X > 0.5][0] 
        ordine.append(next_node) 
        
    # Restituzione delle coordinate dei punti nell'ordine calcolato
    return [punti[idx] for idx in ordine] 

# Inizializzazione del numero di droni a 1 per iniziare la ricerca iterativa
n_droni = 1 

# Variabile di controllo per fermare la ricerca quando i vincoli sono soddisfatti
successo = False 

# Lista per memorizzare i percorsi base ottimizzati per ogni drone
percorsi_tsp = [] 

# Lista per memorizzare le distanze totali dei singoli giri dei droni
distanze_giro = [] 

# Lista per memorizzare l'appartenenza di ogni nodo a un drone specifico
gruppi_labels = [] 

# Inizio del ciclo per trovare il numero minimo di droni necessari
while not successo: 
    print(f"Test con {n_droni} droni...") 
    
    # Suddivisione dei nodi in cluster basata sulla vicinanza spaziale (KMeans)
    km = KMeans(n_clusters=n_droni, n_init=10).fit(nodi_coords) 
    
    # Recupero delle etichette che assegnano ogni nodo a un drone
    labels = km.labels_ 
    
    # Inizializzazione di variabili temporanee per validare la configurazione corrente
    temp_p, temp_d, valido = [], [], True 
    
    # Ciclo per verificare la fattibilità del monitoraggio per ogni drone nel sistema
    for i in range(n_droni): 
        idx = np.where(labels == i)[0] 
        if len(idx) == 0: continue 
        
        # Calcolo del percorso TSP ottimo per il gruppo di nodi assegnato
        p = ottimizza_percorso(nodi_coords[idx], base_coords) 
        
        # Somma delle distanze di tutti gli archi del percorso calcolato
        d = sum(np.linalg.norm(p[j] - p[(j+1)%len(p)]) for j in range(len(p))) 
        
        # Calcolo del tempo necessario per volare lungo tutto il percorso
        tempo_volo = d / velocita_drone 
        
        # Calcolo del tempo totale del giro includendo il tempo di sosta (scan) su ogni nodo
        tempo_totale_giro = tempo_volo + (len(idx) * tempo_scan_nodo) 
        
        # Identificazione della scadenza più restrittiva tra i nodi del cluster
        deadline_limite = min(deadlines[idx]) 
        
        # Controllo se il tempo totale del giro supera la deadline critica
        if tempo_totale_giro > deadline_limite: 
            ritardo = tempo_totale_giro - deadline_limite 
            print(f"   -> Fallito: Drone {i} eccede deadline di {ritardo:.1f}s") 
            valido = False; break 
            
        # Salvataggio temporaneo del percorso e della distanza se validi
        temp_p.append(p); temp_d.append(d) 
        
    # Se la configurazione è valida per tutti i droni, esce dal ciclo while
    if valido: 
        successo = True; percorsi_tsp = temp_p; distanze_giro = temp_d; gruppi_labels = labels 
    else: n_droni += 1 

# Intestazione del report finale delle missioni nel terminale
print("\n--- REPORT MISSIONI ---") 

# Inizializzazione delle liste per memorizzare la missione estesa e i relativi tempi
missioni_complete = [] 

# Lista per memorizzare i timestamp di ogni tappa della missione (volo, scan, ricarica)
cronologie_tempi = [] 

# Ciclo per costruire la sequenza di eventi infinita per ogni drone
for i in range(n_droni): 
    p_tsp = percorsi_tsp[i] 
    nodi_pure = p_tsp[1:] 
    m_punti = [base_coords] 
    m_tempi = [0.0] 
    batteria = AUTONOMIA_MAX 
    pos_att = base_coords 
    idx_n = 0 
    
    # Simulazione di 200 eventi di missione per ogni drone
    for _ in range(200): 
        prossimo = nodi_pure[idx_n % len(nodi_pure)] 
        d_prossimo = np.linalg.norm(prossimo - pos_att) 
        d_rientro = np.linalg.norm(base_coords - prossimo) 
        
        # Verifica se la batteria residua permette di visitare il nodo e tornare in base
        if batteria > (d_prossimo + d_rientro + 10): 
            # Registrazione del movimento verso il nodo sensore
            m_punti.append(prossimo) 
            m_tempi.append(m_tempi[-1] + d_prossimo / velocita_drone) 
            
            # Registrazione della sosta sul nodo per il tempo di scan (posizione invariata)
            m_punti.append(prossimo) 
            m_tempi.append(m_tempi[-1] + tempo_scan_nodo) 
            batteria -= d_prossimo 
            pos_att = prossimo 
            idx_n += 1 
        else: 
            # Registrazione del movimento di rientro forzato alla base per batteria scarica
            d_base = np.linalg.norm(base_coords - pos_att) 
            m_punti.append(base_coords) 
            m_tempi.append(m_tempi[-1] + d_base / velocita_drone) 
            
            # Registrazione della sosta in base per il tempo di ricarica (posizione invariata)
            m_punti.append(base_coords) 
            m_tempi.append(m_tempi[-1] + tempo_ricarica_base) 
            batteria = AUTONOMIA_MAX 
            pos_att = base_coords 
            
    # Salvataggio delle liste complete di punti e tempi per il drone i
    missioni_complete.append(m_punti) 
    cronologie_tempi.append(m_tempi) 
    
    # Stampa dei parametri operativi finali del drone nel report
    print(f"Drone {i}: Giro {distanze_giro[i]:.1f}m. Scan: {tempo_scan_nodo}s/nodo. Ricarica: {tempo_ricarica_base}s.") 

# Creazione della figura e degli assi per la visualizzazione dell'animazione
fig, ax = plt.subplots(figsize=(10, 8)) 

# Impostazione dei limiti visivi della mappa di monitoraggio
ax.set_xlim(-10, 110); ax.set_ylim(-10, 110) 

# Impostazione del titolo grafico con i parametri principali
ax.set_title(f"Monitoraggio: {n_droni} droni | Scan: {tempo_scan_nodo}s | Ricarica: {tempo_ricarica_base}s") 

# Definizione della mappa colori tab10 per distinguere i droni e i loro cluster
colori = plt.get_cmap('tab20') 

# Disegno dell'icona della base centrale sulla mappa
ax.plot(base_coords[0], base_coords[1], 'kP', markersize=15, label="Base") 

# Ciclo per disegnare ogni nodo e la relativa scritta della deadline
for i in range(n_nodi): 
    ax.plot(nodi_coords[i,0], nodi_coords[i,1], 'o', color=colori(gruppi_labels[i]), alpha=0.5, markersize=10) 
    ax.text(nodi_coords[i,0]+1, nodi_coords[i,1]+1, f"{deadlines[i]}s", fontsize=8) 

# Aggiorna la creazione dei droni (le X)
droni_ani = []
for i in range(n_droni):
    # markersize=8 per non coprire tutto, e colore coerente con tab20
    linea, = ax.plot([], [], 'X', color=colori(i), markersize=8, label=f"Drone {i}", zorder=5)
    droni_ani.append(linea) 

# Creazione di un oggetto di testo per mostrare il cronometro digitale in tempo reale
tempo_testo = ax.text(0.02, 0.95, '', transform=ax.transAxes, weight='bold', color='darkblue') 

# Definizione della funzione di aggiornamento per ogni frame dell'animazione
def update(frame): 
    # Calcolo del tempo virtuale di simulazione (avanzamento di 1 secondo per frame)
    t_sim = frame * 1.0 
    
    # Aggiornamento del cronometro visualizzato sul grafico
    tempo_testo.set_text(f'Tempo Missione: {t_sim:.1f}s') 
    
    # Ciclo per aggiornare la posizione di ogni drone nell'istante t_sim
    for i in range(n_droni): 
        pts = missioni_complete[i]; times = cronologie_tempi[i] 
        t_curr = t_sim % times[-1] 
        idx = 0 
        
        # Ricerca dell'intervallo temporale corretto nella cronologia del drone
        for j in range(len(times)-1): 
            if times[j] <= t_curr < times[j+1]: 
                idx = j; break 
                
        # Calcolo della posizione tramite interpolazione tra il punto attuale e il successivo
        inizio, fine = np.array(pts[idx]), np.array(pts[idx+1]) 
        
        # Controllo per evitare divisioni per zero se il drone è fermo (scan o ricarica)
        if times[idx+1] != times[idx]: 
            prog = (t_curr - times[idx]) / (times[idx+1] - times[idx]) 
            pos = inizio + (fine - inizio) * prog 
        else: pos = inizio 
        
        # Aggiornamento delle coordinate dell'icona del drone i
        droni_ani[i].set_data([pos[0]], [pos[1]]) 
        
    # Restituzione degli elementi grafici da aggiornare
    return droni_ani + [tempo_testo] 

# velocità simulazione = t_sim / interval. Con 60 ms per frame, 1 secondo di simulazione = 16.67 secondi reali.
ani = animation.FuncAnimation(fig, update, frames=3000, interval=60, blit=True) 

# Posizionamento della legenda con scala dei simboli ridotta per pulizia visiva
plt.legend(loc='upper right', fontsize='small', markerscale=0.8) 

# Abilitazione della griglia di sfondo semitrasparente
plt.grid(True, alpha=0.3) 

# Comando finale per mostrare la finestra con l'animazione a schermo
plt.show()