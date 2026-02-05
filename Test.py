import gurobipy as gp  
from gurobipy import GRB  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.animation as animation  
from sklearn.cluster import KMeans  
from scipy.spatial.distance import cdist  # Importa cdist per calcolare velocemente le matrici di distanza

# --- 1. CONFIGURAZIONE DELLO SCENARIO ---
np.random.seed(42)                                              # Fissa il seme della casualità per ottenere sempre gli stessi risultati ad ogni avvio
n_nodi = 20                                                     # Stabilisce il numero totale di sensori/punti di interesse da monitorare
base_coords = np.array([50, 50])                                # Definisce le coordinate [X, Y] della stazione base (centrale)
nodi_coords = np.random.rand(n_nodi, 2) * 100                   # Genera 20 coppie di coordinate casuali in un'area 100x100 metri
velocita_drone = 5.0                                            # Imposta la velocità di volo costante di ogni drone in metri al secondo
AUTONOMIA_MAX = 1000.0                                          # Definisce il limite massimo di percorrenza per ogni carica della batteria (metri)

# Genera per ogni nodo una scadenza di visita (deadline) casuale compresa tra 30 e 100 secondi
deadlines = np.random.randint(25, 50, size=n_nodi)

# --- 2. STAMPA DEL PROSPETTO INIZIALE DEI NODI ---
print("-" * 65)                                                 # Stampa una linea separatrice estetica
print(f"{'NODO':<8} | {'COORD X':<12} | {'COORD Y':<12} | {'DEADLINE (s)':<15}")                    # Intestazione della tabella
print("-" * 65)                                                 # Stampa un'altra linea separatrice
for i in range(n_nodi):                                         # Ciclo per stampare i dati di ogni singolo nodo generato
    print(f"{i:<8} | {nodi_coords[i,0]:<12.2f} | {nodi_coords[i,1]:<12.2f} | {deadlines[i]:<15}")   # Dati nodo
print("-" * 65)                                                 # Chiude la tabella iniziale

def ottimizza_percorso(coords_cluster, base):
    """Funzione che calcola il percorso ottimo (TSP) per un drone all'interno del suo cluster."""
    punti = np.vstack([base, coords_cluster])                   # Crea una lista punti che inizia con la base seguita dai nodi
    n = len(punti)                                              # Conta quanti punti il drone deve visitare (nodi + base)
    dist_matrix = cdist(punti, punti)                           # Genera la matrice contenente la distanza tra ogni coppia di punti
    m = gp.Model(); m.Params.OutputFlag = 0                     # Inizializza il modello Gurobi disabilitando i log testuali
    x = m.addVars(n, n, vtype=GRB.BINARY)                       # Crea variabili binarie: x[i,j]=1 se il drone percorre il tratto i->j
    u = m.addVars(n, vtype=GRB.CONTINUOUS)                      # Crea variabili continue necessarie per eliminare i sottotour (MTZ)
    for i in range(n): x[i,i].ub = 0                            # Vincolo: il drone non può volare da un punto verso se stesso
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n))          # Vincolo: da ogni punto deve uscire esattamente un arco
    m.addConstrs(x.sum('*', i) == 1 for i in range(n))          # Vincolo: in ogni punto deve entrare esattamente un arco
    for i in range(1, n):                                       # Implementazione dei vincoli Miller-Tucker-Zemlin per garantire un tour unico
        for j in range(1, n):                                   # Cicla su tutti i nodi escludendo la base per i vincoli di ordine
            if i != j: m.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1)                                         # Formula matematica anti-subtour
    m.setObjective(gp.quicksum(x[i,j] * dist_matrix[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)  # Obiettivo: minima distanza
    m.optimize()                                                # Chiama il solutore Gurobi per trovare la soluzione ottima
    ordine = [0]                                                # Inizia la ricostruzione della sequenza partendo dall'indice della base (0)
    while len(ordine) < n:                                      # Finché non abbiamo visitato tutti i punti della lista
        curr = ordine[-1]                                       # Prende l'ultimo nodo aggiunto alla sequenza
        next_node = [j for j in range(n) if x[curr, j].X > 0.5][0]  # Trova il nodo j connesso ad i nella soluzione
        ordine.append(next_node)                                # Aggiunge il nodo trovato alla sequenza di visita
    return [punti[idx] for idx in ordine]                       # Restituisce le coordinate reali nell'ordine stabilito

# --- 3. RICERCA ITERATIVA DEL NUMERO MINIMO DI DRONI ---
n_droni = 1                                                     # Inizia il tentativo di copertura partendo con un solo drone
successo = False                                                # Variabile booleana per uscire dal ciclo quando troviamo la soluzione valida
percorsi_tsp = []                                               # Lista per memorizzare i percorsi base di ogni drone
distanze_giro = []                                              # Lista per memorizzare la lunghezza totale in metri dei giri
gruppi_labels = []                                              # Lista per memorizzare l'assegnazione finale nodo-drone

while not successo:                                             # Ciclo che incrementa il numero di droni finché i vincoli di tempo non sono soddisfatti
    print(f"Test con {n_droni} droni...")                       # Log del progresso nel terminale
    km = KMeans(n_clusters=n_droni, n_init=10).fit(nodi_coords) # Applica KMeans per dividere i nodi tra i droni
    labels = km.labels_                                         # Estrae l'etichetta del cluster per ogni nodo
    temp_p, temp_d, valido = [], [], True                       # Inizializza variabili temporanee per questo test
    for i in range(n_droni):                                    # Analizza ogni drone singolarmente per verificare la fattibilità
        idx = np.where(labels == i)[0]                          # Trova gli indici dei nodi assegnati al drone i-esimo
        if len(idx) == 0: continue                              # Salta se per qualche motivo un cluster è vuoto
        p = ottimizza_percorso(nodi_coords[idx], base_coords)   # Calcola il percorso TSP per questo drone
        d = sum(np.linalg.norm(p[j] - p[(j+1)%len(p)]) for j in range(len(p)))  # Calcola distanza totale del giro
        tempo_giro = d / velocita_drone                         # Trasforma la distanza in tempo di percorrenza (secondi)
        deadline_limite = min(deadlines[idx])                   # Trova la deadline più urgente tra i nodi di questo drone
        if tempo_giro > deadline_limite:                        # Se il drone è troppo lento per la scadenza...
            ritardo = tempo_giro - deadline_limite              # Calcola l'eccesso di tempo (ritardo)
            print(f"   -> Fallito: Drone {i} eccede la deadline di {ritardo:.1f} secondi (Tempo: {tempo_giro:.1f}s, Limite: {deadline_limite}s)")  # Log errore
            valido = False; break                               # Segnala che la configurazione non è valida e interrompe il ciclo for
        temp_p.append(p); temp_d.append(d)                      # Se valido, salva temporaneamente percorso e distanza
    if valido:                                                  # Se tutti i droni del test hanno superato la verifica...
        successo = True; percorsi_tsp = temp_p; distanze_giro = temp_d; gruppi_labels = labels  # Conferma i risultati finali
    else: n_droni += 1                                          # Altrimenti, aumenta il numero di droni per il prossimo tentativo

# --- 4. COSTRUZIONE DELLE MISSIONI CONTINUE CON RIENTRI ---
print("\n--- REPORT MISSIONI ---")                              # Inizio stampa del report riassuntivo finale
missioni_complete = []                                          # Lista di liste: conterrà tutti i punti del volo infinito per ogni drone
cronologie_tempi = []                                           # Lista di liste: conterrà i tempi di arrivo ad ogni punto della missione

for i in range(n_droni):                                        # Per ogni drone calcola la missione persistente basata sull'autonomia
    p_tsp = percorsi_tsp[i]                                     # Prende il giro base ottimizzato precedentemente
    nodi_pure = p_tsp[1:]                                       # Crea una sequenza dei soli nodi (esclude la base per facilitare il ciclo)
    m_punti = [base_coords]                                     # La missione di ogni drone deve iniziare fisicamente dalla base
    m_tempi = [0.0]                                             # Il cronometro per ogni drone parte da zero
    batteria = AUTONOMIA_MAX                                    # Carica la batteria al massimo (1000m)
    pos_att = base_coords                                       # Inizializza la posizione corrente sulla base
    idx_n = 0                                                   # Indice per scorrere ciclicamente la lista dei nodi da visitare
    for _ in range(150):                                        # Simula 150 tappe di volo (abbastanza per vedere più ricariche nell'animazione)
        prossimo = nodi_pure[idx_n % len(nodi_pure)]            # Identifica il prossimo nodo obiettivo nella sequenza circolare
        d_prossimo = np.linalg.norm(prossimo - pos_att)         # Calcola la distanza tra la posizione attuale e il prossimo nodo
        d_rientro = np.linalg.norm(base_coords - prossimo)      # Calcola la distanza necessaria per tornare alla base dal prossimo nodo
        if batteria > (d_prossimo + d_rientro + 10):            # Controlla se c'è energia per andare al nodo E tornare alla base (con margine)
            m_punti.append(prossimo)                            # Aggiunge il nodo alla traiettoria di volo
            m_tempi.append(m_tempi[-1] + d_prossimo / velocita_drone)  # Registra il tempo di arrivo al nodo
            batteria -= d_prossimo                              # Sottrae la distanza percorsa dall'autonomia residua
            pos_att = prossimo                                  # Aggiorna la posizione attuale del drone sul nodo raggiunto
            idx_n += 1                                          # Passa al puntatore del nodo successivo della lista
        else:                                                   # Se la batteria non è sufficiente per il prossimo nodo e il rientro...
            d_base = np.linalg.norm(base_coords - pos_att)      # Calcola la distanza diretta verso la base
            m_punti.append(base_coords)                         # Aggiunge la base come prossima destinazione
            m_tempi.append(m_tempi[-1] + d_base / velocita_drone)  # Registra il tempo di arrivo per la ricarica
            batteria = AUTONOMIA_MAX                            # Ricarica istantaneamente la batteria (cambio batteria rapido)
            pos_att = base_coords                               # Riposiziona il drone sulla base; idx_n non aumenta per non saltare il nodo
    missioni_complete.append(m_punti)                           # Salva l'intera lista di coordinate della missione del drone i
    cronologie_tempi.append(m_tempi)                            # Salva l'intera cronologia dei tempi per l'animazione
    num_giri_reali = int(AUTONOMIA_MAX // distanze_giro[i])     # Calcola quanti giri interi il drone può fare per carica
    print(f"Drone {i}: Giro da {distanze_giro[i]:.1f}m. Eseguirà {num_giri_reali} giri (Tot: {num_giri_reali*distanze_giro[i]:.1f}m) prima del rientro.")  # Stampa report

# --- 5. VISUALIZZAZIONE E ANIMAZIONE ---
fig, ax = plt.subplots(figsize=(10, 8))                         # Crea una finestra grafica di dimensioni 10x8 pollici
ax.set_xlim(-10, 110); ax.set_ylim(-10, 110)                    # Imposta i limiti degli assi per visualizzare bene l'area 100x100
ax.set_title(f"Monitoraggio: {n_droni} droni | Autonomia: {AUTONOMIA_MAX}m")  # Imposta il titolo informativo del grafico
colori = plt.get_cmap('tab10')                                  # Carica la mappa colori 'tab10' (sintassi corretta per le ultime versioni di Matplotlib)
ax.plot(base_coords[0], base_coords[1], 'kP', markersize=8, label="Base")     # Disegna la base come una croce nera (P)

for i in range(n_nodi):                                         # Ciclo per disegnare i sensori sulla mappa
    ax.plot(nodi_coords[i,0], nodi_coords[i,1], 'o', color=colori(gruppi_labels[i]), alpha=0.4)  # Disegna il nodo come cerchio colorato
    ax.text(nodi_coords[i,0]+1, nodi_coords[i,1]+1, f"{deadlines[i]}s", fontsize=8)              # Scrive accanto al nodo la sua deadline

# Inizializza le icone dei droni (X) con markersize ridotto (8) e prepara la legenda con le distanze TSP
droni_ani = [ax.plot([], [], 'X', color=colori(i), markersize=8, label=f"Drone {i} ({distanze_giro[i]:.1f}m)")[0] for i in range(n_droni)]

def update(frame):
    """Funzione chiamata ripetutamente per aggiornare la posizione dei droni nell'animazione."""
    t_sim = frame * 1.0                                         # Parametro che controlla la velocità dell'animazione (1.0 = velocità normale/calibrata)
    for i in range(n_droni):                                    #  Per ogni drone, calcola la posizione attuale nel tempo simulato
        pts = missioni_complete[i]; times = cronologie_tempi[i] # Recupera dati missione e tempi del drone i
        t_curr = t_sim % times[-1]                              # Fa ripartire la missione dall'inizio una volta completati i 150 step (loop)
        idx = 0                                                 # Variabile per trovare il segmento di volo corrente
        for j in range(len(times)-1):                           # Cerca l'intervallo di tempo [t_j, t_j+1] in cui si trova t_curr
            if times[j] <= t_curr < times[j+1]:                 # Se il tempo attuale è compreso tra questi due istanti...
                idx = j; break                                  # ...abbiamo trovato il segmento e usciamo dal ciclo
        prog = (t_curr - times[idx]) / (times[idx+1] - times[idx])  # Calcola la percentuale di avanzamento nel segmento
        pos = np.array(pts[idx]) + (np.array(pts[idx+1]) - np.array(pts[idx])) * prog  # Calcola la coordinata X,Y esatta
        droni_ani[i].set_data([pos[0]], [pos[1]])               # Aggiorna la posizione dell'icona X sul grafico
    return droni_ani                                            # Restituisce gli oggetti grafici aggiornati per il rendering del frame

# Crea l'animazione con 2000 frame e un intervallo di 60ms tra i frame per un movimento fluido e lento
ani = animation.FuncAnimation(fig, update, frames=2000, interval=60, blit=True)
plt.legend(loc='upper right', fontsize='small', markerscale=0.8)    # Mostra la legenda rimpicciolendo leggermente le icone
plt.grid(True, alpha=0.3)                                       # Aggiunge una griglia di sfondo semitrasparente per migliorare la leggibilità delle coordinate
plt.show()                                                      # Apre la finestra e avvia la simulazione visiva