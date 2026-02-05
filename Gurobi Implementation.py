''' PROBLEMA BASE:
3 Punti di Interesse (Target): Luoghi da sorvegliare.

3 Droni: Pronti a partire.

Il problema: Ogni drone deve andare in uno e un solo punto, e ogni punto deve essere visitato da uno e un solo drone. Vogliamo che la somma delle distanze percorse da tutti i droni sia la minima possibile.

MODELLO: 

Variabili: Le decisioni che Gurobi deve prendere (es. "Il drone A va al punto 1? Sì o No").

Vincoli: (es. "Non puoi mandare due droni nello stesso punto").

Obiettivo: Cosa vogliamo ottimizzare (es. "Riduci al minimo i km totali").'''

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ==========================================
# 1. PARTE GUROBI 
# ==========================================

# Definiamo dove sono i droni (Partenza) e i Target (Arrivo)
pos_droni = {'Drone_A': [0, 2], 'Drone_B': [0, 5], 'Drone_C': [0, 8]}
pos_target = {'Punto_1': [10, 1], 'Punto_2': [10, 9], 'Punto_3': [10, 5]}

droni_nomi = list(pos_droni.keys())
target_nomi = list(pos_target.keys())

# Creiamo il modello Gurobi
modello = gp.Model("Ottimizzazione_Movimento")

# Variabile: Drone X va al Punto Y? (0 = No, 1 = Sì) GRB.BINARY: o il drone va lì, o non ci va
assegnazione = modello.addVars(droni_nomi, target_nomi, vtype=GRB.BINARY, name="assegna")       

# Vincoli: Ogni drone ha 1 target, ogni target ha 1 drone gp.quicksum: Invece di scrivere x1 + x2 + x3, scriviamo una sommatoria
for d in droni_nomi:
    modello.addConstr(gp.quicksum(assegnazione[d, t] for t in target_nomi) == 1)
for t in target_nomi:
    modello.addConstr(gp.quicksum(assegnazione[d, t] for d in droni_nomi) == 1)

# Funzione Obiettivo: Minimizzare la distanza totale, calcolata con la formula della distanza euclidea (radice di ((x2 - x1)^2 + (y2 - y1)^2))
def calcola_distanza(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

obiettivo = gp.quicksum(assegnazione[d, t] * calcola_distanza(pos_droni[d], pos_target[t]) 
                        for d in droni_nomi for t in target_nomi)
modello.setObjective(obiettivo, GRB.MINIMIZE)

# Gurobi trova la soluzione ottima
modello.optimize()

# Salviamo la soluzione in una lista per l'animazione
percorsi_scelti = []
if modello.status == GRB.OPTIMAL:               # Se Gurobi ha trovato la soluzione ottima
    for d in droni_nomi:                        # Controlla ogni singola combinazione possibile tra ogni Drone e ogni Target". 3 droni, 3 target = 9 controlli (3×3)
        for t in target_nomi:
            if assegnazione[d, t].X > 0.5:      # Se la variabile è 1 (cioè il drone va al target)
                percorsi_scelti.append((d, t))  # Aggiungiamo alla lista la coppia (drone, target)

# ==========================================
# 2. PARTE ANIMAZIONE 
# ==========================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 11)
ax.set_title("Simulazione: Droni diretti ai Target ottimizzati da Gurobi")

# Disegniamo i target (punti rossi fissi)
for nome, coord in pos_target.items():
    ax.plot(coord[0], coord[1], 'rs', markersize=10)
    ax.text(coord[0]+0.2, coord[1], nome)

# Creiamo i droni (cerchi blu che si muoveranno)
disegni_droni = [ax.plot([], [], 'bo', markersize=12)[0] for _ in percorsi_scelti]
etichette_droni = [ax.text(0, 0, "") for _ in percorsi_scelti]

def muovi(frame):
    # frame va da 0 a 200
    for i, (d_nome, t_nome) in enumerate(percorsi_scelti):
        partenza = np.array(pos_droni[d_nome])
        arrivo = np.array(pos_target[t_nome])
        
        # Calcoliamo la posizione intermedia
        progresso = frame / 200.0
        attuale = partenza + (arrivo - partenza) * progresso
        
        # Aggiorniamo il disegno
        disegni_droni[i].set_data([attuale[0]], [attuale[1]])
        etichette_droni[i].set_text(d_nome)
        etichette_droni[i].set_position((attuale[0], attuale[1] + 0.5))
        
    return disegni_droni + etichette_droni

# Crea l'animazione
ani = animation.FuncAnimation(fig, muovi, frames=201, interval=30, blit=True)

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()