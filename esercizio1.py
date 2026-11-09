# ESERCIZIO 1 – Gestione della memoria e visualizzazione di 5 milioni di punti

# Importiamo le librerie necessarie
import numpy as np              # Per generare e manipolare array numerici
import pandas as pd             # Per organizzare i dati in un DataFrame
import matplotlib.pyplot as plt # Per creare i grafici


# Numero totale di punti che vogliamo generare
N = 5_000_000

# Creiamo un generatore di numeri casuali con seme 42.
# In questo modo la simulazione è riproducibile: otteniamo gli stessi dati
# quando rieseguiamo il programma con lo stesso codice.
rng = np.random.default_rng(42)

# Numero di gruppi (cluster) da generare
clusters = 6


# Calcoliamo quanti punti assegnare a ciascun gruppo.
# La divisione intera // restituisce un numero intero.
# In questo caso otteniamo 833.333 punti per gruppo.
points_per_cluster = N // clusters

# Prepariamo tre liste vuote: ciascuna conterrà un array per ogni gruppo.
xs = []   # Coordinate orizzontali dei punti
ys = []   # Coordinate verticali dei punti
cats = [] # Etichetta numerica del gruppo di appartenenza


# Generiamo i sei gruppi, uno per ogni iterazione del ciclo.
for i in range(clusters):

    # Generiamo il centro orizzontale del gruppo.
    # loc=i*10.0 sposta progressivamente i gruppi lungo l'asse X:
    # 0, 10, 20, 30, 40 e 50 (con una piccola variazione casuale).
    # scale=0.5 indica la deviazione standard della posizione del centro.
    # size=1 genera un solo valore; [0] estrae quel valore dall'array.
    cx = rng.normal(loc=i * 10.0, scale=0.5, size=1)[0]

    # Generiamo il centro verticale del gruppo.
    # loc=i*5.0 sposta progressivamente i gruppi lungo l'asse Y:
    # 0, 5, 10, 15, 20 e 25 (con una piccola variazione casuale).
    cy = rng.normal(loc=i * 5.0, scale=0.5, size=1)[0]

    # Generiamo i valori X del gruppo intorno al centro cx.
    # scale=1.0 controlla la dispersione dei punti intorno al centro.
    # size=points_per_cluster stabilisce quanti valori generare.
    # append aggiunge l'array alla lista xs.
    xs.append(rng.normal(loc=cx, scale=1.0, size=points_per_cluster))

    # Generiamo nello stesso modo i valori Y intorno al centro cy.
    ys.append(rng.normal(loc=cy, scale=1.0, size=points_per_cluster))

    # Creiamo un'etichetta per ogni punto del gruppo corrente.
    # np.full ripete il valore i per points_per_cluster volte.
    # int8 è un tipo intero leggero, sufficiente per etichette da 0 a 5.
    cats.append(np.full(points_per_cluster, i, dtype=np.int8))


# Uniamo separatamente i sei array di ciascuna lista.
# concatenate non somma i valori: mette gli array uno dopo l'altro.
x = np.concatenate(xs)
y = np.concatenate(ys)
cat = np.concatenate(cats)

# La divisione intera ha prodotto 4.999.998 punti.
# Calcoliamo quanti punti mancano per raggiungere esattamente N.
extra = N - x.size

# Aggiungiamo i punti mancanti alle coordinate X e Y.
# Questi ultimi valori vengono generati con una distribuzione normale standard.
x = np.concatenate([x, rng.normal(size=extra)])
y = np.concatenate([y, rng.normal(size=extra)])

# Aggiungiamo anche le etichette dei punti extra: in questo caso sono zeri.
# Manteniamo int8 per non aumentare inutilmente la memoria.
cat = np.concatenate([cat, np.zeros(extra, dtype=np.int8)])


# Creiamo il DataFrame con tre colonne.
# Ogni riga rappresenta un punto: x e y sono le coordinate,
# mentre cat indica a quale gruppo appartiene.
df = pd.DataFrame({
    "x": x,
    "y": y,
    "cat": cat
})


# Definiamo una funzione che misura la memoria occupata dal DataFrame.
# memory_usage(deep=True) restituisce la memoria usata dalle colonne;
# sum() somma i valori e la divisione per 1024**2 converte i byte in MiB.
# Nota: questa misura riguarda il DataFrame, non tutta la RAM del programma.
def mem_usage(df):
    return df.memory_usage(deep=True).sum() / (1024**2)

# Stampiamo il numero di righe e la memoria del DataFrame iniziale.
print("Rows:", len(df))
print("Memory iniziale MB:", mem_usage(df))


# Creiamo una copia per ottimizzare i tipi senza modificare df.
df_opt = df.copy()

# Convertiamo le coordinate da float64 (8 byte per valore)
# a float32 (4 byte per valore), riducendo la memoria delle due colonne.
df_opt["x"] = df_opt["x"].astype("float32")
df_opt["y"] = df_opt["y"].astype("float32")

# Visualizziamo i tipi delle colonne e la memoria del DataFrame ottimizzato.
# La colonna cat è già int8, quindi non la convertiamo in int16:
# int16 occuperebbe più memoria di int8.
print(df_opt.dtypes)
print("Memory ottimizzata MB:", mem_usage(df_opt))


# Prepariamo un campione casuale di 200.000 punti per lo scatter plot.
sample_size = 200_000

# Estraiamo 200.000 posizioni casuali tra le righe del DataFrame.
# replace=False impedisce di selezionare più volte la stessa posizione.
sample_idx = rng.choice(len(df_opt), size=sample_size, replace=False)

# iloc seleziona le righe in base alle posizioni estratte.
# Il DataFrame originale df_opt non viene modificato.
sample = df_opt.iloc[sample_idx]


# Definiamo il numero di intervalli (bin) sugli assi X e Y.
# La griglia avrà 500 x 500 = 250.000 celle.
nx, ny = 500, 500

# np.histogram2d divide il piano in una griglia bidimensionale e conta
# quanti punti ricadono in ciascuna cella.
# Usiamo tutti i 5 milioni di punti di df_opt, non solo il campione.
# counts contiene i conteggi; xedges e yedges contengono i bordi delle celle.
counts, xedges, yedges = np.histogram2d(
    df_opt["x"],
    df_opt["y"],
    bins=[nx, ny]
)

# Calcoliamo il centro di ogni intervallo sugli assi X e Y.
# xedges[:-1] prende tutti i bordi tranne l'ultimo;
# xedges[1:] prende tutti i bordi tranne il primo.
# La media dei due bordi consecutivi dà il centro di ciascun intervallo.
Xc = (xedges[:-1] + xedges[1:]) / 2
Yc = (yedges[:-1] + yedges[1:]) / 2


# Creiamo una figura con una riga e due grafici affiancati.
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# GRAFICO DI SINISTRA: scatter plot dei 200.000 punti campionati.
# s=1 imposta la dimensione dei marker; alpha=0.3 la loro trasparenza.
axs[0].scatter(sample["x"], sample["y"], s=1, alpha=0.3)
axs[0].set_title(f"Random sample {sample_size} punti")

# GRAFICO DI DESTRA: rappresentazione della densità dei 5 milioni di punti.
# counts.T trasponde la matrice dei conteggi per orientarla correttamente
# rispetto alle coordinate X e Y.
# shading='auto' lascia a Matplotlib la scelta della modalità di colorazione.
# rasterized=True rasterizza questo elemento quando si salva in formati
# vettoriali come PDF o SVG; non cambia il calcolo dei conteggi.
pcm = axs[1].pcolormesh(
    Xc, Yc, counts.T,
    shading="auto",
    rasterized=True
)

# Aggiungiamo la barra dei colori, che indica il numero di punti per cella.
fig.colorbar(pcm, ax=axs[1], label="counts")
axs[1].set_title("2D binning heatmap (500x500)")

# Sistemiamo gli spazi tra i due grafici.
plt.tight_layout()

# Salviamo la figura come immagine PNG.
plt.savefig("esercizio1.png")

# Mostriamo la figura a schermo.
plt.show()
