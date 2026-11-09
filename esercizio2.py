# ESERCIZIO 2 – Introduzione a Dask DataFrame e calcolo distribuito

# Importiamo le librerie utilizzate nello script.
# NumPy e Pandas sono mantenute tra le importazioni del tuo file,
# anche se in questa versione non vengono usate direttamente.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd


# Leggiamo il CSV con Dask invece che con Pandas.
# dd.read_csv restituisce un Dask DataFrame organizzato in partizioni.
# assume_missing=True fa trattare come possibili valori mancanti
# alcune colonne che altrimenti potrebbero essere inferite come intere.
# blocksize="256KB" indica la dimensione dei blocchi usati per suddividere
# il file CSV in partizioni.
# NOTA: transazioni_demo.csv deve esistere nella cartella di lavoro
# e deve contenere almeno le colonne timestamp e value.
ddf = dd.read_csv(
    "transazioni_demo.csv",
    assume_missing=True,
    blocksize="256KB"
)

# Convertiamo timestamp in un vero tipo data/ora.
# Usiamo dd.to_datetime perché la colonna appartiene a un Dask DataFrame.
ddf["timestamp"] = dd.to_datetime(ddf["timestamp"])

# Creiamo una nuova colonna chiamata date.
# .dt.floor("1D") porta ogni timestamp all'inizio del suo giorno:
# per esempio, 10:35 e 14:20 dello stesso giorno diventano entrambe 00:00.
ddf["date"] = ddf["timestamp"].dt.floor("1D")


# Raggruppiamo i dati per giorno e selezioniamo la colonna value.
# mean() costruisce il calcolo della media giornaliera.
# compute() avvia l'elaborazione Dask e restituisce il risultato
# come normale oggetto Pandas (in questo caso una Serie).
daily = ddf.groupby("date")["value"].mean().compute()

# Ordiniamo i risultati in base all'indice temporale.
# È utile farlo prima di disegnare una serie temporale.
daily = daily.sort_index()


# Creiamo la figura Matplotlib con dimensioni 6 x 8 pollici.
plt.figure(figsize=(6, 8))

# Disegniamo la media giornaliera.
# daily.index contiene le date (asse X);
# daily.values contiene le medie calcolate (asse Y).
plt.plot(daily.index, daily.values)

# Aggiungiamo titolo ed etichette degli assi.
# "TITOLO", "DATE" e "VALORI" sono le diciture provvisorie del tuo script.
plt.title("TITOLO")
plt.xlabel("DATE")
plt.ylabel("VALORI")

# Sistemiamo gli spazi del grafico.
plt.tight_layout()

# Salviamo la figura in formato PNG.
plt.savefig("esercizio2.png")

# Mostriamo il grafico a schermo.
plt.show()
