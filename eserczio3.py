# ESERCIZIO 3 - DATASET FINANZIARIO SIMULATO E OTTIMIZZAZIONE MARKOWITZ
# Generiamo 100 titoli su 5 anni, calcoliamo rendimenti e correlazioni,
# e costruiamo una frontiera efficiente tramite scipy.optimize.

# ---------------------------------------------------------------------
# 1. IMPORTAZIONE DELLE LIBRERIE
# ---------------------------------------------------------------------

import numpy as np              # Array, matrici e operazioni matematiche
import pandas as pd             # DataFrame e gestione delle date
import matplotlib.pyplot as plt # Creazione dei grafici
from scipy.optimize import minimize  # Ottimizzazione con vincoli


# ---------------------------------------------------------------------
# 2. PARAMETRI DELLA SIMULAZIONE
# ---------------------------------------------------------------------

n_tickers = 100       # Numero di titoli da simulare
years = 5             # Durata della simulazione in anni
trading_days = 252    # Numero convenzionale di giornate di contrattazione per anno
T = years * trading_days  # Numero totale di osservazioni: 5 * 252 = 1260

# Generatore casuale con seme fisso: rende ripetibili i numeri generati
rng = np.random.default_rng(0)

# Creiamo 100 nomi: T000, T001, T002, ..., T099
# :03d formatta l'intero con almeno tre cifre, aggiungendo zeri a sinistra
# Creiamo 1260 date lavorative (esclude sabati e domeniche, non tutte le festività)
tickers = [f"T{i:03d}" for i in range(n_tickers)]
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=T)


# ---------------------------------------------------------------------
# 3. PARAMETRI FINANZIARI CASUALI PER CIASCUN TITOLO
# ---------------------------------------------------------------------

# mu: 100 rendimenti attesi annuali simulati
# Media della distribuzione = 0.08 (8%); deviazione standard = 0.02
mu = rng.normal(0.08, 0.02, size=n_tickers)

# sigma: 100 volatilità annuali simulate
# Media della distribuzione = 0.20 (20%); deviazione standard = 0.05
sigma = rng.normal(0.20, 0.05, size=n_tickers)

# dt rappresenta la frazione di anno di una giornata di contrattazione
# Serve per convertire i parametri annuali in parametri giornalieri
dt = 1 / trading_days


# ---------------------------------------------------------------------
# 4. GENERAZIONE DEI RENDIMENTI LOGARITMICI GIORNALIERI
# ---------------------------------------------------------------------

# Matrice inizializzata a zero:
# 1260 righe = giornate; 100 colonne = titoli
# float32 riduce la memoria rispetto a float64
log_returns = np.zeros((T, n_tickers), dtype=np.float32)

# Simuliamo separatamente la serie dei rendimenti di ciascun titolo
for i in range(n_tickers):
    # Convertiamo il rendimento atteso annuale del titolo i in valore giornaliero
    daily_mu = mu[i] * dt

    # Convertiamo la volatilità annuale in volatilità giornaliera
    # La volatilità si scala con la radice quadrata del tempo
    daily_sigma = sigma[i] * np.sqrt(dt)

    # Generiamo T rendimenti casuali per il titolo corrente
    eps = rng.normal(loc=daily_mu, scale=daily_sigma, size=T)

    # Inseriamo i rendimenti nella colonna i, convertendoli in float32
    log_returns[:, i] = eps.astype(np.float32)


# ---------------------------------------------------------------------
# 5. CONVERSIONE DEI RENDIMENTI IN PREZZI E CREAZIONE DEL DATAFRAME
# ---------------------------------------------------------------------

# Sommiamo cumulativamente i rendimenti lungo il tempo (axis=0),
# applichiamo l'esponenziale e assumiamo un prezzo iniziale pari a 100
prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

# Organizziamo i prezzi in un DataFrame:
# indice = date; colonne = ticker dei 100 titoli
df = pd.DataFrame(prices, index=dates, columns=tickers)

# Controlliamo le dimensioni e la memoria occupata dal DataFrame
print("Data shape:", df.shape)
print("Memory (MB):", df.memory_usage(deep=True).sum() / (1024**2))


# ---------------------------------------------------------------------
# 6. CALCOLO DEI RENDIMENTI, DELLE MEDIE E DELLE COVARIANZE
# ---------------------------------------------------------------------

# pct_change() calcola la variazione percentuale rispetto al prezzo precedente
# iloc[1:] rimuove la prima riga, per la quale non esiste un prezzo precedente
# astype('float32') mantiene i rendimenti in precisione singola
returns = df.pct_change().iloc[1:].astype('float32')

# Rendimento medio giornaliero di ciascun titolo
mean_daily = returns.mean()

# Matrice delle covarianze giornaliere tra i 100 titoli
cov_daily = returns.cov()

# Annualizziamo media e covarianza moltiplicando per 252 giornate
mean_ann = mean_daily * trading_days
cov_ann = cov_daily * trading_days


# ---------------------------------------------------------------------
# 7. MATRICE DI CORRELAZIONE E RELATIVA HEATMAP
# ---------------------------------------------------------------------

# Calcoliamo la correlazione tra i rendimenti dei 100 titoli
corr = returns.corr()

plt.figure(figsize=(8, 6))

# Mostriamo i valori della matrice come immagine colorata
# corr.values estrae i valori numerici dal DataFrame Pandas
plt.imshow(corr.values, interpolation='nearest', aspect='auto')

plt.colorbar(label='Correlation')
plt.title('Correlation matrix (100 tickers)')
plt.xlabel('Tickers')
plt.ylabel('Tickers')
plt.tight_layout()

# Salviamo la heatmap in PNG prima di mostrarla
plt.savefig("esercizio3_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------
# 8. PREPARAZIONE DEI VINCOLI PER L'OTTIMIZZAZIONE
# ---------------------------------------------------------------------

n = n_tickers

# Ogni peso deve essere compreso tra 0 e 1:
# non permettiamo pesi negativi (vendite allo scoperto) né superiori al 100%
bounds = tuple((0.0, 1.0) for _ in range(n))

# Vincolo di uguaglianza: la somma dei pesi deve essere pari a 1 (100%)
cons = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})


# ---------------------------------------------------------------------
# 9. FUNZIONI DI SUPPORTO PER RENDIMENTO E RISCHIO
# ---------------------------------------------------------------------

def port_stats(weights, mean_returns, cov_matrix):
    # Rendimento del portafoglio: somma pesata dei rendimenti dei titoli
    ret = np.dot(weights, mean_returns)

    # Volatilità del portafoglio, calcolata usando la matrice di covarianza
    vol = np.sqrt(np.dot(weights, cov_matrix.dot(weights)))

    # Restituiamo entrambi i risultati: prima rendimento, poi volatilità
    return ret, vol


def min_vol(weights, mean_returns, cov_matrix):
    # L'ottimizzatore deve minimizzare la volatilità, cioè il secondo risultato
    return port_stats(weights, mean_returns, cov_matrix)[1]


# ---------------------------------------------------------------------
# 10. PREPARAZIONE DEI RENDIMENTI OBIETTIVO E DELLE LISTE RISULTATO
# ---------------------------------------------------------------------

# Creiamo 30 rendimenti obiettivo equidistanti tra questi due limiti
# min() e max() individuano il rendimento annualizzato minimo e massimo
# np.linspace() genera i 30 valori intermedi
# Nota: questi limiti sono quelli scelti dal codice del professore
target_returns = np.linspace(mean_ann.min() * 0.6, mean_ann.max() * 1.4, 30)

# Liste che raccoglieranno volatilità e rendimento dei portafogli ottimizzati
frontier_vol = []
frontier_ret = []

# Pesi iniziali: distribuzione equa del capitale tra i 100 titoli (1% ciascuno)
w0 = np.repeat(1 / n, n)


# ---------------------------------------------------------------------
# 11. OTTIMIZZAZIONE DELLA FRONTIERA EFFICIENTE
# ---------------------------------------------------------------------

for targ in target_returns:

    # Aggiungiamo al vincolo sulla somma dei pesi un secondo vincolo:
    # il rendimento del portafoglio deve essere uguale al rendimento obiettivo targ
    # targ=targ nella lambda conserva il valore dell'obiettivo di questa iterazione
    cons_t = (
        cons,
        {'type': 'eq', 'fun': lambda w, targ=targ: np.dot(w, mean_ann) - targ}
    )

    # minimize cerca i pesi che rendono minima la volatilità
    # rispettando bounds e vincoli
    res = minimize(
        min_vol,
        w0,
        args=(mean_ann.values, cov_ann.values),
        method='SLSQP',
        bounds=bounds,
        constraints=cons_t,
        options={'maxiter': 200, 'ftol': 1e-6}
    )

    # Se l'ottimizzazione fallisce, il codice riprova usando soltanto
    # il vincolo sulla somma dei pesi. In questo caso il rendimento obiettivo
    # non è più imposto: il punto ottenuto potrebbe quindi non rispettare targ.
    if not res.success:
        res = minimize(
            min_vol,
            w0,
            args=(mean_ann.values, cov_ann.values),
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

    # res.x contiene i pesi trovati dall'ottimizzatore
    w_opt = res.x

    # Calcoliamo rendimento e volatilità del portafoglio ottimizzato
    r_opt, v_opt = port_stats(
        w_opt,
        mean_ann.values,
        cov_ann.values
    )

    # Conserviamo i risultati nelle rispettive liste
    frontier_ret.append(r_opt)
    frontier_vol.append(v_opt)

    # Usiamo i pesi appena trovati come punto di partenza
    # per l'ottimizzazione del rendimento obiettivo successivo
    w0 = w_opt


# Convertiamo le liste in array NumPy per utilizzarle nei grafici
frontier_ret = np.array(frontier_ret)
frontier_vol = np.array(frontier_vol)


# ---------------------------------------------------------------------
# 12. GRAFICO DELLA FRONTIERA EFFICIENTE
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))

# Asse X = volatilità annualizzata; asse Y = rendimento annualizzato
# '-o' disegna una linea con marcatori circolari
plt.plot(frontier_vol, frontier_ret, '-o', markersize=4, alpha=0.8)

plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.title("Efficient Frontier (Markowitz) - Simulated")
plt.grid(True)
plt.tight_layout()

# Salviamo la frontiera efficiente in un secondo file PNG
plt.savefig("esercizio3_frontiera.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------
# 13. CONFRONTO TRA I 100 TITOLI SINGOLI
# ---------------------------------------------------------------------

# Deviazione standard giornaliera dei singoli titoli, annualizzata
single_vol = returns.std() * np.sqrt(trading_days)

# Rendimenti medi annualizzati dei singoli titoli
single_ret = mean_ann

plt.figure(figsize=(8, 5))

# Ogni punto rappresenta un singolo titolo:
# asse X = volatilità annualizzata; asse Y = rendimento annualizzato
plt.scatter(single_vol, single_ret, s=10, alpha=0.6)

plt.xlabel("Volatility (ann)")
plt.ylabel("Return (ann)")
plt.title("Single assets: Return vs Volatility")
plt.tight_layout()

# Salviamo il confronto dei singoli titoli in un terzo file PNG
plt.savefig("esercizio3_asset_singoli.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------
# 14. CONTROLLO FINALE DELLA MATRICE DI COVARIANZA
# ---------------------------------------------------------------------

# Verifichiamo le dimensioni della matrice annualizzata: 100 x 100
print("Cov matrix shape:", cov_ann.shape)

# nbytes misura i byte occupati dai soli valori numerici della matrice
# Convertiamo i byte in MB (MiB) dividendo per 1024**2
print(
    "Cov matrix memory (MB):",
    cov_ann.values.nbytes / (1024**2)
)
