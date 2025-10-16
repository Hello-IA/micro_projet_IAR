import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from scipy import stats

from log_ddpg_td3 import *

# --- Lecture des données ---
all_qf1 = []
all_Gpi = []
steps = None
name = "DDPG"
log_dirs = log_dirs_ddpg

for log in log_dirs:
    ea = event_accumulator.EventAccumulator(log)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if "losses/qf1_values" not in tags or "losses/G_pi" not in tags:
        print(f"[WARN] Tags manquants dans {log}")
        continue

    qf1_events = ea.Scalars("losses/qf1_values")
    Gpi_events = ea.Scalars("losses/G_pi")
    qf1_vals = np.array([e.value for e in qf1_events])
    Gpi_vals = np.array([e.value for e in Gpi_events])
    steps = np.array([e.step for e in qf1_events])

    all_qf1.append(qf1_vals)
    all_Gpi.append(Gpi_vals)

all_qf1 = np.array(all_qf1)  # shape = (n_runs, n_steps)
all_Gpi = np.array(all_Gpi)
n_runs = all_qf1.shape[0]

# --- Statistiques ---
mean_qf1 = np.mean(all_qf1, axis=0)
std_qf1 = np.std(all_qf1, axis=0, ddof=1)
sem_qf1 = std_qf1 / np.sqrt(n_runs)
ci95_qf1 = stats.t.ppf(0.975, n_runs-1) * sem_qf1
q25_qf1 = np.percentile(all_qf1, 25, axis=0)
q75_qf1 = np.percentile(all_qf1, 75, axis=0)

mean_Gpi = np.mean(all_Gpi, axis=0)
std_Gpi = np.std(all_Gpi, axis=0, ddof=1)
sem_Gpi = std_Gpi / np.sqrt(n_runs)
ci95_Gpi = stats.t.ppf(0.975, n_runs-1) * sem_Gpi
q25_Gpi = np.percentile(all_Gpi, 25, axis=0)
q75_Gpi = np.percentile(all_Gpi, 75, axis=0)

# --- Tracé ---
plt.figure(figsize=(14, 6))

# Qf1
plt.plot(steps, mean_qf1, color="blue", label="Q_valeur (moyenne)")
plt.fill_between(steps, mean_qf1 - std_qf1, mean_qf1 + std_qf1, color="blue", alpha=0.1, label="± écart-type")
plt.fill_between(steps, mean_qf1 - sem_qf1, mean_qf1 + sem_qf1, color="blue", alpha=0.2, label="± erreur type")
plt.fill_between(steps, mean_qf1 - ci95_qf1, mean_qf1 + ci95_qf1, color="blue", alpha=0.3, label="IC 95%")
plt.fill_between(steps, q25_qf1, q75_qf1, color="blue", alpha=0.15, label="25%-75% quartiles")

# G_pi
plt.plot(steps, mean_Gpi, color="orange", label="Q_pi (moyenne)")
plt.fill_between(steps, mean_Gpi - std_Gpi, mean_Gpi + std_Gpi, color="orange", alpha=0.1, label="± écart-type")
plt.fill_between(steps, mean_Gpi - sem_Gpi, mean_Gpi + sem_Gpi, color="orange", alpha=0.2, label="± erreur type")
plt.fill_between(steps, mean_Gpi - ci95_Gpi, mean_Gpi + ci95_Gpi, color="orange", alpha=0.3, label="IC 95%")
plt.fill_between(steps, q25_Gpi, q75_Gpi, color="orange", alpha=0.15, label="25%-75% quartiles")

# Axes et titre
plt.xlabel("Step")
plt.ylabel("Valeur")
plt.title(f"{name} — Moyenne et multiples mesures de variabilité sur {n_runs} runs")
plt.grid(True, linestyle="--", alpha=0.5)

# Légende
plt.legend(loc="upper left", fontsize=9, frameon=True, ncol=2)
plt.figtext(0.5, -0.05,
            "Bande ombrée : écart-type, erreur-type, intervalle de confiance 95%, quartiles 25%-75%\n"
            "Moyenne calculée sur l'ensemble des runs",
            ha="center", fontsize=9, style="italic")

plt.tight_layout()
plt.show()
