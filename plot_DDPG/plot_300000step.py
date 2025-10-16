import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from log_ddpg_td3 import *



# --- Lecture des données ---
all_qf1 = []
all_Gpi = []
steps = None
name = "TD3"
log_dirs = log_dirs_td3
for log in log_dirs:
    ea = event_accumulator.EventAccumulator(log)
    ea.Reload()

    # Vérifie que les tags existent
    tags = ea.Tags()["scalars"]
    if "losses/qf1_values" not in tags or "losses/G_pi" not in tags:
        print(f"[WARN] Tags manquants dans {log}")
        continue

    qf1_events = ea.Scalars("losses/qf1_values")
    Gpi_events = ea.Scalars("losses/G_pi")

    # Récupère les steps et valeurs
    qf1_vals = np.array([e.value for e in qf1_events])
    Gpi_vals = np.array([e.value for e in Gpi_events])
    steps = np.array([e.step for e in qf1_events])  # mêmes steps supposés

    all_qf1.append(qf1_vals)
    all_Gpi.append(Gpi_vals)

# --- Conversion en tableaux numpy ---
all_qf1 = np.array(all_qf1)  # shape = (n_runs, n_steps)
all_Gpi = np.array(all_Gpi)

# --- Calcul de la moyenne et de l'écart-type ---
mean_qf1 = np.mean(all_qf1, axis=0)
std_qf1 = np.std(all_qf1, axis=0)

mean_Gpi = np.mean(all_Gpi, axis=0)
std_Gpi = np.std(all_Gpi, axis=0)

# --- Tracé ---
plt.figure(figsize=(12, 6))

# Qf1
plt.plot(steps, mean_qf1, label="Q_values (moyenne)", color="blue")
plt.fill_between(steps, mean_qf1 - std_qf1, mean_qf1 + std_qf1, color="blue", alpha=0.2)

# G_pi
plt.plot(steps, mean_Gpi, label="Q_pi (moyenne)", color="orange")
plt.fill_between(steps, mean_Gpi - std_Gpi, mean_Gpi + std_Gpi, color="orange", alpha=0.2)

plt.xlabel("Step")
plt.ylabel("Valeur")
plt.title(f"{name} Moyenne et variance de Q_values et Q_pi sur 10 apprentissages")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
