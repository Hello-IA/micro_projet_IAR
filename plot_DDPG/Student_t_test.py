import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from log_ddpg_td3 import *



# --- Lecture des données ---
all_qf1 = []
all_Gpi = []
steps = None
name = "DDPG"
log_dirs = log_dirs_ddpg
Q_valeur =[]
Q_pi =[]
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
    Q_valeur.append(np.array([ea.value for ea in qf1_events]))
    Q_pi.append(np.array([ea.value for ea in Gpi_events]))
Q_valeur_array = np.array(Q_valeur)
Q_pi_array = np.array(Q_pi)
# --- Vérification dimensions ---
assert Q_valeur_array.shape == Q_pi_array.shape, "Dimensions de Q_valeur et Q_pi doivent correspondre"

# --- Différences par run et par step ---
diff_runs = Q_valeur_array - Q_pi_array  # shape = (n_runs, n_steps)

# --- Moyenne sur les runs pour chaque step ---
diff_mean = np.mean(diff_runs, axis=0)
diff_std = np.std(diff_runs, axis=0)

# --- Test t global (toutes les différences flattenées) ---
t_stat, p_value = stats.ttest_1samp(diff_runs.flatten(), 0)

print("t-statistic global:", t_stat)
print("p-value global:", p_value)
if p_value < 0.05:
    if t_stat > 0:
        print("Sur-estimation significative globalement")
    else:
        print("Sous-estimation significative globalement")
else:
    print("Pas de sur/sous-estimation significative globalement")