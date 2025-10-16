import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import json
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib.colors as mcolors
from log_ddpg_td3 import *
log_dirs = log_dirs_ddpg
name = "DDPG"
all_diffs = []  # Liste de matrices (n_runs × n_steps × len(G0))
list_step = []
for log in log_dirs:
    ea = event_accumulator.EventAccumulator(log)
    ea.Reload()

    tags = ea.Tags()
    if "losses/qf1_values" not in tags["scalars"] or "losses/G0_list/text_summary" not in tags["tensors"]:
        print(f"[WARN] Tags manquants dans {log}")
        continue

    qf1_events = ea.Scalars("losses/qf1_values")
    G0_events = ea.Tensors("losses/G0_list/text_summary")

    qf1_vals = np.array([e.value for e in qf1_events])

    list_step = np.array([e.step for e in qf1_events])

    # Extraction des listes G0 (elles sont stockées en texte dans TensorBoard)
    G0_lists = []
    for e in G0_events:
        t = e.tensor_proto.string_val[0].decode("utf-8")
        try:
            g0 = json.loads(t)  # suppose que c'est une liste JSON
        except json.JSONDecodeError:
            continue
        G0_lists.append(np.array(g0))
    
    # Alignement si besoin (tronquer au plus petit nombre d'étapes)


    # Calcul des différences G0 - qf1 pour chaque step
    diffs = []

    for g0, q in zip(G0_lists, qf1_vals):
        diffs.append(q - g0)

    diffs = np.vstack(diffs)  # shape = (n_steps, len(G0))
    all_diffs.append(diffs)

all_diffs = np.array(all_diffs)
# --- Calcul du t-test Student sur les runs pour chaque (step, index G0) ---
n_runs, n_steps, len_g0 = all_diffs.shape
t_stats = np.zeros((n_steps, len_g0))
p_vals = np.zeros((n_steps, len_g0))

for s in range(n_steps):
    for i in range(len_g0):
        t, p = stats.ttest_1samp(all_diffs[:, s, i], 0.0)
        t_stats[s, i] = t
        p_vals[s, i] = p

# --- Heatmap des t-stat ---

norm = mcolors.TwoSlopeNorm(vmin=-np.max(np.abs(t_stats)), vcenter=0.0, vmax=np.max(np.abs(t_stats)))
plt.figure(figsize=(10, 8))
im = plt.imshow(
    t_stats,
    aspect="auto",
    cmap="coolwarm",
    origin="lower",
    extent=[0, len_g0, list_step[0], list_step[-1]],
    norm = norm
)


# --- Ajout de la légende explicite ---
red_patch = mpatches.Patch(color=plt.cm.coolwarm(1.0), label="Sur-estimation (Q_val > G₀)")
blue_patch = mpatches.Patch(color=plt.cm.coolwarm(0.0), label="Sous-estimation (Q_val < G₀)")
plt.legend(
    handles=[red_patch, blue_patch],
    loc="upper right",
    frameon=True,
    title="Interprétation des couleurs"
)
plt.colorbar(im, label="t-stat (Student test : Q_val - Gt)")
plt.xlabel("Index dans Gt_list")
plt.ylabel("Step")
plt.title(f"{name} Heatmap des t-statistiques du test de Student (sur/sous-estimation de Q)")
plt.tight_layout()
plt.show()