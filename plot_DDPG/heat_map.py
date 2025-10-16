import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import json

log_dirs = [
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__1__1760473491",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__2__1760481683",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__3__1760494865",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__4__1760500427",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__5__1760505781",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__6__1760510240",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__7__1760519109",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__8__1760526606",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__9__1760536208",
    "..\\src\\runs\\LunarLanderContinuous-v3__ddpg_monte_carlo_continuous_action__10__1760543190",
]

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
    min_len = min(len(G0_lists), len(qf1_vals))
    G0_lists = G0_lists[:min_len]
    qf1_vals = qf1_vals[:min_len]

    # Calcul des différences G0 - qf1 pour chaque step
    diffs = []
    for g0, q in zip(G0_lists, qf1_vals):
        diffs.append(g0 - q)
    diffs = np.vstack(diffs)  # shape = (n_steps, len(G0))
    all_diffs.append(diffs)

# --- Moyenne sur les 10 runs ---

mean_diffs = np.mean(all_diffs, axis=0)  # shape = (n_steps, len(G0))

mean_diffs_sorted = np.sort(mean_diffs, axis=1)



# --- Heatmap ---
plt.figure(figsize=(10, 8))
im = plt.imshow(mean_diffs_sorted, aspect="auto", cmap="coolwarm", origin="lower",
                extent=[0, mean_diffs.shape[1], list_step[0], list_step[-1]])

plt.colorbar(im, label="G0 - Q_pi (moyenne sur 10 runs)")
plt.xlabel("liste des G0 - Q_pi (dans l'ordre croissant)")
plt.ylabel("Step")
plt.title("Heatmap des différences moyennes G0 - Q_valeur triées à l’intérieur de chaque step")
plt.tight_layout()
plt.show()
