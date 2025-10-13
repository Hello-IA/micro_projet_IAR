import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = "..\\cleanrl-master\\runs\\LunarLanderContinuous-v3__ddpg_continuous_action__1__1760289655"  # remplace par ton run_name

ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print("Tags disponibles:", ea.Tags()["scalars"])

qf1_values_events = ea.Scalars("losses/qf1_values")
G_pi_events = ea.Scalars("losses/G_pi")

qf1_values = [(e.step, e.value) for e in qf1_values_events]
G_pi_values = [(e.step, e.value) for e in G_pi_events]

steps_qf1, qf1_vals = zip(*qf1_values)
steps_G_pi, G_pi_vals = zip(*G_pi_values)

plt.figure(figsize=(12,6))
plt.plot(steps_qf1, qf1_vals, label="qf1_values", color="blue")
plt.plot(steps_G_pi, G_pi_vals, label="G_pi", color="orange")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Évolution de qf1_values et G_pi")
plt.legend()
plt.grid(True)
plt.show()
