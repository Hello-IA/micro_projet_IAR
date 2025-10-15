import json
import ast
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

# ====== PARTIE 1 — ÉCRITURE DANS TENSORBOARD ======

writer = SummaryWriter("runs/demo_text")

# Exemple de liste à enregistrer
list_G0 = [1.23, 0.87, 1.05, 0.99]*100

# Enregistrement sous forme de texte (JSON ou str)
writer.add_text("losses/G0_list", json.dumps(list_G0), global_step=0)

# On peut aussi enregistrer d'autres scalaires
writer.add_scalar("losses/G_pi", sum(list_G0) / len(list_G0), 0)

writer.close()
print("✅ Données enregistrées dans runs/demo_text")

# ====== PARTIE 2 — LECTURE DEPUIS TENSORBOARD ======

ea = event_accumulator.EventAccumulator("runs/demo_text")
ea.Reload()

# Affiche les tags disponibles
print("\nTags disponibles :")
print(ea.Tags())

# Lecture du texte
events = ea.Tensors("losses/G0_list/text_summary")
raw_str = events[0].tensor_proto.string_val[0].decode("utf-8")

# Conversion en liste Python
list_G0_loaded = json.loads(raw_str)
print("\n✅ Liste relue depuis TensorBoard :", list_G0_loaded, events[0].step)
