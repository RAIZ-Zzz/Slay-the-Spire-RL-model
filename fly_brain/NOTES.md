# fly_brain — side project, not core

Source: <https://github.com/thejimkeen/fruit-fly-brain> (MIT code, CC-BY 4.0 data).
Started 2026-09-15. **This is a side quest.** Stage 1 tabular Q-learning in `../rl/`
is the main line and this must not displace it.

Everything below was read off the repo's own README / PROJECT_SUMMARY /
fly_brain.py / fly_play_dino.py on 2026-09-15. **Nothing here has been run yet.**

---

## What the thing actually is

A *connectome* — a wiring diagram of a real fruit fly brain — used directly as
the weight matrix of a network, with no training.

| | |
|---|---|
| Data | Drosophila MaleCNS v1.0, released by Google + Janelia |
| Size | ~165,178 neurons, ~50M synapses |
| Access | neuPrint REST API, needs a free `NEUPRINT_TOKEN` |
| Model | directed weighted graph: neuron = node, synapse = edge, weight = synapse count |
| Inference | spreading activation, `steps=2`, `decay=0.6` |
| Learning | **none.** Weights are biological measurements and are frozen. |

The whole forward pass is:

```
act[next] += act[cur] * edge_weight * decay          # repeat `steps` times
```

That is it. No activation function, no time, no membrane potential, no
neurotransmitter sign (excitatory and inhibitory are treated identically —
the repo lists this as a known limitation).

## The three demos in the repo

| File | What it drives |
|---|---|
| `fly_brain.py` | CLI, three preset threat scenarios, prints which DNs light up |
| `fly_brain_ui.py` | Streamlit version of the same |
| `fly_play_dino.py` | plays the Chrome dinosaur game |
| `fly_maze.py`, `fly_robot.py`, `robot_sim.py` | maze + PyBullet robot (robot untested, PyBullet install unfinished) |

## The dino agent, in full

This is the one worth understanding, because it is the same shape as our
`autoplay.choose = policy` seam.

```
obs = [dino_y, obstacle_x, obstacle_y, game_speed]

# --- encode: game state -> neuron activations -----------------------
distance_factor = (200 - obstacle_x) / 200        # only when obstacle_x < 200
speed_factor    = 1 + game_speed / 10
threat          = 15000 * distance_factor * speed_factor

if obstacle_y >= 280:   seed = {left  LPLC2 neurons (up to 30): threat}
else:                   seed = {right LPLC2 neurons (up to 30): threat}

# --- propagate ------------------------------------------------------
activation = propagator.propagate(seed, steps=2)   # decay 0.6

# --- decode: DN activation -> game action ---------------------------
if peak DN activation < 3000:  ACTION_NOTHING
elif obstacle_y >= 280:        ACTION_JUMP   (15-frame cooldown)
else:                          ACTION_DUCK   (10-frame cooldown)
```

LPLC2 = a real visual neuron type that responds to looming objects.
DN = descending neurons, the ones that carry commands from brain to body.
A hardcoded `DN_BEHAVIOR_MAP` names a few: DNp01 escape/jump, DNp02 run,
DNp04 left turn, DNp06 brake, DNg56 stillness.

## 🔴 The thing to verify first

Read that decode block again. **`obstacle_y >= 280` appears in both the encoder
and the decoder.** The jump-vs-duck choice is made by a Python `if` on the raw
observation, not by which DN fired. The connectome contributes exactly one
scalar — a magnitude, thresholded at 3000 — i.e. "act / don't act".

The repo's own PROJECT_SUMMARY says the same thing from the other side:

> "Testing revealed encoder choice affects performance by 10x more than brain
> topology itself."

⚠️ This reading comes from a *summary* of `fly_play_dino.py`, not from the
literal source. **Step 1 is to read the real file and confirm or kill it.**
If it holds, the honest description of the demo is "a hand-written rule with a
connectome-shaped gate in front of it", and any claim about the brain deciding
anything has to be tested by ablation, not assumed.

That test is cheap and it is the same ablation logic already written up for
`draw_pile` on the 进度看板: keep everything else identical, replace the
connectome with a degree-matched random graph, and see whether the score moves.

## Practical prerequisites (none met yet)

- [ ] neuPrint account + `NEUPRINT_TOKEN`
- [ ] `neuprint-python`, `pandas`, `streamlit`, `pygame`
- [ ] Python 3.11 — **this machine has 3.14** (`AppData/Roaming/Python/Python314`), unverified whether the deps build
- [ ] network egress to neuprint.janelia.org, and some idea of how much data the
      adjacency fetch pulls down
