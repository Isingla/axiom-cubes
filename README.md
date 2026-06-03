# Axiom Cubes: Diagnostic Benchmarks for Learned Heuristic Search

Three modified Rubik's Cubes that isolate the structural properties determining whether learned heuristic search succeeds or fails.

## The Three Axioms

We formalize learnable heuristic search over a state space (S, A, gamma, S_g) with three axioms:

| Axiom | Definition | Intuition |
|-------|-----------|-----------|
| **Conservation** | A measurable invariant is preserved under every action | The puzzle never gains or loses pieces |
| **Ergodicity** | The state graph is strongly connected | You can get from anywhere to anywhere |
| **Uniformity** | The same action produces the same local transformation regardless of state | A move works the same way no matter how scrambled things are |

## The Cubes

Each cube breaks exactly one axiom while preserving the other two:

| Cube | Conservation | Ergodicity | Uniformity | What changes |
|------|:-----------:|:----------:|:----------:|-------------|
| `StandardCube` | Y | Y | Y | Nothing (baseline) |
| `JokerCube` | **N** | Y | Y | 25% of stickers are wild -- they crystallize after each move |
| `TrapCube` | Y | **N** | Y | 10% of states are deterministic dead ends |
| `ContextCube` | Y | Y | **N** | Same move rotates CW or CCW depending on center sticker color |

## Key Finding

Conservation violation alone accounts for **94%** of total degradation in learned heuristic search.

```
Standard Cube:   81.7% solve rate (baseline)
Joker Cube:      17.2% solve rate (-64.5%)
Trap Cube:       81.1% solve rate (-0.6%)
Context Cube:    78.3% solve rate (-3.4%)
```

## The Counting Shortcut Paradox

The Joker Cube reveals a counterintuitive failure mode: **better training metrics, worse search performance**.

The model trained on the Joker Cube achieves:
- Lower training loss (0.013 vs 0.161)
- Higher ranking accuracy (96.4% vs 94.7%)

But solves **0% of cubes at depth 5 or greater**.

Why? The number of joker stickers correlates perfectly with scramble depth. The network learns to count jokers instead of learning cube geometry. Counting gives near-perfect distance estimation but tells you nothing about which move to make.

This is the same failure mode observed when applying learned heuristics to molecular retrosynthesis -- atom count correlates with synthesis depth, so the GNN counts atoms instead of learning reaction structure.

## Installation

```bash
pip install axiom-cubes
```

Or just copy `axiom_cubes.py` into your project. No dependencies beyond Python standard library.

## Usage

```python
from axiom_cubes import StandardCube, JokerCube, TrapCube, ContextCube

# Create a Joker Cube and scramble it
cube = JokerCube()
state = cube.scramble(depth=10)

# Inspect the state
print(cube.to_string(state))
print(f"Jokers: {cube.joker_count(state)}")
print(f"Colors: {cube.color_counts(state)}")

# Try all 12 moves
for move in range(12):
    next_state = cube.move(state, move)
    print(f"Move {cube.move_name(move)}: "
          f"jokers={cube.joker_count(next_state)}, "
          f"solved={cube.is_solved(next_state)}")

# Get tensor representation for neural network input
tensor_data = cube.to_tensor_data(state)  # 54 x 7 one-hot
flat_dim = cube.tensor_dim()  # 378

# For PyTorch:
import torch
tensor = torch.tensor(tensor_data).flatten()  # [378]
```

### Trap Cube

```python
from axiom_cubes import TrapCube

cube = TrapCube(trap_rate=0.10)
state = cube.scramble(depth=8)

# Check if a state is a dead end
print(f"Is trap: {cube.is_trap(state)}")

# Get only moves that don't lead to traps
safe = cube.safe_moves(state)
print(f"Safe moves: {[cube.move_name(m) for m in safe]}")
```

### Context Cube

```python
from axiom_cubes import ContextCube

cube = ContextCube()
state = cube.scramble(depth=5)

# The same move number does different things depending on center colors
# Even center color -> clockwise, Odd center color -> counter-clockwise
state2 = cube.move(state, 0)  # "U" but direction depends on Up center color
```

### Compare All Cubes

```python
from axiom_cubes import describe_all
describe_all()
```

Output:
```
AXIOM DIAGNOSTIC CUBE SUITE
============================================================

Three modified Rubik's Cubes that each break exactly one
axiom of learnable heuristic search.

  Cube              C   E   U  Description
  -------------------------------------------------------
  Standard          Y   Y   Y  Standard 3x3 Rubik's Cube
  Joker             N   Y   Y  Wild stickers that crystallize
  Trap              Y   N   Y  10% dead-end states
  Context           Y   Y   N  Context-dependent moves
```

## Training Your Own Heuristic

The cubes provide scrambled states and distance labels. Train any architecture:

```python
from axiom_cubes import JokerCube

cube = JokerCube()

# Generate training data: (state, scramble_depth) pairs
train_data = []
for _ in range(100000):
    depth = random.randint(1, 20)
    state = cube.scramble(depth)
    tensor = cube.to_tensor_data(state)
    train_data.append((tensor, depth))

# Train your model to predict depth from state
# Then use it to solve: at each step, try all 12 moves,
# pick the one your model scores as closest to solved
```

## Why This Matters

If you're building learned heuristics for any combinatorial domain, run your method on all three cubes. The results tell you which axiom your method is vulnerable to:

- **Fails on Joker Cube**: Your model finds counting shortcuts. Fix the state representation so countable quantities are invisible.
- **Fails on Trap Cube**: Your model doesn't handle dead ends. Add trap detection or use search algorithms that backtrack.
- **Fails on Context Cube**: Your model can't handle context-dependent actions. Increase model capacity or add state-conditional action encoding.

Most methods fail only on the Joker Cube. Conservation is the critical axiom.

## Applications

The counting shortcut paradox explains failures in:

- **Molecular retrosynthesis**: GNNs count atoms instead of learning reaction structure
- **PDDL planning**: Domains with object creation/destruction break conservation
- **Game playing**: Games with piece capture (chess, Go) violate conservation

The axiom framework predicts which domains are learnable and which require representation normalization.

## Citation

```
@software{singla2026axiomcubes,
  author = {Singla, Ishaan},
  title = {Axiom Cubes: Diagnostic Benchmarks for Learned Heuristic Search},
  year = {2026},
  url = {https://github.com/Isingla/axiom-cubes}
}
```

## License

MIT
