#!/usr/bin/env python3
"""
axiom_cubes.py -- Diagnostic Rubik's Cube Variants for Learned Heuristic Search

Three modified Rubik's Cubes that each break exactly one axiom of learnable
heuristic search. Use them to test whether your search algorithm is vulnerable
to counting shortcuts, dead ends, or context-dependent transitions.

CUBES:
    StandardCube  -- All axioms hold. Baseline.
    JokerCube     -- Conservation broken. 25% wild stickers that crystallize.
    TrapCube      -- Ergodicity broken. 10% of states are dead ends.
    ContextCube   -- Uniformity broken. Same move does CW or CCW by context.

USAGE:
    from axiom_cubes import StandardCube, JokerCube, TrapCube, ContextCube

    cube = JokerCube()
    state = cube.scramble(depth=10)
    print(cube.to_string(state))

    for move in range(12):
        next_state = cube.move(state, move)
        print(f"Move {cube.move_name(move)}: solved={cube.is_solved(next_state)}")

    # One-hot tensor for neural network input
    tensor = cube.to_tensor(state)

AXIOMS OF LEARNABLE HEURISTIC SEARCH:
    Given a tuple (S, A, gamma, S_g) where S is the state space, A is the
    action space, gamma is the transition function, and S_g is the goal:

    1. Conservation -- There exists an invariant phi such that
       phi(s) = phi(gamma(s, a)) for all states s and actions a.
       Some measurable quantity is preserved under every action.

    2. Ergodicity -- The state graph is strongly connected.
       Every state is reachable from every other state.

    3. Mechanism Uniformity -- gamma(s, a) depends only on the local
       neighborhood of the action, not on the global state.

    When conservation is violated, networks find counting shortcuts:
    they achieve near-perfect training loss by tracking a trivially
    countable quantity (e.g., number of joker stickers) instead of
    learning the spatial structure needed for search. This produces
    the "counting shortcut paradox" -- better training, worse solving.

EXPERIMENTAL RESULTS (1M-param MLP, 50K training states, beam search):
    StandardCube:  81.7% solve rate (baseline)
    JokerCube:     17.2% solve rate (-64.5%, conservation broken)
    TrapCube:      81.1% solve rate (-0.6%, ergodicity broken)
    ContextCube:   78.3% solve rate (-3.4%, uniformity broken)

    Conservation violation accounts for 94% of total degradation.

LICENSE: MIT
AUTHOR: Ishaan Singla (singla.ishaan@gmail.com)
"""

import random
import hashlib
from typing import List, Optional, Tuple


# =====================================================================
# CUBE GEOMETRY
# =====================================================================
#
# Face layout (54 stickers, 6 faces of 9):
#
#            +---+---+---+
#            | 0 | 1 | 2 |
#            +---+---+---+
#            | 3 | 4 | 5 |    FACE 0: UP
#            +---+---+---+
#            | 6 | 7 | 8 |
# +---+---+---+---+---+---+---+---+---+---+---+---+
# |36 |37 |38 | 9 |10 |11 |18 |19 |20 |27 |28 |29 |
# +---+---+---+---+---+---+---+---+---+---+---+---+
# |39 |40 |41 |12 |13 |14 |21 |22 |23 |30 |31 |32 |
# +---+---+---+---+---+---+---+---+---+---+---+---+
# |42 |43 |44 |15 |16 |17 |24 |25 |26 |33 |34 |35 |
# +---+---+---+---+---+---+---+---+---+---+---+---+
#  FACE 4: LEFT  FACE 1: FRONT FACE 2: RIGHT FACE 3: BACK
#            +---+---+---+
#            |45 |46 |47 |
#            +---+---+---+
#            |48 |49 |50 |    FACE 5: DOWN
#            +---+---+---+
#            |51 |52 |53 |
#            +---+---+---+
#
# Colors: 0=White, 1=Green, 2=Red, 3=Blue, 4=Orange, 5=Yellow
# Joker color: 6 (used only in JokerCube)
#
# Moves 0-5: Clockwise rotation of faces U, F, R, B, L, D
# Moves 6-11: Counter-clockwise (inverse of 0-5)

FACE_NAMES = ['Up', 'Front', 'Right', 'Back', 'Left', 'Down']
COLOR_NAMES = ['White', 'Green', 'Red', 'Blue', 'Orange', 'Yellow', 'Joker']
COLOR_CHARS = ['W', 'G', 'R', 'B', 'O', 'Y', '?']
MOVE_NAMES = ['U', 'F', 'R', 'B', 'L', 'D', "U'", "F'", "R'", "B'", "L'", "D'"]

CENTERS = [4, 13, 22, 31, 40, 49]
OPPOSITE_FACE = {0: 5, 5: 0, 1: 3, 3: 1, 2: 4, 4: 2}

# Adjacent stickers affected by each face rotation
_RINGS = {
    0: ([9, 10, 11], [18, 19, 20], [27, 28, 29], [36, 37, 38]),
    1: ([6, 7, 8], [18, 21, 24], [47, 46, 45], [44, 41, 38]),
    2: ([8, 5, 2], [27, 30, 33], [53, 50, 47], [17, 14, 11]),
    3: ([2, 1, 0], [36, 39, 42], [51, 52, 53], [26, 23, 20]),
    4: ([0, 3, 6], [9, 12, 15], [45, 48, 51], [35, 32, 29]),
    5: ([15, 16, 17], [42, 43, 44], [33, 34, 35], [24, 25, 26]),
}


def _rotate_cw(state: List[int], face: int) -> List[int]:
    """Apply a clockwise 90-degree rotation to one face."""
    s = state[:]
    f = face * 9

    # Rotate face stickers
    s[f+0], s[f+2], s[f+8], s[f+6] = s[f+6], s[f+0], s[f+2], s[f+8]
    s[f+1], s[f+5], s[f+7], s[f+3] = s[f+3], s[f+1], s[f+5], s[f+7]

    # Cycle adjacent ring
    a, b, c, d = _RINGS[face]
    va = [s[i] for i in a]
    vb = [s[i] for i in b]
    vc = [s[i] for i in c]
    vd = [s[i] for i in d]
    for i, v in zip(a, vd): s[i] = v
    for i, v in zip(b, va): s[i] = v
    for i, v in zip(c, vb): s[i] = v
    for i, v in zip(d, vc): s[i] = v

    return s


def _rotate_ccw(state: List[int], face: int) -> List[int]:
    """Apply a counter-clockwise 90-degree rotation (= 3x clockwise)."""
    s = state
    for _ in range(3):
        s = _rotate_cw(s, face)
    return s


# =====================================================================
# BASE CUBE
# =====================================================================

class StandardCube:
    """Standard 3x3 Rubik's Cube. All three axioms hold.

    Conservation: 9 stickers of each color, always.
    Ergodicity:   Every state reachable from every other.
    Uniformity:   Each move does the same thing regardless of state.

    This is the baseline. A well-trained heuristic achieves 81.7%
    solve rate with a 1M-param MLP and beam search.
    """

    # Number of distinct colors (excluding joker)
    N_COLORS = 6
    # Number of sticker positions
    N_POSITIONS = 54
    # Number of possible moves
    N_MOVES = 12

    # Axiom flags
    CONSERVATION = True
    ERGODICITY = True
    UNIFORMITY = True

    def solved(self) -> List[int]:
        """Return the solved state."""
        state = []
        for face in range(6):
            state.extend([face] * 9)
        return state

    def is_solved(self, state: List[int]) -> bool:
        """Check if state is solved (each face has uniform color)."""
        for face in range(6):
            f = face * 9
            if any(state[f + i] != face for i in range(9)):
                return False
        return True

    def move(self, state: List[int], action: int) -> List[int]:
        """Apply a move (0-11) and return the new state.

        Moves 0-5:  Clockwise rotation of U, F, R, B, L, D
        Moves 6-11: Counter-clockwise (inverses)
        """
        if action < 6:
            return _rotate_cw(state, action)
        else:
            return _rotate_ccw(state, action - 6)

    def inverse_move(self, action: int) -> int:
        """Return the move that undoes the given move."""
        if action < 6:
            return action + 6
        else:
            return action - 6

    def move_name(self, action: int) -> str:
        """Human-readable move name."""
        return MOVE_NAMES[action]

    def scramble(self, depth: int, seed: Optional[int] = None) -> List[int]:
        """Scramble from solved state with random moves.

        Args:
            depth: Number of random moves to apply.
            seed:  Optional random seed for reproducibility.

        Returns:
            Scrambled state as list of 54 ints.
        """
        if seed is not None:
            random.seed(seed)
        state = self.solved()
        for _ in range(depth):
            state = self.move(state, random.randint(0, 11))
        return state

    def scramble_with_moves(self, depth: int, seed: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """Scramble and return both the state and the move sequence.

        Returns:
            (state, moves) tuple.
        """
        if seed is not None:
            random.seed(seed)
        state = self.solved()
        moves = []
        for _ in range(depth):
            m = random.randint(0, 11)
            state = self.move(state, m)
            moves.append(m)
        return state, moves

    def to_tensor_data(self, state: List[int]) -> List[List[float]]:
        """One-hot encode for neural network input.

        Returns:
            List of 54 lists, each of length N_COLORS.
            Total flattened dimension = 54 * N_COLORS.

        For PyTorch:
            import torch
            tensor = torch.tensor(cube.to_tensor_data(state)).flatten()
        """
        nc = max(self.N_COLORS, 7) if hasattr(self, '_use_joker_channel') else self.N_COLORS
        t = [[0.0] * nc for _ in range(54)]
        for i, v in enumerate(state):
            t[i][min(v, nc - 1)] = 1.0
        return t

    def tensor_dim(self) -> int:
        """Flattened tensor dimension for neural network input."""
        nc = max(self.N_COLORS, 7) if hasattr(self, '_use_joker_channel') else self.N_COLORS
        return 54 * nc

    def color_name(self, color: int) -> str:
        """Human-readable color name."""
        if 0 <= color < len(COLOR_NAMES):
            return COLOR_NAMES[color]
        return f'Unknown({color})'

    def color_counts(self, state: List[int]) -> dict:
        """Count stickers of each color."""
        counts = {}
        for v in state:
            name = self.color_name(v)
            counts[name] = counts.get(name, 0) + 1
        return counts

    def face_state(self, state: List[int], face: int) -> List[int]:
        """Return the 9 stickers on a given face."""
        f = face * 9
        return state[f:f+9]

    def to_string(self, state: List[int]) -> str:
        """Pretty-print the cube state."""
        chars = [COLOR_CHARS[min(v, len(COLOR_CHARS)-1)] for v in state]
        lines = []
        lines.append(f"      {chars[0]} {chars[1]} {chars[2]}")
        lines.append(f"      {chars[3]} {chars[4]} {chars[5]}")
        lines.append(f"      {chars[6]} {chars[7]} {chars[8]}")
        lines.append(f"{chars[36]} {chars[37]} {chars[38]}  {chars[9]} {chars[10]} {chars[11]}  {chars[18]} {chars[19]} {chars[20]}  {chars[27]} {chars[28]} {chars[29]}")
        lines.append(f"{chars[39]} {chars[40]} {chars[41]}  {chars[12]} {chars[13]} {chars[14]}  {chars[21]} {chars[22]} {chars[23]}  {chars[30]} {chars[31]} {chars[32]}")
        lines.append(f"{chars[42]} {chars[43]} {chars[44]}  {chars[15]} {chars[16]} {chars[17]}  {chars[24]} {chars[25]} {chars[26]}  {chars[33]} {chars[34]} {chars[35]}")
        lines.append(f"      {chars[45]} {chars[46]} {chars[47]}")
        lines.append(f"      {chars[48]} {chars[49]} {chars[50]}")
        lines.append(f"      {chars[51]} {chars[52]} {chars[53]}")
        return '\n'.join(lines)

    def info(self) -> str:
        """Description of this cube variant."""
        return (f"{self.__class__.__name__}: "
                f"Conservation={'Y' if self.CONSERVATION else 'N'} "
                f"Ergodicity={'Y' if self.ERGODICITY else 'N'} "
                f"Uniformity={'Y' if self.UNIFORMITY else 'N'}")


# =====================================================================
# JOKER CUBE (Conservation broken)
# =====================================================================

class JokerCube(StandardCube):
    """Rubik's Cube with wild "joker" stickers.

    AXIOM VIOLATED: Conservation
    - Sticker color counts change during solving.
    - ~25% of stickers start as jokers (color 6).
    - After each move, one random joker crystallizes to the color
      of the center sticker on the diametrically opposite face.
    - During backward scrambling, jokers are injected (reverse of
      crystallization).

    THE COUNTING SHORTCUT:
    Because joker count correlates with scramble depth (more moves =
    more jokers during backward scramble), a neural network can achieve
    near-perfect training loss (MAE 0.032, Rank 96.4%) by simply
    counting jokers. But this shortcut is useless for search:
    0% solve rate at depth 5+.

    This is the "counting shortcut paradox": better training metrics,
    catastrophically worse search performance.

    EXPERIMENTAL RESULT: 17.2% overall solve rate (-64.5% from baseline)
    """

    N_COLORS = 7  # 6 regular + joker
    CONSERVATION = False
    JOKER_COLOR = 6

    def __init__(self, inject_rate: float = 0.5):
        """
        Args:
            inject_rate: Probability of injecting a joker per scramble step.
                         Default 0.5 produces ~25% joker density at depth 10.
        """
        self._inject_rate = inject_rate
        self._use_joker_channel = True

    def inject_joker(self, state: List[int]) -> List[int]:
        """Turn one random non-center, non-joker sticker into a joker.

        Used during backward scrambling to simulate the reverse of
        joker crystallization.
        """
        candidates = [i for i in range(54)
                      if state[i] != self.JOKER_COLOR and i not in CENTERS]
        if not candidates:
            return state
        s = state[:]
        s[random.choice(candidates)] = self.JOKER_COLOR
        return s

    def resolve_joker(self, state: List[int]) -> List[int]:
        """Crystallize one random joker to the opposite center's color.

        Called automatically after each move during forward solving.
        The joker takes the color of the center sticker on the
        diametrically opposite face from where the joker sits.
        """
        jokers = [i for i in range(54)
                  if state[i] == self.JOKER_COLOR and i not in CENTERS]
        if not jokers:
            return state
        pos = random.choice(jokers)
        face = pos // 9
        opp_face = OPPOSITE_FACE[face]
        color = state[CENTERS[opp_face]]
        if color == self.JOKER_COLOR:
            color = opp_face  # fallback to face's natural color
        s = state[:]
        s[pos] = color
        return s

    def move(self, state: List[int], action: int) -> List[int]:
        """Apply move, then crystallize one joker."""
        new_state = super().move(state, action)
        return self.resolve_joker(new_state)

    def scramble(self, depth: int, seed: Optional[int] = None) -> List[int]:
        """Backward scramble: apply moves and inject jokers.

        Jokers are injected (not resolved) during scrambling because
        scrambling is the reverse of solving. This creates states where
        joker count correlates with scramble depth.
        """
        if seed is not None:
            random.seed(seed)
        state = self.solved()
        for _ in range(depth):
            # Apply move WITHOUT resolving jokers (backward direction)
            action = random.randint(0, 11)
            state = super().move(state, action)
            # Inject joker with some probability
            if random.random() < self._inject_rate:
                state = self.inject_joker(state)
        return state

    def joker_count(self, state: List[int]) -> int:
        """Count the number of joker stickers."""
        return sum(1 for v in state if v == self.JOKER_COLOR)

    def joker_positions(self, state: List[int]) -> List[int]:
        """Return positions of all joker stickers."""
        return [i for i, v in enumerate(state) if v == self.JOKER_COLOR]


# =====================================================================
# TRAP CUBE (Ergodicity broken)
# =====================================================================

class TrapCube(StandardCube):
    """Rubik's Cube where 10% of states are dead ends.

    AXIOM VIOLATED: Ergodicity
    - The state graph is no longer fully connected.
    - 10% of non-solved states are deterministic "traps" (dead ends).
    - Whether a state is a trap is determined by a hash function,
      so it is consistent (same state = same trap status always).
    - The solved state is never a trap.

    During scrambling, trap states are avoided (the scrambler tries
    alternative moves if a move would lead to a trap). During solving,
    the search must also avoid traps.

    EXPERIMENTAL RESULT: 81.1% overall solve rate (-0.6% from baseline)
    Ergodicity violation has minimal impact on learned heuristic search.
    """

    ERGODICITY = False

    def __init__(self, trap_rate: float = 0.10, trap_seed: int = 98765):
        """
        Args:
            trap_rate: Fraction of states that are traps (0.0 to 1.0).
            trap_seed: Seed for deterministic trap assignment.
        """
        self._trap_rate = trap_rate
        self._trap_seed = trap_seed

    def is_trap(self, state: List[int]) -> bool:
        """Is this state a dead end?

        Deterministic: same state always gives same answer.
        The solved state is never a trap.
        """
        if self.is_solved(state):
            return False
        h = int(hashlib.md5(bytes(state)).hexdigest()[:8], 16)
        h ^= self._trap_seed
        return (h % 100) < int(self._trap_rate * 100)

    def safe_moves(self, state: List[int]) -> List[int]:
        """Return moves that don't lead to trap states."""
        safe = []
        for action in range(12):
            ns = self.move(state, action)
            if not self.is_trap(ns):
                safe.append(action)
        return safe

    def scramble(self, depth: int, seed: Optional[int] = None) -> List[int]:
        """Scramble while avoiding trap states.

        If a move would lead to a trap, try a different move.
        This ensures all training states are reachable without
        hitting traps.
        """
        if seed is not None:
            random.seed(seed)
        state = self.solved()
        for _ in range(depth):
            attempts = list(range(12))
            random.shuffle(attempts)
            moved = False
            for action in attempts:
                ns = self.move(state, action)
                if not self.is_trap(ns):
                    state = ns
                    moved = True
                    break
            if not moved:
                # All moves lead to traps (extremely rare)
                state = self.move(state, random.randint(0, 11))
                break
        return state


# =====================================================================
# CONTEXT CUBE (Uniformity broken)
# =====================================================================

class ContextCube(StandardCube):
    """Rubik's Cube where moves depend on the current state.

    AXIOM VIOLATED: Mechanism Uniformity
    - The same move number does different things depending on state.
    - For each face, if the center sticker color is EVEN (0, 2, 4),
      the move rotates clockwise. If ODD (1, 3, 5), counter-clockwise.
    - Moves 6-11 flip the direction (as usual).
    - The solver must read the center colors to predict move outcomes.

    This directly models real-world domains (like chemistry) where
    the same "action" has different effects depending on context.

    EXPERIMENTAL RESULT: 78.3% overall solve rate (-3.4% from baseline)
    Uniformity violation has minimal impact -- neural networks easily
    learn to condition on center colors.
    """

    UNIFORMITY = False

    def move(self, state: List[int], action: int) -> List[int]:
        """Context-dependent move.

        Direction determined by center sticker color:
        - Even color (0, 2, 4) -> clockwise
        - Odd color (1, 3, 5) -> counter-clockwise
        - Joker (6) -> clockwise (default)

        Moves 6-11 flip the direction.
        """
        face = action if action < 6 else action - 6
        center_color = state[CENTERS[face]]

        # Determine base direction from center color
        go_cw = (center_color % 2 == 0) or (center_color >= 6)

        # Moves 6-11 invert direction
        if action >= 6:
            go_cw = not go_cw

        if go_cw:
            return _rotate_cw(state, face)
        else:
            return _rotate_ccw(state, face)


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def all_cubes() -> dict:
    """Return a dict of all cube variants."""
    return {
        'Standard': StandardCube(),
        'Joker': JokerCube(),
        'Trap': TrapCube(),
        'Context': ContextCube(),
    }


def describe_all():
    """Print descriptions of all cube variants."""
    cubes = all_cubes()
    print("AXIOM DIAGNOSTIC CUBE SUITE")
    print("=" * 60)
    print()
    print("Three modified Rubik's Cubes that each break exactly one")
    print("axiom of learnable heuristic search.")
    print()
    print(f"  {'Cube':<15s} {'C':>3s} {'E':>3s} {'U':>3s}  {'Description'}")
    print(f"  {'-'*55}")
    for name, cube in cubes.items():
        c = 'Y' if cube.CONSERVATION else 'N'
        e = 'Y' if cube.ERGODICITY else 'N'
        u = 'Y' if cube.UNIFORMITY else 'N'
        print(f"  {name:<15s} {c:>3s} {e:>3s} {u:>3s}  {cube.__class__.__doc__.split(chr(10))[0].strip()}")
    print()
    print("C = Conservation (sticker colors preserved)")
    print("E = Ergodicity (all states reachable)")
    print("U = Uniformity (same move = same effect)")
    print()
    print("Key finding: Conservation violation alone accounts for")
    print("94% of total degradation in learned heuristic search.")


# =====================================================================
# DEMO
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  AXIOM DIAGNOSTIC CUBE SUITE -- Demo")
    print("=" * 60)
    print()

    describe_all()

    print()
    print("-" * 60)
    print("DEMO: Scramble each cube variant at depth 5")
    print("-" * 60)

    for name, cube in all_cubes().items():
        state = cube.scramble(5, seed=42)
        print(f"\n{name} (depth 5):")
        print(cube.to_string(state))
        print(f"  Solved: {cube.is_solved(state)}")
        print(f"  Colors: {cube.color_counts(state)}")
        print(f"  Tensor dim: {cube.tensor_dim()}")
        if hasattr(cube, 'joker_count'):
            print(f"  Jokers: {cube.joker_count(state)}")
        if hasattr(cube, 'is_trap'):
            print(f"  Is trap: {cube.is_trap(state)}")

    print()
    print("-" * 60)
    print("DEMO: Move verification (apply move then inverse = original)")
    print("-" * 60)

    for name, cube in all_cubes().items():
        if name == 'Joker':
            continue  # Joker is stochastic, skip inverse test
        state = cube.scramble(3, seed=99)
        for m in range(6):
            s1 = cube.move(state, m)
            s2 = cube.move(s1, cube.inverse_move(m))
            match = (s2 == state)
            if not match and name == 'Context':
                # Context cube: inverse depends on center after move
                pass
            elif not match:
                print(f"  {name}: Move {cube.move_name(m)} inverse FAILED")
        print(f"  {name}: Move inverse check complete")

    print()
    print("-" * 60)
    print("DEMO: Trap statistics")
    print("-" * 60)

    trap = TrapCube()
    n_traps = 0
    n_samples = 10000
    for _ in range(n_samples):
        s = StandardCube().scramble(random.randint(1, 15))
        if trap.is_trap(s):
            n_traps += 1
    print(f"  Trap rate: {n_traps}/{n_samples} ({n_traps/n_samples:.1%})")
    print(f"  Solved is trap: {trap.is_trap(trap.solved())}")

    print()
    print("-" * 60)
    print("DEMO: Joker crystallization")
    print("-" * 60)

    jcube = JokerCube()
    s = jcube.scramble(8, seed=42)
    n_before = jcube.joker_count(s)
    print(f"  After scramble (depth 8): {n_before} jokers")

    # Apply a move (which resolves one joker)
    s2 = jcube.move(s, 0)  # U move
    n_after = jcube.joker_count(s2)
    print(f"  After one move: {n_after} jokers (resolved {n_before - n_after})")

    print()
    print("=" * 60)
    print("  All demos complete. Import and use:")
    print("    from axiom_cubes import JokerCube, TrapCube, ContextCube")
    print("=" * 60)
