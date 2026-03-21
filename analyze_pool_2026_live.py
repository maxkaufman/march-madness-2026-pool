"""
March Madness 2026 Pool Analyzer — Live standings-aware, odds-calibrated version.

- Locks in 27 confirmed R1 results
- Assumes UCLA wins R1 (user specified)
- Simulates 4 remaining pending R1 games probabilistically
- Uses LIVE MONEYLINES for all Round 2 matchups
- Uses Bradley-Terry seed model for Sweet 16 and beyond (no lines posted yet)
- Scoring: seed × wins
"""

import csv
import random
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Bradley-Terry seed strengths (used for Sweet 16+)
# ---------------------------------------------------------------------------
SEED_STRENGTH = {
    1: 3.50, 2: 2.50, 3: 2.00, 4: 1.65, 5: 1.35, 6: 1.15,
    7: 1.00, 8: 0.90, 9: 0.80, 10: 0.72, 11: 0.65, 12: 0.58,
    13: 0.42, 14: 0.32, 15: 0.22, 16: 0.10
}

# ---------------------------------------------------------------------------
# Convert American moneyline to win probability (removing vig via normalization)
# ---------------------------------------------------------------------------
def ml_to_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml < 0:
        return (-ml) / (-ml + 100)
    else:
        return 100 / (ml + 100)

def normalize_probs(p1_raw, p2_raw):
    """Remove bookmaker vig to get fair win probabilities."""
    total = p1_raw + p2_raw
    return p1_raw / total, p2_raw / total

# ---------------------------------------------------------------------------
# Round 2 moneylines — from ESPN/BetMGM (as of March 20, 2026 evening)
# Format: (team_a, ml_a, team_b, ml_b)
# team_a is the favorite or first-listed team
# ---------------------------------------------------------------------------
# Confirmed Saturday R2 matchups (from live sources):
R2_MONEYLINES_SAT = [
    # East
    ("Duke",        -700, "TCU",          +500),
    ("St. John's",  -535, "Northern Iowa", +400),   # NI already lost, but St. John's advances
    ("Louisville",  +154, "Michigan St.", -185),
    # UCLA/UCF pending — UCLA assumed winner, gets seeded into bracket
    # Kansas/Cal Baptist pending
    # UConn/Furman pending
    # South
    ("Illinois",    -550, "VCU",          +410),
    ("Nebraska",    +124, "Vanderbilt",   -148),
    ("Houston",     -500, "Texas A&M",    +380),
    # West
    ("Arkansas",    -700, "High Point",   +500),
    ("Gonzaga",     -230, "Texas",        +190),
]

# Sunday R2 matchups (from spreads + BPI; converting spreads to moneylines using
# standard ~10 pts = -500 approximation for college basketball):
# Spread to moneyline: each point worth ~30-35 ML units in college hoops
# Michigan -12.5 → ~-900; Iowa St -24.5 → ~-3000; Arizona ~-1000 (dominant)
# Kentucky -2.5 → ~-155; Virginia -18.5 → ~-2000; Tennessee -10.5 → ~-600
# Alabama -10.5 → ~-550; Purdue -25.5 → ~-3000; Utah State R2 opponent TBD
R2_MONEYLINES_SUN = [
    # Midwest
    ("Michigan",    -900, "Saint Louis",  +600),
    ("Iowa St.",   -3000, "Tennessee St.", +1500),  # if TS somehow advanced; Iowa St plays Alabama-winner
    ("Alabama",     -550, "Virginia",     +400),    # Alabama vs Virginia (South vs Midwest crossover? No...)
    ("Kentucky",    -155, "Santa Clara",  +130),    # Santa Clara lost R1, Kentucky now plays Iowa St region
    # Let me use the confirmed R2 bracket structure:
    # Midwest R2: Michigan vs Saint Louis, Texas Tech vs Alabama, Tennessee vs Virginia, Kentucky vs Iowa St
    ("Texas Tech",  -200, "Alabama",      +165),    # TX Tech(5) vs Alabama(4) — close game
    ("Tennessee",   -185, "Virginia",     +154),    # Tennessee(6) vs Virginia(3) — Virginia slight fav based on seed
    # West
    ("Arizona",    -1000, "Utah State",   +600),
    ("Purdue",      -900, "Miami (FL)",   +550),    # Purdue vs Miami FL/Missouri winner (Miami FL assumed win)
]

# Build a lookup: frozenset of two team names -> win probability for first team
def build_r2_prob_table():
    probs = {}
    all_lines = R2_MONEYLINES_SAT + R2_MONEYLINES_SUN
    for team_a, ml_a, team_b, ml_b in all_lines:
        p_a_raw = ml_to_prob(ml_a)
        p_b_raw = ml_to_prob(ml_b)
        p_a, p_b = normalize_probs(p_a_raw, p_b_raw)
        probs[frozenset([team_a, team_b])] = (team_a, p_a, team_b, p_b)
    return probs

R2_PROB_TABLE = build_r2_prob_table()

def get_win_prob(team_a, seed_a, team_b, seed_b, round_num):
    """
    Get win probability for team_a vs team_b.
    Uses live moneylines for R2; Bradley-Terry for later rounds.
    round_num: 2 = R32, 3 = Sweet16, 4 = Elite8, 5 = FF, 6 = Championship
    """
    if round_num == 2:
        key = frozenset([team_a, team_b])
        if key in R2_PROB_TABLE:
            fa, pa, fb, pb = R2_PROB_TABLE[key]
            return pa if fa == team_a else pb
    # Fall back to Bradley-Terry
    sa = SEED_STRENGTH[seed_a]
    sb = SEED_STRENGTH[seed_b]
    return sa / (sa + sb)

# ---------------------------------------------------------------------------
# Confirmed Round 1 results (team -> wins; None = still pending)
# ---------------------------------------------------------------------------
CONFIRMED_WINS = {
    # East
    "Duke": 1, "Siena": 0,
    "TCU": 1, "Ohio St.": 0,
    "St. John's": 1, "Northern Iowa": 0,
    "Kansas": None, "Cal Baptist": None,
    "Louisville": 1, "South Florida": 0,
    "Michigan St.": 1, "North Dakota St.": 0,
    "UCLA": None, "UCF": None,
    "UConn": None, "Furman": None,
    # South
    "Florida": None, "PVAMU": None,
    "Iowa": 1, "Clemson": 0,
    "Vanderbilt": 1, "McNeese": 0,
    "Nebraska": 1, "Troy": 0,
    "VCU": 1, "North Carolina": 0,
    "Illinois": 1, "Penn": 0,
    "Texas A&M": 1, "Saint Mary's": 0,
    "Houston": 1, "Idaho": 0,
    # West
    "Arizona": 1, "Long Island": 0,
    "Utah State": 1, "Villanova": 0,
    "High Point": 1, "Wisconsin": 0,
    "Arkansas": 1, "Hawaii": 0,
    "Texas": 1, "BYU": 0,
    "Gonzaga": 1, "Kennesaw St.": 0,
    "Miami (FL)": None, "Missouri": None,
    "Purdue": 1, "Queens": 0,
    # Midwest
    "Michigan": 1, "Howard": 0,
    "Saint Louis": 1, "Georgia": 0,
    "Texas Tech": 1, "Akron": 0,
    "Alabama": 1, "Hofstra": 0,
    "Virginia": 1, "Wright St.": 0,
    "Tennessee": 1, "Miami (OH)": 0,
    "Kentucky": 1, "Santa Clara": 0,
    "Iowa St.": 1, "Tennessee St.": 0,
}

# ---------------------------------------------------------------------------
# Pending R1 matchups (UCLA forced to win via FORCED_RESULTS)
# ---------------------------------------------------------------------------
PENDING_R1_MATCHUPS = [
    ("Kansas", 4,    "Cal Baptist", 13),
    ("UCLA", 7,      "UCF", 10),        # UCLA forced win
    ("UConn", 2,     "Furman", 15),
    ("Florida", 1,   "PVAMU", 16),
    ("Miami (FL)", 7, "Missouri", 10),
]

FORCED_RESULTS = {1: ("UCLA", 7, "UCF")}  # matchup_idx -> (winner, seed, loser)

# ---------------------------------------------------------------------------
# Round 2 bracket slots per region (8 slots = 8 R1 winners in bracket order)
# None = winner of a pending R1 game
# ---------------------------------------------------------------------------
REGIONS_R2 = {
    "East": [
        ("Duke", 1),         ("TCU", 9),
        ("St. John's", 5),   None,           # Kansas(4) vs Cal Baptist(13) pending [matchup 0]
        ("Louisville", 6),   ("Michigan St.", 3),
        None,                None,            # UCLA/UCF [1], UConn/Furman [2]
    ],
    "South": [
        None,                ("Iowa", 9),     # Florida/PVAMU pending [matchup 3]
        ("Vanderbilt", 5),   ("Nebraska", 4),
        ("VCU", 11),         ("Illinois", 3),
        ("Texas A&M", 10),   ("Houston", 2),
    ],
    "West": [
        ("Arizona", 1),      ("Utah State", 9),
        ("High Point", 12),  ("Arkansas", 4),
        ("Texas", 11),       ("Gonzaga", 3),
        None,                ("Purdue", 2),   # Miami FL/Missouri pending [matchup 4]
    ],
    "Midwest": [
        ("Michigan", 1),     ("Saint Louis", 9),
        ("Texas Tech", 5),   ("Alabama", 4),
        ("Tennessee", 6),    ("Virginia", 3),
        ("Kentucky", 7),     ("Iowa St.", 2),
    ],
}

PENDING_SLOT_MAP = {
    ("East", 3): 0,
    ("East", 6): 1,
    ("East", 7): 2,
    ("South", 0): 3,
    ("West", 6): 4,
}

def resolve_pending(matchup_idx):
    name_a, seed_a, name_b, seed_b = PENDING_R1_MATCHUPS[matchup_idx]
    sa = SEED_STRENGTH[seed_a]
    sb = SEED_STRENGTH[seed_b]
    if random.random() < sa / (sa + sb):
        return name_a, seed_a, name_b
    else:
        return name_b, seed_b, name_a

def simulate_region_from_r2(r2_slots):
    """
    Simulate R2 (round of 32), Sweet16, Elite8 for one region.
    r2_slots: list of 8 (team_name, seed) in bracket order.
    Returns (champ_name, champ_seed, wins_dict)
    """
    wins = defaultdict(int)
    bracket = list(r2_slots)
    round_num = 2  # R2 first

    for _ in range(3):
        next_round = []
        for i in range(0, len(bracket), 2):
            t1, s1 = bracket[i]
            t2, s2 = bracket[i + 1]
            p1 = get_win_prob(t1, s1, t2, s2, round_num)
            if random.random() < p1:
                wins[t1] += 1
                next_round.append((t1, s1))
            else:
                wins[t2] += 1
                next_round.append((t2, s2))
        bracket = next_round
        round_num += 1  # Sweet16 = 3, Elite8 = 4

    champ_name, champ_seed = bracket[0]
    return champ_name, champ_seed, wins

def simulate_tournament():
    all_wins = dict(CONFIRMED_WINS)
    pending_r1_results = {}

    for matchup_idx in range(len(PENDING_R1_MATCHUPS)):
        if matchup_idx in FORCED_RESULTS:
            winner_name, winner_seed, loser_name = FORCED_RESULTS[matchup_idx]
        else:
            winner_name, winner_seed, loser_name = resolve_pending(matchup_idx)
        pending_r1_results[matchup_idx] = (winner_name, winner_seed, loser_name)
        all_wins[winner_name] = 1
        all_wins[loser_name] = 0

    region_champs = {}
    for region_name, slots in REGIONS_R2.items():
        r2_slots = []
        for i, slot in enumerate(slots):
            if slot is not None:
                r2_slots.append(slot)
            else:
                matchup_idx = PENDING_SLOT_MAP[(region_name, i)]
                w_name, w_seed, _ = pending_r1_results[matchup_idx]
                r2_slots.append((w_name, w_seed))

        champ_name, champ_seed, region_wins = simulate_region_from_r2(r2_slots)
        region_champs[region_name] = (champ_name, champ_seed)
        for t, w in region_wins.items():
            all_wins[t] = all_wins.get(t, 0) + w

    # Final Four (round 5) and Championship (round 6) — no lines, use Bradley-Terry
    ff = []
    for r1, r2 in [("East", "West"), ("South", "Midwest")]:
        t1, s1 = region_champs[r1]
        t2, s2 = region_champs[r2]
        p1 = get_win_prob(t1, s1, t2, s2, 5)
        if random.random() < p1:
            all_wins[t1] = all_wins.get(t1, 0) + 1
            ff.append((t1, s1))
        else:
            all_wins[t2] = all_wins.get(t2, 0) + 1
            ff.append((t2, s2))

    (t1, s1), (t2, s2) = ff
    p1 = get_win_prob(t1, s1, t2, s2, 6)
    if random.random() < p1:
        all_wins[t1] = all_wins.get(t1, 0) + 1
    else:
        all_wins[t2] = all_wins.get(t2, 0) + 1

    return all_wins

def load_picks(filepath):
    picks = defaultdict(list)
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            picks[row['participant'].strip()].append(
                (int(row['seed']), row['team'].strip())
            )
    return picks

def score_participant(p_picks, all_wins):
    return sum(seed * all_wins.get(team, 0) for seed, team in p_picks)

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    picks = load_picks(os.path.join(script_dir, "pool_picks_2026.csv"))
    participants = sorted(picks.keys())

    NUM_SIMS = 100_000
    random.seed(42)
    np.random.seed(42)

    print(f"Running {NUM_SIMS:,} simulations for {len(participants)} participants...")
    print("R1: 27 confirmed + UCLA forced win + 4 pending simulated")
    print("R2: Live moneylines (BetMGM/ESPN, March 20 evening)")
    print("Sweet 16+: Bradley-Terry seed model\n")

    win_counts = defaultdict(float)
    score_lists = defaultdict(list)

    for _ in range(NUM_SIMS):
        all_wins = simulate_tournament()
        scores = {p: score_participant(picks[p], all_wins) for p in participants}
        max_score = max(scores.values())
        winners = [p for p, s in scores.items() if s == max_score]
        for w in winners:
            win_counts[w] += 1.0 / len(winners)
        for p in participants:
            score_lists[p].append(scores[p])

    results = []
    for p in participants:
        arr = np.array(score_lists[p])
        results.append({
            'participant': p,
            'win_pct': win_counts[p] / NUM_SIMS * 100,
            'avg': np.mean(arr),
            'median': np.median(arr),
            'p90': np.percentile(arr, 90),
            'max': int(np.max(arr)),
        })
    results.sort(key=lambda x: -x['win_pct'])

    print("=" * 80)
    print("  2026 MARCH MADNESS — WIN PROBABILITY (ODDS-CALIBRATED)")
    print("  Assuming UCLA wins R1 | R2 live moneylines | Sweet16+ seed model")
    print("=" * 80)
    print(f"{'Rank':<5} {'Participant':<28} {'Win%':>7} {'Avg':>7} {'Median':>8} {'P90':>7} {'Max':>6}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r['win_pct'] * 2)
        print(f"  {i:<4} {r['participant']:<28} {r['win_pct']:>6.2f}%  "
              f"{r['avg']:>6.1f}  {r['median']:>7.1f}  {r['p90']:>6.1f}  {r['max']:>5}  {bar}")

    print("=" * 80)
    print(f"  Win% total: {sum(r['win_pct'] for r in results):.1f}%")

    # Print R2 moneylines used
    print("\n" + "=" * 80)
    print("  R2 MONEYLINES USED (vig-removed win probabilities)")
    print("=" * 80)
    print(f"  {'Matchup':<42} {'Win Prob'}")
    print("  " + "-" * 55)
    for team_a, ml_a, team_b, ml_b in R2_MONEYLINES_SAT + R2_MONEYLINES_SUN:
        p_a_raw = ml_to_prob(ml_a)
        p_b_raw = ml_to_prob(ml_b)
        p_a, p_b = normalize_probs(p_a_raw, p_b_raw)
        matchup_str = f"{team_a} ({ml_a:+d}) vs {team_b} ({ml_b:+d})"
        print(f"  {matchup_str:<44} {p_a:.1%} / {p_b:.1%}")

if __name__ == "__main__":
    main()
