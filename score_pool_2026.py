"""
Score the 2026 pool based on confirmed first round results (as of March 20, 2026 evening).
Scoring: seed × wins.

CONFIRMED R1 RESULTS (27 of 32 games completed):
Still pending (late Friday night games):
  - Kansas vs Cal Baptist
  - UCLA vs UCF
  - UConn vs Furman
  - Florida vs PVAMU
  - Miami (FL) vs Missouri
"""

import csv
from collections import defaultdict

# ---------------------------------------------------------------------------
# Confirmed Round 1 winners (team_name -> wins_so_far)
# Teams still playing or pending are NOT included (treat as 0 wins until known)
# ---------------------------------------------------------------------------

# Format: team_name: [seed, wins_confirmed, still_alive]
# wins_confirmed = wins earned so far
# still_alive = True if still in tournament (advanced past R1)

RESULTS = {
    # East
    "Duke":           {"seed": 1,  "wins": 1, "alive": True},
    "Siena":          {"seed": 16, "wins": 0, "alive": False},
    "TCU":            {"seed": 9,  "wins": 1, "alive": True},   # beat Ohio St.
    "Ohio St.":       {"seed": 8,  "wins": 0, "alive": False},
    "St. John's":     {"seed": 5,  "wins": 1, "alive": True},   # beat Northern Iowa
    "Northern Iowa":  {"seed": 12, "wins": 0, "alive": False},
    "Kansas":         {"seed": 4,  "wins": None, "alive": None},  # PENDING
    "Cal Baptist":    {"seed": 13, "wins": None, "alive": None},  # PENDING
    "Louisville":     {"seed": 6,  "wins": 1, "alive": True},   # beat South Florida
    "South Florida":  {"seed": 11, "wins": 0, "alive": False},
    "Michigan St.":   {"seed": 3,  "wins": 1, "alive": True},   # beat North Dakota St.
    "North Dakota St.": {"seed": 14, "wins": 0, "alive": False},
    "UCLA":           {"seed": 7,  "wins": None, "alive": None},  # PENDING
    "UCF":            {"seed": 10, "wins": None, "alive": None},  # PENDING
    "UConn":          {"seed": 2,  "wins": None, "alive": None},  # PENDING
    "Furman":         {"seed": 15, "wins": None, "alive": None},  # PENDING

    # South
    "Florida":        {"seed": 1,  "wins": None, "alive": None},  # PENDING
    "PVAMU":          {"seed": 16, "wins": None, "alive": None},  # PENDING
    "Iowa":           {"seed": 9,  "wins": 1, "alive": True},    # beat Clemson
    "Clemson":        {"seed": 8,  "wins": 0, "alive": False},
    "Vanderbilt":     {"seed": 5,  "wins": 1, "alive": True},    # beat McNeese
    "McNeese":        {"seed": 12, "wins": 0, "alive": False},
    "Nebraska":       {"seed": 4,  "wins": 1, "alive": True},    # beat Troy
    "Troy":           {"seed": 13, "wins": 0, "alive": False},
    "VCU":            {"seed": 11, "wins": 1, "alive": True},    # beat North Carolina OT
    "North Carolina": {"seed": 6,  "wins": 0, "alive": False},
    "Illinois":       {"seed": 3,  "wins": 1, "alive": True},    # beat Penn
    "Penn":           {"seed": 14, "wins": 0, "alive": False},
    "Texas A&M":      {"seed": 10, "wins": 1, "alive": True},    # beat Saint Mary's
    "Saint Mary's":   {"seed": 7,  "wins": 0, "alive": False},
    "Houston":        {"seed": 2,  "wins": 1, "alive": True},    # beat Idaho
    "Idaho":          {"seed": 15, "wins": 0, "alive": False},

    # West
    "Arizona":        {"seed": 1,  "wins": 1, "alive": True},    # beat Long Island
    "Long Island":    {"seed": 16, "wins": 0, "alive": False},
    "Utah State":     {"seed": 9,  "wins": 1, "alive": True},    # beat Villanova
    "Villanova":      {"seed": 8,  "wins": 0, "alive": False},
    "Wisconsin":      {"seed": 5,  "wins": 0, "alive": False},   # lost to High Point
    "High Point":     {"seed": 12, "wins": 1, "alive": True},    # beat Wisconsin
    "Arkansas":       {"seed": 4,  "wins": 1, "alive": True},    # beat Hawaii
    "Hawaii":         {"seed": 13, "wins": 0, "alive": False},
    "Texas":          {"seed": 11, "wins": 1, "alive": True},    # beat BYU
    "BYU":            {"seed": 6,  "wins": 0, "alive": False},
    "Gonzaga":        {"seed": 3,  "wins": 1, "alive": True},    # beat Kennesaw St.
    "Kennesaw St.":   {"seed": 14, "wins": 0, "alive": False},
    "Miami (FL)":     {"seed": 7,  "wins": None, "alive": None},  # PENDING
    "Missouri":       {"seed": 10, "wins": None, "alive": None},  # PENDING
    "Purdue":         {"seed": 2,  "wins": 1, "alive": True},    # beat Queens
    "Queens":         {"seed": 15, "wins": 0, "alive": False},

    # Midwest
    "Michigan":       {"seed": 1,  "wins": 1, "alive": True},    # beat Howard
    "Howard":         {"seed": 16, "wins": 0, "alive": False},
    "Saint Louis":    {"seed": 9,  "wins": 1, "alive": True},    # beat Georgia
    "Georgia":        {"seed": 8,  "wins": 0, "alive": False},
    "Texas Tech":     {"seed": 5,  "wins": 1, "alive": True},    # beat Akron
    "Akron":          {"seed": 12, "wins": 0, "alive": False},
    "Alabama":        {"seed": 4,  "wins": 1, "alive": True},    # beat Hofstra
    "Hofstra":        {"seed": 13, "wins": 0, "alive": False},
    "Virginia":       {"seed": 3,  "wins": 1, "alive": True},    # beat Wright St.
    "Wright St.":     {"seed": 14, "wins": 0, "alive": False},
    "Tennessee":      {"seed": 6,  "wins": 1, "alive": True},    # beat Miami (OH)
    "Miami (OH)":     {"seed": 11, "wins": 0, "alive": False},
    "Kentucky":       {"seed": 7,  "wins": 1, "alive": True},    # beat Santa Clara OT
    "Santa Clara":    {"seed": 10, "wins": 0, "alive": False},
    "Iowa St.":       {"seed": 2,  "wins": 1, "alive": True},    # beat Tennessee St.
    "Tennessee St.":  {"seed": 15, "wins": 0, "alive": False},
}

# Load picks
picks = defaultdict(list)
with open("pool_picks_2026.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        picks[row["participant"]].append((int(row["seed"]), row["team"]))

# Calculate scores
scores = {}
pending_teams = {}
for participant, team_picks in picks.items():
    total = 0
    pending = []
    for seed, team in team_picks:
        r = RESULTS.get(team)
        if r is None:
            pending.append(team)
            continue
        if r["wins"] is None:
            pending.append(team)
            continue
        total += seed * r["wins"]
    scores[participant] = total
    pending_teams[participant] = pending

# Sort by score descending
ranked = sorted(scores.items(), key=lambda x: -x[1])

print(f"\n{'='*65}")
print(f"  2026 MARCH MADNESS POOL STANDINGS")
print(f"  After Round 1 (27/32 games complete — 5 games still pending)")
print(f"{'='*65}")
print(f"{'Rank':<5} {'Participant':<28} {'Pts':>5}  {'Pending Picks'}")
print(f"{'-'*65}")

for i, (participant, pts) in enumerate(ranked, 1):
    pend = pending_teams[participant]
    pend_str = ", ".join(pend) if pend else "—"
    print(f"  {i:<4} {participant:<28} {pts:>5}  {pend_str}")

print(f"\n{'='*65}")
print("PENDING GAMES (5 late Friday night games):")
print("  Kansas vs Cal Baptist")
print("  UCLA vs UCF")
print("  UConn vs Furman")
print("  Miami (FL) vs Missouri")
print("  Florida vs PVAMU")
print(f"{'='*65}")

# Show what's at stake for pending games
print("\nPENDING GAME IMPACT (pts at stake per participant):")
pending_games = [
    ("Kansas", 4), ("Cal Baptist", 13),
    ("UCLA", 7), ("UCF", 10),
    ("UConn", 2), ("Furman", 15),
    ("Miami (FL)", 7), ("Missouri", 10),
    ("Florida", 1), ("PVAMU", 16),
]
pending_team_names = {t for t, s in pending_games}

for team, seed in pending_games:
    owners = [p for p, team_list in picks.items() if any(t == team for s, t in team_list)]
    if owners:
        print(f"  {team} (seed {seed}, {seed} pts/win): picked by {len(owners)} → {', '.join(owners)}")
