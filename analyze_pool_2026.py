"""
March Madness 2026 Pool Analyzer
Monte Carlo simulation to determine each participant's odds of winning the pool.
Scoring: seed_number x number_of_wins for each of 10 picks.
"""

import csv
import random
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Seed strengths (Bradley-Terry ratings from existing optimizer)
# ---------------------------------------------------------------------------
SEED_STRENGTH = {
    1: 3.50, 2: 2.50, 3: 2.00, 4: 1.65, 5: 1.35, 6: 1.15,
    7: 1.00, 8: 0.90, 9: 0.80, 10: 0.72, 11: 0.65, 12: 0.58,
    13: 0.42, 14: 0.32, 15: 0.22, 16: 0.10
}

# ---------------------------------------------------------------------------
# 2026 Bracket definition
# Bracket seed order: consecutive pairs play in R64
# ---------------------------------------------------------------------------
BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

REGIONS = {
    "East": [
        ("Duke", 1), ("Siena", 16),
        ("Ohio St.", 8), ("TCU", 9),
        ("St. John's", 5), ("Northern Iowa", 12),
        ("Kansas", 4), ("Cal Baptist", 13),
        ("Louisville", 6), ("South Florida", 11),
        ("Michigan St.", 3), ("North Dakota St.", 14),
        ("UCLA", 7), ("UCF", 10),
        ("UConn", 2), ("Furman", 15),
    ],
    "South": [
        ("Florida", 1), ("PVAMU", 16),
        ("Clemson", 8), ("Iowa", 9),
        ("Vanderbilt", 5), ("McNeese", 12),
        ("Nebraska", 4), ("Troy", 13),
        ("North Carolina", 6), ("VCU", 11),
        ("Illinois", 3), ("Penn", 14),
        ("Saint Mary's", 7), ("Texas A&M", 10),
        ("Houston", 2), ("Idaho", 15),
    ],
    "West": [
        ("Arizona", 1), ("Long Island", 16),
        ("Villanova", 8), ("Utah State", 9),
        ("Wisconsin", 5), ("High Point", 12),
        ("Arkansas", 4), ("Hawaii", 13),
        ("BYU", 6), ("NC State", 11),
        ("Gonzaga", 3), ("Kennesaw St.", 14),
        ("Miami (FL)", 7), ("Missouri", 10),
        ("Purdue", 2), ("Queens", 15),
    ],
    "Midwest": [
        ("Michigan", 1), ("Howard", 16),
        ("Georgia", 8), ("Saint Louis", 9),
        ("Texas Tech", 5), ("Akron", 12),
        ("Alabama", 4), ("Hofstra", 13),
        ("Tennessee", 6), ("SMU", 11),
        ("Virginia", 3), ("Wright St.", 14),
        ("Kentucky", 7), ("Santa Clara", 10),
        ("Iowa St.", 2), ("Tennessee St.", 15),
    ],
}

# Note: West region has Texas as 11-seed (play-in) alongside NC State
# South region has Lehigh as 16-seed play-in alongside PVAMU
# Midwest has Miami (OH) as 11-seed (play-in) alongside SMU
# and UMBC as 16-seed play-in alongside Howard
# For simplicity, we treat play-in games as already resolved to the listed team
# The bracket above uses the "main" team that made it to R64

# Additional teams that may appear in picks but need region/seed mapping
# (for teams in play-in spots that might be picked)
EXTRA_TEAM_SEEDS = {
    "Texas": 11,        # West, play-in
    "Lehigh": 16,       # South, play-in
    "Miami (OH)": 11,   # Midwest, play-in
    "UMBC": 16,         # Midwest, play-in
}

# ---------------------------------------------------------------------------
# Build a flat list of all teams with their region and seed
# ---------------------------------------------------------------------------
def build_team_registry():
    registry = {}  # team_name -> (region, seed)
    for region, teams in REGIONS.items():
        for team, seed in teams:
            registry[team] = (region, seed)
    # Add play-in alternates (assign them to the same region slot concept)
    # Texas is West 11, Miami (OH) is Midwest 11
    registry["Texas"] = ("West", 11)
    registry["Miami (OH)"] = ("Midwest", 11)
    registry["Lehigh"] = ("South", 16)
    registry["UMBC"] = ("Midwest", 16)
    return registry

TEAM_REGISTRY = build_team_registry()

# ---------------------------------------------------------------------------
# Simulate a single-elimination bracket for one region
# Returns dict: team -> wins in this region
# ---------------------------------------------------------------------------
def simulate_region(teams_seeds):
    """
    teams_seeds: list of (team_name, seed) in bracket order (16 teams)
    Simulates R64, R32, Sweet 16, Elite 8 for one region.
    Returns (winner_name, wins_dict)
    """
    wins = defaultdict(int)
    bracket = list(teams_seeds)  # list of (name, seed)

    # 4 rounds in a region
    for _ in range(4):
        next_round = []
        for i in range(0, len(bracket), 2):
            t1, s1 = bracket[i]
            t2, s2 = bracket[i + 1]
            str1 = SEED_STRENGTH[s1]
            str2 = SEED_STRENGTH[s2]
            p1 = str1 / (str1 + str2)
            if random.random() < p1:
                wins[t1] += 1
                next_round.append((t1, s1))
            else:
                wins[t2] += 1
                next_round.append((t2, s2))
        bracket = next_round

    return bracket[0], wins  # (champion, wins_dict)

# ---------------------------------------------------------------------------
# Simulate full tournament
# Returns: dict team -> total wins across whole tournament
# ---------------------------------------------------------------------------
def simulate_tournament():
    all_wins = defaultdict(int)

    region_champs = {}
    for region_name, teams in REGIONS.items():
        # Handle play-in: if Texas is listed as West-11, it competes vs NC State
        # For simplicity the bracket already has the play-in winner listed;
        # but picks might name Texas or NC State. We keep the bracket as-is.
        champ, wins = simulate_region(teams)
        region_champs[region_name] = (champ, champ_seed(champ))
        for team, w in wins.items():
            all_wins[team] += w

    # Final Four: East vs West, South vs Midwest
    semifinal_matchups = [
        (region_champs["East"], region_champs["West"]),
        (region_champs["South"], region_champs["Midwest"]),
    ]
    final_four = []
    for (t1, s1), (t2, s2) in semifinal_matchups:
        str1 = SEED_STRENGTH[s1]
        str2 = SEED_STRENGTH[s2]
        p1 = str1 / (str1 + str2)
        if random.random() < p1:
            all_wins[t1] += 1
            final_four.append((t1, s1))
        else:
            all_wins[t2] += 1
            final_four.append((t2, s2))

    # Championship
    (t1, s1), (t2, s2) = final_four
    str1 = SEED_STRENGTH[s1]
    str2 = SEED_STRENGTH[s2]
    p1 = str1 / (str1 + str2)
    if random.random() < p1:
        all_wins[t1] += 1
    else:
        all_wins[t2] += 1

    return all_wins


def champ_seed(team_name):
    if team_name in TEAM_REGISTRY:
        return TEAM_REGISTRY[team_name][1]
    return 8  # fallback


# ---------------------------------------------------------------------------
# Load pool picks
# ---------------------------------------------------------------------------
def load_picks(filepath):
    picks = defaultdict(list)  # participant -> [(seed, team), ...]
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant = row['participant'].strip()
            seed = int(row['seed'])
            team = row['team'].strip()
            picks[participant].append((seed, team))
    return picks


# ---------------------------------------------------------------------------
# Score a participant given tournament wins
# ---------------------------------------------------------------------------
def score_participant(picks, all_wins):
    total = 0
    for seed, team in picks:
        wins = all_wins.get(team, 0)
        total += seed * wins
    return total


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    picks_file = os.path.join(script_dir, "pool_picks_2026.csv")

    picks = load_picks(picks_file)
    participants = sorted(picks.keys())
    n_participants = len(participants)

    NUM_SIMS = 50_000
    print(f"Running {NUM_SIMS:,} Monte Carlo simulations for {n_participants} participants...")

    win_counts = defaultdict(int)
    score_accumulator = defaultdict(list)

    # Also track leverage: for each team, count how often a team's win in a sim
    # coincides with the pool winner's pick changing
    all_teams_in_picks = set()
    for p_picks in picks.values():
        for seed, team in p_picks:
            all_teams_in_picks.add(team)

    team_leverage_scores = defaultdict(float)

    random.seed(42)

    for sim_idx in range(NUM_SIMS):
        all_wins = simulate_tournament()

        scores = {}
        for p in participants:
            scores[p] = score_participant(picks[p], all_wins)

        max_score = max(scores.values())
        winners = [p for p, s in scores.items() if s == max_score]

        # Award fractional wins for ties
        for w in winners:
            win_counts[w] += 1.0 / len(winners)

        for p in participants:
            score_accumulator[p].append(scores[p])

    # ---------------------------------------------------------------------------
    # Compute leverage: simulate with and without each team advancing
    # We estimate leverage as the variance in "winner identity" when team wins vs loses
    # Simplified: for each team in picks, compute |change in win probability| when
    # we condition on that team winning round 1 vs losing round 1.
    # ---------------------------------------------------------------------------
    print("Computing leverage scores (5,000 conditional sims per team)...")

    LEVERAGE_SIMS = 5_000
    leverage_results = {}

    # We'll compute leverage for teams that appear in at least 5 picks
    team_pick_count = defaultdict(int)
    for p_picks in picks.values():
        for seed, team in p_picks:
            team_pick_count[team] += 1

    candidate_teams = [(team, cnt) for team, cnt in team_pick_count.items() if cnt >= 3]
    candidate_teams.sort(key=lambda x: -x[1])
    top_candidates = candidate_teams[:30]  # evaluate top 30 most-picked teams

    for team, pick_count in top_candidates:
        # Find which region this team is in
        if team not in TEAM_REGISTRY:
            continue
        region_name, team_seed = TEAM_REGISTRY[team]

        win_prob_if_wins_r1 = defaultdict(float)
        win_prob_if_loses_r1 = defaultdict(float)

        for condition in ['wins', 'loses']:
            local_win_counts = defaultdict(float)
            for _ in range(LEVERAGE_SIMS):
                # Simulate tournament but force team's R64 result
                all_wins_local = defaultdict(int)
                region_champs_local = {}

                for rname, rteams in REGIONS.items():
                    if rname != region_name:
                        rchamp, rwins = simulate_region(rteams)
                        region_champs_local[rname] = (rchamp, champ_seed(rchamp))
                        for t, w in rwins.items():
                            all_wins_local[t] += w
                    else:
                        # Force team's result in first round
                        modified = list(rteams)
                        # Find team's position
                        team_idx = None
                        for idx, (tname, tseed) in enumerate(modified):
                            if tname == team:
                                team_idx = idx
                                break

                        if team_idx is None:
                            # Team not in this region's bracket listing, simulate normally
                            rchamp, rwins = simulate_region(rteams)
                            region_champs_local[rname] = (rchamp, champ_seed(rchamp))
                            for t, w in rwins.items():
                                all_wins_local[t] += w
                        else:
                            # Determine opponent (consecutive pairs)
                            if team_idx % 2 == 0:
                                opp_idx = team_idx + 1
                            else:
                                opp_idx = team_idx - 1
                            opp_name, opp_seed = modified[opp_idx]

                            wins_local2 = defaultdict(int)
                            bracket_r1 = []
                            for i in range(0, len(modified), 2):
                                t1n, t1s = modified[i]
                                t2n, t2s = modified[i + 1]
                                if {t1n, t2n} == {team, opp_name}:
                                    if condition == 'wins':
                                        wins_local2[team] += 1
                                        bracket_r1.append((team, team_seed))
                                    else:
                                        wins_local2[opp_name] += 1
                                        bracket_r1.append((opp_name, opp_seed))
                                else:
                                    str1 = SEED_STRENGTH[t1s]
                                    str2 = SEED_STRENGTH[t2s]
                                    p1 = str1 / (str1 + str2)
                                    if random.random() < p1:
                                        wins_local2[t1n] += 1
                                        bracket_r1.append((t1n, t1s))
                                    else:
                                        wins_local2[t2n] += 1
                                        bracket_r1.append((t2n, t2s))

                            # Continue remaining 3 rounds in region
                            bracket = bracket_r1
                            for _ in range(3):
                                next_round = []
                                for i in range(0, len(bracket), 2):
                                    t1n, t1s = bracket[i]
                                    t2n, t2s = bracket[i + 1]
                                    str1 = SEED_STRENGTH[t1s]
                                    str2 = SEED_STRENGTH[t2s]
                                    p1 = str1 / (str1 + str2)
                                    if random.random() < p1:
                                        wins_local2[t1n] += 1
                                        next_round.append((t1n, t1s))
                                    else:
                                        wins_local2[t2n] += 1
                                        next_round.append((t2n, t2s))
                                bracket = next_round

                            region_champ_name, region_champ_seed_val = bracket[0]
                            region_champs_local[rname] = (region_champ_name, region_champ_seed_val)
                            for t, w in wins_local2.items():
                                all_wins_local[t] += w

                # Final Four
                sf = [
                    (region_champs_local["East"], region_champs_local["West"]),
                    (region_champs_local["South"], region_champs_local["Midwest"]),
                ]
                ff = []
                for (t1, s1), (t2, s2) in sf:
                    str1 = SEED_STRENGTH[s1]
                    str2 = SEED_STRENGTH[s2]
                    p1 = str1 / (str1 + str2)
                    if random.random() < p1:
                        all_wins_local[t1] += 1
                        ff.append((t1, s1))
                    else:
                        all_wins_local[t2] += 1
                        ff.append((t2, s2))
                (t1, s1), (t2, s2) = ff
                str1 = SEED_STRENGTH[s1]
                str2 = SEED_STRENGTH[s2]
                p1 = str1 / (str1 + str2)
                if random.random() < p1:
                    all_wins_local[t1] += 1
                else:
                    all_wins_local[t2] += 1

                # Score
                scores = {}
                for p in participants:
                    scores[p] = score_participant(picks[p], all_wins_local)
                max_score = max(scores.values())
                winners_local = [p for p, s in scores.items() if s == max_score]
                for w in winners_local:
                    local_win_counts[w] += 1.0 / len(winners_local)

            if condition == 'wins':
                for p in participants:
                    win_prob_if_wins_r1[p] = local_win_counts[p] / LEVERAGE_SIMS
            else:
                for p in participants:
                    win_prob_if_loses_r1[p] = local_win_counts[p] / LEVERAGE_SIMS

        # Leverage = max change in any participant's win prob
        max_delta = max(
            abs(win_prob_if_wins_r1[p] - win_prob_if_loses_r1[p])
            for p in participants
        )
        leverage_results[team] = {
            'pick_count': pick_count,
            'seed': team_seed,
            'max_delta': max_delta,
        }

    # ---------------------------------------------------------------------------
    # Output results
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2026 MARCH MADNESS POOL ANALYSIS")
    print(f"Monte Carlo Simulation Results ({NUM_SIMS:,} runs)")
    print("Scoring: seed × wins  |  10 picks per participant  |  Winner-take-all")
    print("=" * 80)

    # Build results table
    results = []
    for p in participants:
        scores_arr = np.array(score_accumulator[p])
        results.append({
            'participant': p,
            'win_pct': win_counts[p] / NUM_SIMS * 100,
            'avg_score': np.mean(scores_arr),
            'median_score': np.median(scores_arr),
            'p90_score': np.percentile(scores_arr, 90),
            'max_score': np.max(scores_arr),
        })

    results.sort(key=lambda x: -x['win_pct'])

    # Print ranked table
    header = f"{'Rank':<5} {'Participant':<30} {'Win%':>7} {'Avg Pts':>9} {'Median':>8} {'P90':>7} {'Max':>6}"
    print(header)
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<5} {r['participant']:<30} {r['win_pct']:>6.2f}% "
            f"{r['avg_score']:>9.1f} {r['median_score']:>8.1f} "
            f"{r['p90_score']:>7.1f} {r['max_score']:>6.0f}"
        )

    print("\n" + "=" * 80)
    print("TOP 5 LEVERAGE TEAMS")
    print("(Teams whose Round 1 result most swings pool win probabilities)")
    print("=" * 80)
    leverage_list = [
        (team, info) for team, info in leverage_results.items()
    ]
    leverage_list.sort(key=lambda x: -x[1]['max_delta'])
    top5 = leverage_list[:5]
    print(f"{'Team':<20} {'Seed':>5} {'# Picked':>9} {'Max Win% Swing':>15}")
    print("-" * 55)
    for team, info in top5:
        print(
            f"{team:<20} {info['seed']:>5} {info['pick_count']:>9} "
            f"{info['max_delta']*100:>14.2f}%"
        )

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("- Simulation uses Bradley-Terry model with seed-based strengths")
    print("- Final Four: East vs West, South vs Midwest")
    print("- Kansas treated as seed 4, East region")
    print("- Play-in teams (Texas/NC State at West-11, Miami OH/SMU at Midwest-11,")
    print("  PVAMU/Lehigh at South-16, Howard/UMBC at Midwest-16) use listed bracket team")
    print(f"- Total participants: {n_participants}")
    total_pct = sum(r['win_pct'] for r in results)
    print(f"- Win% sum check: {total_pct:.2f}% (should be ~100%)")


if __name__ == "__main__":
    main()
