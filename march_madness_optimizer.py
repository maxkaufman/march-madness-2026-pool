#!/usr/bin/env python3
"""
March Madness Pool Optimizer — Winner Take All
===============================================
Scoring rules: Pick 10 teams from the field.
  Points = seed_number × number_of_wins in the tournament.
  e.g. an 11-seed that wins 2 games = 22 points.

Strategy: In a 50-person winner-take-all pool you don't want to maximize
*expected* score — you want to maximize P(your score > everyone else's).
This means deliberately taking on high-seed upset picks that other players
are unlikely to have, giving you a unique score spike if they run.

Pipeline:
  1. Fetch live bracket from ESPN public API (falls back to bracket.json or seed labels)
  2. Optionally calibrate team strengths from live moneylines (The Odds API)
  3. Monte Carlo simulate the tournament 50,000 times
  4. Model opponents using pick probabilities calibrated from 2025 pool data
     (47 players): base rate by seed × moneyline multiplier (ratio^1.8)
  5. Simulated annealing to find the 10 teams that maximize P(you win)

Usage:
    pip install -r requirements.txt
    python march_madness_optimizer.py
    python march_madness_optimizer.py --odds-api-key YOUR_KEY
    python march_madness_optimizer.py --simulations 100000 --pool-size 50
    python march_madness_optimizer.py --no-fetch   # skip ESPN, use bracket.json
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─── Defaults ────────────────────────────────────────────────────────────────

POOL_SIZE       = 50
PORTFOLIO_SIZE  = 10
N_SIMULATIONS   = 50_000
SA_STEPS        = 4_000

# Standard NCAA bracket seed order within each region
# (consecutive pairs play each other in R64)
BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

REGIONS = ["East", "West", "South", "Midwest"]

# Bradley-Terry strength ratings by seed, calibrated to historical NCAA win rates.
# P(seed_A beats seed_B) = strength_A / (strength_A + strength_B)
# You can override per-team strengths via moneylines (--odds-api-key).
SEED_STRENGTH = {
    1: 3.50,
    2: 2.50,
    3: 2.00,
    4: 1.65,
    5: 1.35,
    6: 1.15,
    7: 1.00,
    8: 0.90,
    9: 0.80,
    10: 0.72,
    11: 0.65,
    12: 0.58,
    13: 0.42,
    14: 0.32,
    15: 0.22,
    16: 0.10,
}

# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Team:
    name: str
    seed: int
    region: str
    idx: int = 0           # global index (assigned after bracket is built)
    strength: float = 0.0  # Bradley-Terry strength (overridden by odds if available)

    def __post_init__(self):
        if self.strength == 0.0:
            self.strength = SEED_STRENGTH.get(self.seed, 0.5)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Team) and self.name == other.name

    def __repr__(self):
        return f"({self.seed}) {self.name}"

# ─── Bracket Construction ─────────────────────────────────────────────────────

def build_default_bracket() -> dict[str, list[Team]]:
    """Seed-labeled placeholder bracket. Replace names via bracket.json."""
    bracket = {}
    for region in REGIONS:
        bracket[region] = [
            Team(name=f"{region} {seed}-seed", seed=seed, region=region)
            for seed in BRACKET_SEED_ORDER
        ]
    return bracket


def load_bracket_json(path: str = "bracket.json") -> Optional[dict[str, list[Team]]]:
    """Load a user-edited bracket.json file."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        bracket = {}
        for region in REGIONS:
            teams_raw = data.get(region, [])
            bracket[region] = [
                Team(name=t["name"], seed=int(t["seed"]), region=region,
                     strength=float(t.get("strength", 0.0)))
                for t in teams_raw
            ]
            if len(bracket[region]) != 16:
                print(f"  Warning: {region} has {len(bracket[region])} teams (expected 16)")
        return bracket
    except Exception as e:
        print(f"  bracket.json parse error: {e}")
        return None


def save_bracket_json(bracket: dict[str, list[Team]], path: str = "bracket.json"):
    data = {
        region: [{"name": t.name, "seed": t.seed, "region": t.region, "strength": t.strength}
                 for t in teams]
        for region, teams in bracket.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fetch_espn_bracket() -> Optional[dict[str, list[Team]]]:
    """
    Try ESPN's unofficial public API to pull the current NCAA tournament bracket.
    Tries several endpoint patterns since ESPN changes them yearly.
    """
    if not HAS_REQUESTS:
        return None

    # Try fetching tournament list to find current tournament ID
    endpoints = [
        "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/tournaments",
        "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events?limit=50&dates=20260301-20260405",
    ]

    for url in endpoints:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            result = _parse_espn_response(data)
            if result:
                return result
        except Exception:
            continue
    return None


def _parse_espn_response(data: dict) -> Optional[dict[str, list[Team]]]:
    """Best-effort parse of various ESPN API shapes."""
    # Shape 1: { tournaments: [ { bracket: { groups: [...] } } ] }
    tournaments = data.get("tournaments", [data])
    for tourney in tournaments:
        bracket_data = tourney.get("bracket", tourney)
        groups = bracket_data.get("groups", [])
        result = _parse_groups(groups)
        if result:
            return result
    return None


def _parse_groups(groups: list) -> Optional[dict[str, list[Team]]]:
    bracket: dict[str, list[Team]] = {}
    for group in groups:
        region_name = group.get("name", "")
        # Match one of our known region names
        region = next((r for r in REGIONS if r.lower() in region_name.lower()), None)
        if not region:
            continue
        seeds_map: dict[int, str] = {}
        for entry in group.get("seeds", []):
            seed_num = int(entry.get("displayOrder", entry.get("seed", 0)))
            teams_list = entry.get("teams", [{}])
            if teams_list:
                name = teams_list[0].get("displayName", teams_list[0].get("name", ""))
                if name:
                    seeds_map[seed_num] = name
        if not seeds_map:
            continue
        bracket[region] = [
            Team(name=seeds_map.get(s, f"{region} {s}-seed"), seed=s, region=region)
            for s in BRACKET_SEED_ORDER
        ]
    return bracket if len(bracket) == 4 else None

# ─── Live Odds Integration ────────────────────────────────────────────────────

import re as _re

def _norm(name: str) -> str:
    """Normalize a team name for fuzzy matching.

    Handles:
      - Special chars / apostrophes  (Hawai'i → hawaii)
      - Parentheticals               (Miami (OH) → miami)
      - 'St' mid/end-of-name → State (Michigan St → michigan state)
        but NOT at the start        (St. John's stays st johns)
      - Strips nickname suffix words that appear after the school name
        (handled by the multi-strategy caller, not here)
    """
    s = name.lower()
    s = _re.sub(r"['\u2018\u2019\u02bb`]", '', s)   # apostrophes + Hawaiian ʻokina
    s = _re.sub(r'\(.*?\)', '', s)              # parentheticals
    s = _re.sub(r'[^a-z0-9& ]', ' ', s)        # periods, slashes, etc.
    s = _re.sub(r'\s+', ' ', s).strip()
    # Expand "st" → "state" only when NOT the first token
    tokens = s.split()
    if len(tokens) > 1:
        tokens = [tokens[0]] + ['state' if t == 'st' else t for t in tokens[1:]]
    return ' '.join(tokens)


def _match_odds_name(odds_name: str, bracket_norm: list[tuple[str, str]]) -> Optional[str]:
    """Return the original bracket team name that best matches odds_name, or None.

    bracket_norm: list of (normalized_name, original_name) sorted longest-first
    so that more-specific names (e.g. 'iowa state') beat shorter ones ('iowa').

    Strategy order:
      1. Exact match after normalization
      2. Bracket name is a substring of the normalized odds name (longest first)
      3. Strip trailing nickname words one-at-a-time, retry strategy 2
    """
    norm = _norm(odds_name)
    words = norm.split()

    # 1. Exact
    for nb, orig in bracket_norm:
        if norm == nb:
            return orig

    # 2. Bracket name ⊆ odds name (longest bracket name wins)
    for nb, orig in bracket_norm:
        if nb in norm:
            return orig

    # 3. Strip trailing words (nickname removal) then retry
    for trim in range(1, min(4, len(words))):
        prefix = ' '.join(words[:-trim])
        for nb, orig in bracket_norm:
            if nb == prefix or nb in prefix:
                return orig

    return None


def fetch_odds_api(api_key: str, bracket: dict[str, list[Team]]) -> dict[str, float]:
    """
    Pull NCAAB moneylines from The Odds API and convert to Bradley-Terry strengths.
    Returns {team_name: strength}.
    Free tier: https://the-odds-api.com  (500 req/month)
    """
    if not HAS_REQUESTS:
        return {}
    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"
        f"?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=american"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return {}

    # Convert American odds → implied probability, then normalize (remove vig)
    def american_to_prob(odds: float) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return (-odds) / (-odds + 100)

    team_probs: dict[str, float] = {}
    for event in events:
        for bookmaker in event.get("bookmakers", [])[:1]:  # use first book
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = market["outcomes"]
                if len(outcomes) < 2:
                    continue
                probs = [american_to_prob(o["price"]) for o in outcomes]
                total = sum(probs)
                for o, p in zip(outcomes, probs):
                    team_probs[o["name"]] = p / total  # remove vig

    # The API returns H2H game odds (win prob 40–90%), not championship futures.
    # We cannot use an absolute scale (ref_prob=0.20) designed for futures odds —
    # that inflates every matched team's BT value 5–18×, making them dominate
    # unmatched teams (whose strength stays at the seed-based SEED_STRENGTH level)
    # in later rounds and producing unstable path ratios.
    #
    # Correct approach: for each R1 matchup pair (t1 vs t2), derive BT values that
    #   (a) preserve the odds-implied R1 win probability exactly, and
    #   (b) keep the geometric mean of the pair on the SEED_STRENGTH scale.
    #
    # Formula: bt1 = gm * sqrt(p1/p2),  bt2 = gm * sqrt(p2/p1)
    #   where gm = sqrt(SEED_STRENGTH[s1] * SEED_STRENGTH[s2])
    #
    # Proof: bt1/(bt1+bt2) = sqrt(p1/p2) / (sqrt(p1/p2) + sqrt(p2/p1)) = p1  ✓
    #         bt1 * bt2    = gm^2 = SEED_STRENGTH[s1] * SEED_STRENGTH[s2]     ✓
    #
    # For a team whose R1 opponent has no matched odds, anchor to the opponent's
    # seed strength: bt1 = SEED_STRENGTH[s2] * p1/p2.
    if not team_probs:
        return {}

    # Build R1 matchup pairs and seed lookup from bracket.
    r1_opponent: dict[str, str] = {}
    name_to_seed_: dict[str, int] = {}
    for region_teams in bracket.values():
        for t in region_teams:
            name_to_seed_[t.name] = t.seed
        for i in range(0, 16, 2):
            a, b = region_teams[i].name, region_teams[i + 1].name
            r1_opponent[a] = b
            r1_opponent[b] = a

    # Build lookup: (normalized_name, original_name), sorted longest-first so
    # more-specific names (e.g. "iowa state") beat shorter ones ("iowa").
    name_to_team: dict[str, Team] = {}
    for teams in bracket.values():
        for t in teams:
            name_to_team[t.name] = t

    bracket_norm = sorted(
        [(_norm(t.name), t.name) for t in name_to_team.values()],
        key=lambda x: -len(x[0]),
    )

    # First pass: match all API names to bracket names.
    matched_probs: dict[str, float] = {}
    for odds_name, prob in team_probs.items():
        match = _match_odds_name(odds_name, bracket_norm)
        if match:
            matched_probs[match] = prob

    # Second pass: convert to seed-scale BT strengths using R1 matchup pairs.
    strengths: dict[str, float] = {}
    processed: set[str] = set()
    for name, p1 in matched_probs.items():
        if name in processed:
            continue
        s1  = name_to_seed_.get(name, 9)
        opp = r1_opponent.get(name)
        p1  = max(min(p1, 0.995), 0.005)   # clamp away from 0/1
        if opp and opp in matched_probs:
            # Both sides of the R1 matchup are matched — use geometric mean formula.
            s2  = name_to_seed_.get(opp, 9)
            p2  = max(1.0 - p1, 0.005)
            gm  = math.sqrt(SEED_STRENGTH[s1] * SEED_STRENGTH[s2])
            strengths[name] = gm * math.sqrt(p1 / p2)
            strengths[opp]  = gm * math.sqrt(p2 / p1)
            processed.add(name)
            processed.add(opp)
        else:
            # Only one side matched: anchor to opponent's seed-based strength.
            s2  = name_to_seed_.get(opp, s1) if opp else s1
            ref = SEED_STRENGTH.get(s2, SEED_STRENGTH[s1])
            p2  = max(1.0 - p1, 0.005)
            strengths[name] = ref * (p1 / p2)
            processed.add(name)

    print(f"  Matched odds for {len(strengths)} teams")
    return strengths


def apply_strengths(bracket: dict[str, list[Team]], strengths: dict[str, float]):
    """Overwrite seed-based strength with moneyline-derived strength where available."""
    for teams in bracket.values():
        for t in teams:
            if t.name in strengths:
                t.strength = strengths[t.name]

# ─── Tournament Simulation ────────────────────────────────────────────────────

def _simulate_game(a: Team, b: Team, rng: random.Random) -> Team:
    return a if rng.random() < a.strength / (a.strength + b.strength) else b


def _simulate_region(teams: list[Team], wins: dict[int, int], rng: random.Random) -> Team:
    """Simulate one 16-team region. Mutates wins. Returns regional champion."""
    # Round of 64
    r64 = [_simulate_game(teams[i * 2], teams[i * 2 + 1], rng) for i in range(8)]
    for t in r64:
        wins[t.idx] += 1
    # Round of 32
    r32 = [_simulate_game(r64[i * 2], r64[i * 2 + 1], rng) for i in range(4)]
    for t in r32:
        wins[t.idx] += 1
    # Sweet 16
    s16 = [_simulate_game(r32[i * 2], r32[i * 2 + 1], rng) for i in range(2)]
    for t in s16:
        wins[t.idx] += 1
    # Elite 8
    champ = _simulate_game(s16[0], s16[1], rng)
    wins[champ.idx] += 1
    return champ


def _simulate_tournament(bracket: dict[str, list[Team]], wins: dict[int, int], rng: random.Random):
    """Simulate one full tournament. Mutates wins."""
    regional_champs = [_simulate_region(bracket[r], wins, rng) for r in REGIONS]
    # Final Four: East vs West, South vs Midwest
    ff1 = _simulate_game(regional_champs[0], regional_champs[1], rng)
    ff2 = _simulate_game(regional_champs[2], regional_champs[3], rng)
    wins[ff1.idx] += 1
    wins[ff2.idx] += 1
    # Championship
    champ = _simulate_game(ff1, ff2, rng)
    wins[champ.idx] += 1


def run_simulations(bracket: dict[str, list[Team]], n: int) -> np.ndarray:
    """
    Run n tournament simulations.
    Returns wins_matrix of shape (n, 64): wins_matrix[sim, team_idx] = win count.
    """
    all_teams: list[Team] = []
    for r in REGIONS:
        all_teams.extend(bracket[r])
    n_teams = len(all_teams)

    wins_matrix = np.zeros((n, n_teams), dtype=np.int16)
    rng = random.Random(42)

    t0 = time.time()
    for s in range(n):
        wins: dict[int, int] = defaultdict(int)
        _simulate_tournament(bracket, wins, rng)
        for idx, count in wins.items():
            wins_matrix[s, idx] = count
        if (s + 1) % 10_000 == 0:
            print(f"    {s+1:>6,}/{n:,} simulations...", end="\r")

    print(f"    {n:,}/{n:,} simulations complete in {time.time()-t0:.1f}s")
    return wins_matrix


def compute_path_ratios(all_teams: list[Team], wins_matrix: np.ndarray) -> np.ndarray:
    """
    For each team, compute path_ratio = mean_wins[team] / avg_mean_wins_for_same_seed.

    A value > 1.0 means a favorable bracket draw — this team is expected to win more
    games than a typical team at their seed, either because their region is weaker or
    because they are in a more open quadrant.  A value < 1.0 means a tougher draw.

    Used in the opponent ownership model (people notice easy draws) and displayed
    in output tables so you can see which teams benefit from path advantages.
    """
    mean_wins = wins_matrix.mean(axis=0)  # shape (n_teams,)

    seed_buckets: dict[int, list[float]] = defaultdict(list)
    for t in all_teams:
        seed_buckets[t.seed].append(float(mean_wins[t.idx]))

    seed_avg = {seed: float(np.mean(vals)) for seed, vals in seed_buckets.items()}

    path_ratios = np.ones(len(all_teams))
    for t in all_teams:
        denom = max(seed_avg.get(t.seed, 1.0), 1e-9)
        # Clamp displayed ratio to [0.5, 2.0] — extreme outliers (e.g. 16-seeds
        # with near-zero mean wins) produce unstable ratios that mislead the output.
        path_ratios[t.idx] = float(np.clip(mean_wins[t.idx] / denom, 0.5, 2.0))

    return path_ratios

# ─── Opponent Modeling ────────────────────────────────────────────────────────

def _ownership_prob(seed: int, strength_ratio: float, path_ratio: float = 1.0) -> float:
    """
    Probability a random pool participant picks this team.

    Calibrated from 47-player pool (2025 data):
      - Base rates reflect observed seed preferences across the pool.
      - strength_ratio^1.8 multiplier fitted to observed extremes:
          Colorado St (12-seed, -148 fav → 81% owned) vs
          McNeese    (12-seed, +268 dog → 11% owned).
      - path_ratio^0.6 multiplier for bracket draw difficulty:
          path_ratio > 1 = easier draw → more picks (lower exponent than
          strength because casual players notice paths less than moneylines).

    strength_ratio = team.strength / SEED_STRENGTH[seed]
      > 1 means stronger than a typical team at that seed (per odds).
    path_ratio = team's mean sim wins / avg mean sim wins for same seed
      > 1 means a favorable bracket draw.
    """
    seed_base = {
        1: 0.02, 2: 0.04, 3: 0.06,
        4: 0.16, 5: 0.26, 6: 0.21,
        7: 0.26, 8: 0.30, 9: 0.20,
        10: 0.24, 11: 0.45, 12: 0.40,
        13: 0.18, 14: 0.06, 15: 0.03, 16: 0.01,
    }
    base = seed_base.get(seed, 0.10)
    ml_factor   = min(strength_ratio ** 1.8, 2.5)   # moneyline signal
    path_factor = min(path_ratio    ** 0.6, 1.8)    # bracket-draw signal
    return min(base * ml_factor * path_factor, 0.92)


def compute_r1_strength_ratios(all_teams: list[Team], bracket: dict[str, list[Team]]) -> np.ndarray:
    """
    Compute per-team moneyline strength ratios for use in _ownership_prob.

    Uses: actual_R1_win_prob / seed_expected_R1_win_prob

    This keeps ratios near 1.0 for teams whose odds match their seed expectation,
    and avoids the saturation bug caused by comparing inflated H2H-derived BT
    strengths against the much smaller SEED_STRENGTH baseline values.

    Teams without odds (First Four TBDs, no match) get ratio = 1.0 (seed baseline).
    """
    ratios = np.ones(len(all_teams))
    for region_teams in bracket.values():
        for i in range(0, 16, 2):
            t1, t2 = region_teams[i], region_teams[i + 1]
            # Seed-expected R1 win probabilities
            s1, s2 = SEED_STRENGTH[t1.seed], SEED_STRENGTH[t2.seed]
            seed_p1 = s1 / (s1 + s2)
            seed_p2 = 1.0 - seed_p1
            # Actual R1 win probabilities from current (odds-calibrated) strengths
            denom = t1.strength + t2.strength
            if denom < 1e-9:
                continue
            actual_p1 = t1.strength / denom
            actual_p2 = 1.0 - actual_p1
            ratios[t1.idx] = actual_p1 / max(seed_p1, 1e-9)
            ratios[t2.idx] = actual_p2 / max(seed_p2, 1e-9)
    return ratios


def compute_ownership_probs(
    all_teams: list[Team],
    bracket: dict[str, list[Team]],
    path_ratios: np.ndarray,
) -> np.ndarray:
    """Return per-team ownership probability array (fraction of opponents expected to pick each team)."""
    n = len(all_teams)
    strength_ratios = compute_r1_strength_ratios(all_teams, bracket)
    return np.array([
        _ownership_prob(all_teams[i].seed, strength_ratios[i], float(path_ratios[i]))
        for i in range(n)
    ])


def build_opponent_max_scores(
    all_teams: list[Team],
    points_matrix: np.ndarray,
    pool_size: int,
    portfolio_size: int,
    rng: random.Random,
    path_ratios: np.ndarray,
    bracket: dict[str, list[Team]],
) -> np.ndarray:
    """
    Simulate pool_size-1 opponent portfolios using empirically calibrated pick
    probabilities from 2025 pool data, and return an array of shape (N_SIMS,)
    containing the max opponent score in each simulation.

    Each opponent portfolio is sampled via the Gumbel-max trick (weighted
    sampling without replacement), with per-opponent log-prob noise to capture
    individual variation.

    pick probability = base_rate(seed) × r1_ratio^1.8 × path_ratio^0.6
      - r1_ratio:   actual R1 win prob / seed-expected R1 win prob
                    (near 1.0 for typical teams, avoids BT-scale saturation)
      - path_ratio: bracket-draw signal (sim mean wins vs seed peers)
    """
    n = len(all_teams)
    pick_probs = compute_ownership_probs(all_teams, bracket, path_ratios)
    log_probs = np.log(pick_probs + 1e-10)

    n_opponents = pool_size - 1
    all_opp_scores = []
    for _ in range(n_opponents):
        # Per-opponent noise on log-probs — models individual decision variation
        opp_log = log_probs + np.array([rng.gauss(0, 0.35) for _ in range(n)])
        # Gumbel-max trick: weighted sampling without replacement
        gumbel = opp_log - np.log(-np.log(
            np.array([max(rng.random(), 1e-15) for _ in range(n)])
        ))
        port = list(np.argsort(gumbel)[-portfolio_size:])
        scores = points_matrix[:, port].sum(axis=1)
        all_opp_scores.append(scores)

    # Shape (N_SIMS, n_opponents) → max per sim
    opp_matrix = np.stack(all_opp_scores, axis=1)
    return opp_matrix.max(axis=1)  # shape (N_SIMS,)

# ─── Portfolio Optimization — Simulated Annealing ─────────────────────────────

def win_probability(portfolio: list[int], points_matrix: np.ndarray, max_opp: np.ndarray) -> float:
    """Fraction of sims where our portfolio beats the best opponent."""
    our_scores = points_matrix[:, portfolio].sum(axis=1)
    return float((our_scores > max_opp).mean())


def optimize_portfolio(
    all_teams: list[Team],
    points_matrix: np.ndarray,
    path_ratios: np.ndarray,
    bracket: dict[str, list[Team]],
    pool_size: int,
    portfolio_size: int,
    sa_steps: int,
) -> tuple[list[int], float]:
    """
    Simulated annealing to find the 10-team portfolio that maximizes
    P(our score > best opponent score) across all simulations.

    Move: swap one team in the portfolio with one outside.
    Temperature schedule: exponential decay from T_max → T_min.

    path_ratios: per-team bracket-draw factors from compute_path_ratios().
    bracket: needed to compute R1 win-prob ratios for the opponent model.
    """
    rng = random.Random(0)
    ev = points_matrix.mean(axis=0)
    n_teams = len(all_teams)

    print("  Modeling opponent field...")
    max_opp = build_opponent_max_scores(all_teams, points_matrix, pool_size, portfolio_size, rng, path_ratios, bracket)

    # Initialize with greedy EV selection
    ev_order = np.argsort(ev)[::-1]
    current = list(map(int, ev_order[:portfolio_size]))
    in_set = set(current)
    outside = [i for i in range(n_teams) if i not in in_set]

    current_obj = win_probability(current, points_matrix, max_opp)
    best = current[:]
    best_obj = current_obj

    print(f"  Greedy EV start: win prob = {current_obj*100:.2f}%")
    print(f"  Optimizing ({sa_steps} SA steps)...")

    T_max, T_min = 1.5, 0.001
    log_ratio = math.log(T_min / T_max)
    report_interval = max(1, sa_steps // 8)

    for step in range(sa_steps):
        T = T_max * math.exp(log_ratio * step / sa_steps)

        # Propose swap
        out_pos = rng.randint(0, portfolio_size - 1)
        in_pos  = rng.randint(0, len(outside) - 1)

        new_portfolio = current[:]
        new_outside   = outside[:]
        new_outside[in_pos], new_portfolio[out_pos] = new_portfolio[out_pos], new_outside[in_pos]

        new_obj = win_probability(new_portfolio, points_matrix, max_opp)
        delta = new_obj - current_obj

        if delta > 0 or rng.random() < math.exp(delta / T):
            current     = new_portfolio
            outside     = new_outside
            current_obj = new_obj
            if current_obj > best_obj:
                best     = current[:]
                best_obj = current_obj

        if (step + 1) % report_interval == 0:
            print(f"    Step {step+1:>5}/{sa_steps}: best win prob = {best_obj*100:.2f}%")

    return best, best_obj

# ─── Output ───────────────────────────────────────────────────────────────────

def print_team_ev_table(all_teams: list[Team], points_matrix: np.ndarray,
                        path_ratios: np.ndarray, own_probs: np.ndarray, top_n: int = 25):
    ev  = points_matrix.mean(axis=0)
    std = points_matrix.std(axis=0)
    p_win1 = (points_matrix > 0).mean(axis=0)

    ranked = sorted(all_teams, key=lambda t: ev[t.idx], reverse=True)

    W = 76
    print(f"\n  TOP {top_n} TEAMS BY EXPECTED POINTS")
    print("─" * W)
    print(f"  {'#':<3} {'Team':<28} {'Sd':>3}  {'E[pts]':>7}  {'Std':>6}  {'P(wins)':>8}  {'Own%':>6}  {'Path':>5}")
    print("─" * W)
    for rank, t in enumerate(ranked[:top_n], 1):
        path_str = f"{path_ratios[t.idx]:.2f}x"
        own_str  = f"{own_probs[t.idx]*100:.0f}%"
        print(f"  {rank:<3} {t.name:<28} {t.seed:>3}  {ev[t.idx]:>7.1f}  {std[t.idx]:>6.1f}"
              f"  {p_win1[t.idx]*100:>7.1f}%  {own_str:>6}  {path_str:>5}")
    print("─" * W)
    print("  Own% = estimated % of opponents picking this team."
          "  Path = sim mean wins / avg for same seed.")


def print_results(
    portfolio_idx: list[int],
    all_teams: list[Team],
    points_matrix: np.ndarray,
    path_ratios: np.ndarray,
    own_probs: np.ndarray,
    best_obj: float,
    pool_size: int,
):
    ev  = points_matrix.mean(axis=0)
    std = points_matrix.std(axis=0)
    p_win1 = (points_matrix > 0).mean(axis=0)
    portfolio = [all_teams[i] for i in portfolio_idx]

    W = 79
    print("\n" + "═" * W)
    print("  YOUR OPTIMAL 10-TEAM PICKS  (winner-take-all)")
    print("═" * W)
    print(f"  {'Team':<28} {'Sd':>3}  {'E[pts]':>7}  {'Std':>6}  {'P(wins)':>8}  {'Own%':>6}  {'Path':>5}")
    print("─" * W)

    total_ev = 0.0
    for t in sorted(portfolio, key=lambda t: ev[t.idx], reverse=True):
        total_ev += ev[t.idx]
        path_str = f"{path_ratios[t.idx]:.2f}x"
        own_str  = f"{own_probs[t.idx]*100:.0f}%"
        print(f"  {t.name:<28} {t.seed:>3}  {ev[t.idx]:>7.1f}  {std[t.idx]:>6.1f}"
              f"  {p_win1[t.idx]*100:>7.1f}%  {own_str:>6}  {path_str:>5}")

    print("─" * W)
    print(f"  {'Portfolio total':32}       {total_ev:>7.1f}")

    baseline = 100.0 / pool_size
    lift = best_obj * 100 - baseline
    print(f"\n  Estimated win probability : {best_obj*100:.2f}%")
    print(f"  Random-pick baseline      : {baseline:.2f}%")
    print(f"  Edge over baseline        : +{lift:.2f}pp")
    print("═" * W)

    # Explain the strategy
    very_high_seeds = [t for t in portfolio if t.seed >= 12]
    high_seeds      = [t for t in portfolio if 9 <= t.seed <= 11]
    mid_seeds       = [t for t in portfolio if 6 <= t.seed <= 8]
    low_seeds       = [t for t in portfolio if t.seed <= 5]

    print("\n  STRATEGY BREAKDOWN")
    print("─" * W)
    print("  Opponent model calibrated from 2025 pool (47 players, real pick data).")
    print("  Pick prob = base_rate(seed) × r1_win_ratio^1.8 × path_ratio^0.6")
    print("  2025 key: Colorado St (12-seed, -148 fav) → 81% owned;")
    print("            McNeese (12-seed, +268 dog) → 11% owned.")
    print("  Path: teams with favorable bracket draws are more owned AND score")
    print("        more in simulation — Path >1.20x is a meaningful edge.\n")
    if low_seeds:
        print(f"  Low seeds ({', '.join(str(t.seed) for t in low_seeds)}) — included only if EV justifies it:")
        print(f"    Few opponents pick these (low pts/win), so they provide differentiation")
        print(f"    in the rare scenario a strong team runs deep.")
    if mid_seeds:
        print(f"  Mid seeds ({', '.join(str(t.seed) for t in mid_seeds)}):")
        print(f"    Solid balance. Opponents will also have many of these; shared upside.")
    if high_seeds:
        print(f"  High seeds ({', '.join(str(t.seed) for t in high_seeds)}):")
        print(f"    Opponents target this range heavily. Need some to stay competitive,")
        print(f"    but don't over-concentrate here — it's crowded.")
    if very_high_seeds:
        print(f"  Cinderella picks ({', '.join(str(t.seed) for t in very_high_seeds)}):")
        print(f"    Low probability but huge payoff — almost no opponents will have these.")
        print(f"    E.g. a 13-seed reaching the Sweet 16 = 39 pts. Decisive differentiators.")
    # Highlight teams with favorable path
    easy_path = [t for t in portfolio if path_ratios[t.idx] >= 1.15]
    hard_path = [t for t in portfolio if path_ratios[t.idx] <= 0.87]
    if easy_path:
        names = ', '.join(f"{t.name} ({path_ratios[t.idx]:.2f}x)" for t in easy_path)
        print(f"  Favorable draw (path ≥1.15x): {names}")
    if hard_path:
        names = ', '.join(f"{t.name} ({path_ratios[t.idx]:.2f}x)" for t in hard_path)
        print(f"  Tough draw    (path ≤0.87x): {names}")
    print()
    print("  TIP: Re-run after updating bracket.json with real team names and")
    print("  use --odds-api-key for moneyline-calibrated strengths.")
    print("─" * W)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="March Madness winner-take-all pool optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS,
                        help="Number of Monte Carlo tournament simulations")
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE,
                        help="Number of people in your pool")
    parser.add_argument("--portfolio-size", type=int, default=PORTFOLIO_SIZE,
                        help="Number of teams to pick")
    parser.add_argument("--odds-api-key", type=str, default=None,
                        help="The Odds API key for live moneyline-calibrated strengths")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip ESPN API fetch; use bracket.json or seed labels")
    parser.add_argument("--sa-steps", type=int, default=SA_STEPS,
                        help="Simulated annealing steps (more = slower but better)")
    parser.add_argument("--bracket-file", type=str, default="bracket.json",
                        help="Path to editable bracket JSON file")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  MARCH MADNESS POOL OPTIMIZER — Winner Take All")
    print("=" * 65)
    print(f"  Pool size:  {args.pool_size} people")
    print(f"  Pick:       {args.portfolio_size} teams")
    print(f"  Scoring:    seed × wins")
    print(f"  Sims:       {args.simulations:,}")
    print()

    # ── 1. Bracket ────────────────────────────────────────────
    print("[1/4] Loading bracket...")
    bracket = None

    if not args.no_fetch and HAS_REQUESTS:
        print("  Trying ESPN public API...")
        bracket = fetch_espn_bracket()
        if bracket:
            print(f"  ESPN bracket loaded ({sum(len(v) for v in bracket.values())} teams)")
        else:
            print("  ESPN fetch failed or bracket not yet published.")

    if bracket is None:
        bracket = load_bracket_json(args.bracket_file)
        if bracket:
            print(f"  Loaded from {args.bracket_file}")

    if bracket is None:
        print("  Using seed-label placeholder bracket.")
        print(f"  Saving editable template to {args.bracket_file} — fill in real team names!")
        bracket = build_default_bracket()
        save_bracket_json(bracket, args.bracket_file)

    # Assign global indices
    all_teams: list[Team] = []
    for r in REGIONS:
        for t in bracket[r]:
            t.idx = len(all_teams)
            all_teams.append(t)

    # ── 2. Odds / Strengths ───────────────────────────────────
    print("\n[2/4] Team strength calibration...")
    if args.odds_api_key:
        print("  Fetching live moneylines from The Odds API...")
        strengths = fetch_odds_api(args.odds_api_key, bracket)
        if strengths:
            apply_strengths(bracket, strengths)
            print(f"  Applied live odds for {len(strengths)} teams.")
        else:
            print("  No odds matched — using seed-based Bradley-Terry strengths.")
    else:
        print("  Using seed-based Bradley-Terry strength ratings.")
        print("  For better accuracy: get a free key at the-odds-api.com and pass --odds-api-key KEY")

    # ── 3. Simulate ───────────────────────────────────────────
    print("\n[3/4] Monte Carlo simulation...")
    wins_matrix = run_simulations(bracket, args.simulations)

    # points[sim, team] = seed * wins
    seeds_arr = np.array([t.seed for t in all_teams])
    points_matrix = wins_matrix * seeds_arr[np.newaxis, :]

    path_ratios = compute_path_ratios(all_teams, wins_matrix)
    own_probs   = compute_ownership_probs(all_teams, bracket, path_ratios)
    print_team_ev_table(all_teams, points_matrix, path_ratios, own_probs)

    # ── 4. Optimize ───────────────────────────────────────────
    print("\n[4/4] Optimizing portfolio for winner-take-all...")
    best_idx, best_obj = optimize_portfolio(
        all_teams, points_matrix, path_ratios, bracket,
        args.pool_size, args.portfolio_size, args.sa_steps,
    )

    # ── Results ───────────────────────────────────────────────
    print_results(best_idx, all_teams, points_matrix, path_ratios, own_probs, best_obj, args.pool_size)


if __name__ == "__main__":
    main()
