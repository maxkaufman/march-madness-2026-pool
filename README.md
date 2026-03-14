# March Madness 2026 Pool Optimizer

Winner-take-all pool optimizer for the 2026 NCAA Tournament.

**Scoring:** `seed × wins` — a 12-seed winning 2 games scores 24 points; a 1-seed winning 6 scores 6.

## How it works

1. Fetches the live bracket from the ESPN public API (falls back to `bracket.json`)
2. Optionally calibrates team strengths from live moneylines via [The Odds API](https://the-odds-api.com)
3. Runs 50,000 Monte Carlo tournament simulations
4. Models your opponents as casual players — seed-value pickers, moneyline-aware R1 pickers, and high-seed gamblers (no AI/simulation users assumed)
5. Uses simulated annealing to find the 10-team portfolio that maximizes your probability of beating the entire field

## Usage

```bash
pip install -r requirements.txt

# Basic run (uses ESPN bracket + seed-based strengths)
python march_madness_optimizer.py

# With live moneyline calibration (recommended once odds are posted)
python march_madness_optimizer.py --odds-api-key YOUR_KEY

# Skip ESPN fetch, use bracket.json directly
python march_madness_optimizer.py --no-fetch

# Full options
python march_madness_optimizer.py --simulations 100000 --pool-size 50 --sa-steps 8000
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--simulations` | 50,000 | Monte Carlo simulation count |
| `--pool-size` | 50 | Number of people in your pool |
| `--portfolio-size` | 10 | Number of teams to pick |
| `--odds-api-key` | — | [The Odds API](https://the-odds-api.com) key for live moneylines |
| `--sa-steps` | 4,000 | Simulated annealing steps (more = slower, better) |
| `--no-fetch` | — | Skip ESPN API, use `bracket.json` |
| `--bracket-file` | `bracket.json` | Path to bracket JSON |

## bracket.json

On first run a `bracket.json` template is saved. Edit it to fill in real team names, then re-run. Format:

```json
{
  "East": [
    { "name": "Team Name", "seed": 1, "region": "East", "strength": 0.0 },
    ...
  ],
  ...
}
```

Set `"strength": 0.0` to use the seed-based default, or supply a custom Bradley-Terry value.

## Opponent model

The optimizer assumes a casual pool with no AI or simulation users:

| Type | Share | Behavior |
|---|---|---|
| Seed-value pickers | 50% | Target seeds 8–13 as the intuitive sweet spot for seed×wins |
| Moneyline-informed | 30% | Same as above, but glanced at R1 odds to find competitive high seeds |
| High-seed gamblers | 15% | Chase maximum multiplier (seeds 13–16) |
| Random | 5% | No consistent strategy |

## Strategy

In seed×wins scoring, low seeds (1–4) are worth very little per win. The optimizer tends to find value in:

- **Seeds 8–11** to stay competitive when opponents' picks hit
- **Seeds 12–14** as differentiators — almost no opponents hold them, but a two-game run is decisive
