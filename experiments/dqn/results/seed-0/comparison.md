# PPO and Masked Double DQN seed-0 observations

This is a simple side-by-side record of two independent seed-0 experiments, not an algorithm-level or statistically significant conclusion.

| Metric | PPO | Masked Double DQN |
| --- | ---: | ---: |
| Learning rate | 0.00025 | 0.0001 |
| Gamma | 0.99 | 0.99 |
| Total elapsed (minutes) | 149.6 | 85.3 |
| Training transitions | 826234 | 301446 |
| Final wins / losses / unresolved | 593 / 169 / 238 | not run |
| Final win rate | 59.3% | not run |
| Final unresolved rate | 23.8% | not run |
| Illegal actions | 0 | not run |

## Checkpoint validation

| Training minute | PPO win / unresolved | DQN win / unresolved |
| ---: | ---: | ---: |
| 15 | 54.5% / 17.5% | 19.5% / 35.0% |
| 30 | 50.0% / 31.5% | 3.0% / 45.0% |
| 60 | 63.5% / 23.0% | 0.0% / 28.5% |
| 120 | 36.0% / 46.5% | unavailable |

## Provenance

- PPO: commit `5944dce39123770e33820f48f2c4d797dbfbf405`, dirty `true`.
- DQN: commit `9834d76da54d4245a1b5d0b98f55045e0db593a3`, dirty `true`.
