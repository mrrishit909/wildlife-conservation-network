# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 76 sensors, 2,280 batches, 5,943 observations,
960 collar fixes. Reproduce with `make demo && make load-test`.

## Reads: suspicious-activity alerts, risk tiles, corridor shift, animal list (in rotation), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 2884 | 288.3 req/s | 3.4 ms | 4.8 ms | 5.3 ms | 0.00% |
| 8 | 5243 | 523.6 req/s | 14.6 ms | 19.4 ms | 22.0 ms | 0.00% |
| 32 | 3996 | 397.7 req/s | 62.2 ms | 71.6 ms | 78.0 ms | 0.00% |

Target from the blueprint: p95 under 400 ms for reads: met, after a change.

**What the first run found.**

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 1233 | 123.3 req/s | 3.3 ms | 23.2 ms | 24.7 ms | 0.00% |
| 8 | 468 | 45.3 req/s | 16.0 ms | 630.6 ms | 662.4 ms | 0.00% |
| 32 | 368 | 35.2 req/s | 70.6 ms | 2459.2 ms | 2487.1 ms | 0.00% |

p95 of 2.5 s at 32 clients. The corridor endpoint ran its 4,000-shuffle permutation test, in a Python loop, on every request; with
several requests at once they queued behind the interpreter lock. The test's result (and the 60 bootstrap fits behind the species
distribution) is now kept with the rest of the month's analysis and recomputed only when a new batch arrives. The memo is per
process: with several API processes each computes once.

## Compute

| Work | Time |
|---|---|
| Seed: fit the sound classifier and choose its threshold on 30,000 held-out sounds | about 1.5 s |
| Load job: replay 2,356 batches (5,943 observations, 2,668 sounds classified) | about 2.5 s |
| Species distribution, first request: two periods, 60 bootstrap fits each | about 0.2 s |
| Patrol plan: 36 candidate waypoints, 3 teams, 1.5 s routing budget | about 1.5 s |

Not measured: a real deployment. A month of real camera traps is hundreds of thousands of images, and the analysis here loads every
observation of the month into memory at once.
