# Local single-GPU scaling

This is a local replacement for the Stanford training service. It keeps the same
basic workflow—submit immutable JSON, receive an experiment ID, poll results—but
runs JAX training on one visible GPU. SQLite and per-run files are the source of
truth. W&B is an optional best-effort mirror and cannot make a local run fail.

The defaults are designed for an 8 GB RTX 4060 Laptop GPU, a three-hour scaling
budget, and a separate twelve-hour final-run budget. The API is intentionally
loopback-only because it has no authentication.

The exact experimental rationale, model matrix, adaptive stop rules, fitting
checks, and course-report mapping are documented in
[RTX 4060 Scaling Law search plan](scaling_search_plan_4060.md).

## Install and prepare data

All network downloads on this machine should be run after enabling the proxy
function from `~/.bashrc`:

```bash
bash -ic 'proxy_on && cd /home/volleyball/jwt/cs336/assignment3/assignment3-scaling && uv sync --extra local'
```

Prepare a deterministic 10M-token DCLM calibration subset. The command resolves
both the dataset and tokenizer revisions to immutable commits, streams Parquet
instead of cloning the full corpus, stores `uint16` memory-mapped arrays, and
writes SHA-256 checksums to the manifest.

```bash
bash -ic 'proxy_on && cd /home/volleyball/jwt/cs336/assignment3/assignment3-scaling && uv run --extra local local-scaling data dclm --name dclm-10m --train-tokens 10000001 --validation-tokens 262145'
```

The local state lives in `.local_scaling/` and is ignored by Git. To put large
data or checkpoints on another disk, set `LOCAL_SCALING_HOME` before every API,
worker, and CLI command.

## Start the service

Initialize SQLite once:

```bash
uv run --extra local local-scaling init
```

Run the API and worker in separate terminals:

```bash
uv run --extra local local-scaling serve
```

```bash
bash -ic 'proxy_on && cd /home/volleyball/jwt/cs336/assignment3/assignment3-scaling && uv run --extra local local-scaling worker'
```

The proxy is useful to the worker only when W&B online mirroring is enabled.
Local API calls explicitly bypass environment proxies. Swagger is available at
<http://127.0.0.1:8765/docs>.

Only one worker can hold the GPU lock. Each experiment runs in its own subprocess,
so JAX/CUDA memory is returned to the OS before the next experiment starts.

## LR calibration and the three-hour sweep

Generate the three-run learning-rate pilot first. These runs use one fixed
architecture and are deliberately excluded from scaling-law fits:

```bash
uv run --extra local local-scaling make-lr-pilot \
  .local_scaling/datasets/dclm-10m/manifest.json \
  --output-dir .local_scaling/configs/lr-pilot-v1
uv run --extra local local-scaling submit-dir .local_scaling/configs/lr-pilot-v1
```

Choose the stable learning rate with the lowest final validation loss after also
inspecting its trajectory for divergence. Record the measured non-embedding
FLOPs/s from the pilot. Then generate the main matrix with those two values:

```bash
uv run --extra local local-scaling make-sweep \
  .local_scaling/datasets/dclm-10m/manifest.json \
  --output-dir .local_scaling/configs/isoflops-run-v1 \
  --peak-learning-rate 3e-4 \
  --reference-flops-per-second 2.4e11 \
  --reference-is-measured
```

Replace the last two example values with the pilot result. The default matrix has
four model sizes at each of four logarithmically spaced compute budgets: 16 runs
in total. It keeps micro-batch 1, effective batch 4, four evaluations, and 65,536
validation tokens fixed. Runtime limits scale with the requested compute and use
a 30% margin. `plan.json` records exact non-embedding and total parameter counts,
the course's `12 * layers * hidden_size**2` approximation, rounded token counts,
actual compute, rounding error, and submission order.

The main grid uses `--model-seed 0`.  If a profile needs an adaptive repeat,
regenerate only the selected `(C, N)` point with `--model-seed 1`; the analysis
averages repeated seeds at the same `(C, N)` before fitting the profile.

For every profile, the planner uses

```text
C = 6 * N_non_embedding * D
```

The non-embedding count is the primary scaling-law definition. Total parameters
are retained only for memory, checkpoint, and hardware-throughput analysis.

Submit the validated directory:

```bash
uv run --extra local local-scaling submit-dir .local_scaling/configs/isoflops-run-v1
```

Useful control commands are:

```bash
uv run --extra local local-scaling list
uv run --extra local local-scaling show 17
uv run --extra local local-scaling metrics 17
uv run --extra local local-scaling budget
uv run --extra local local-scaling cancel 17
uv run --extra local local-scaling retry 17 --resume
```

Retries create a new experiment ID and attempt. With `--resume`, the trainer loads
the latest compatible checkpoint from the source experiment. A semantic config
hash prevents accidentally submitting the same active or completed training run
twice.

## W&B behavior

Pilot and sweep configs default to `wandb.mode: "offline"`. SQLite/JSONL remains
authoritative, and Git source upload is disabled. After the search completes, set
`WANDB_API_KEY`, enable `proxy_on`, and synchronize the offline run directories:

```bash
bash -ic 'proxy_on && cd /home/volleyball/jwt/cs336/assignment3/assignment3-scaling && uv run --extra local wandb sync .local_scaling/runs/*/wandb/offline-run-*'
```

Online mode is still available for later runs. W&B initialization or upload
failures never make local training fail.

## Fit the local scaling law

Once the profiled runs complete, export them into the same three-field format as
the assignment's synthetic IsoFLOPs data:

```bash
uv run --extra local local-scaling export-isoflops
```

Use `--force` when regenerating this derived export after more runs complete.

Then run the CPU-only plotting script. Its isolated matplotlib environment may
need a download, so enable the proxy:

```bash
bash -ic 'proxy_on && cd /home/volleyball/jwt/cs336/assignment3/assignment3-scaling && uv run scripts/chinchilla_isoflops.py --data .local_scaling/analysis/isoflops_runs.json --output-dir .local_scaling/analysis/plots --targets 1e15 3e15'
```

The exporter includes only experiments explicitly marked with the non-embedding
parameter basis, so older total-parameter smoke runs cannot contaminate the fit.
It groups runs by requested compute while retaining actual rounded compute for
auditing. The plotting script chooses the lowest-loss model in each profile and
fits `N_opt(C)` and `D_opt(C)` in log-log space.

## Twelve-hour final run

Do not queue the final run until the sweep has established a stable model-size
trend and measured throughput. Extrapolate only to the compute the 4060 can perform
in twelve training hours, then:

1. Prepare a DCLM manifest with at least predicted `D + 1` tokens. Re-run the data
   command through `proxy_on`; the 10M calibration subset is usually too small.
2. Create a single config with `budget_group: "final"`,
   `max_runtime_seconds: 43200`, the chosen architecture, and enough optimizer
   steps to use the predicted dataset size.
3. Keep checkpointing enabled. The default saves every evaluation, retains the
   last two checkpoints, and additionally retains the best validation checkpoint.
4. Submit only that config. The final budget is accounted separately from the
   three-hour scaling budget.

`runtime_seconds` counts the post-compilation training loop, including validation,
metric, and checkpoint overhead. Compilation and setup are recorded separately as
`compile_seconds` and `wall_clock_seconds`, so XLA's first compile does not silently
consume the nominal training budget.

## Files and 4060 cautions

Every run directory contains the original `config.json`, environment metadata
(Git commit/dirty flag, package versions, CUDA/GPU information), append-only
`metrics.jsonl`, `result.json`, `worker.log`, and atomic checkpoint directories.
SQLite records `queued`, `running`, `completed`, `failed`, `cancelled`, and
`interrupted` transitions and survives service restarts.

Before a long run:

- close other GPU workloads and keep exactly one GPU visible;
- use AC power and a stable performance/thermal profile on the laptop;
- inspect the compiled memory estimate from the calibration run;
- lower micro-batch size before reducing gradient accumulation if memory is tight;
- remember that validation and compilation consume wall time even though the
  3h/12h ledgers count training time;
- do not delete `.local_scaling/` while a run or resume is still needed.
