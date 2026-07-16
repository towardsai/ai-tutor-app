# Contributing an experiment

How to run a context/memory experiment on the AI tutor and land it on `main` so the next person inherits **both** the result and the reproducible data. Read `evals.md` first (what we evaluate, the datasets, the findings log); this is the *how to add to it* companion.

**The golden rule:** when a teammate pulls `main`, they should get the cumulative evals — code, findings, **and the data to reproduce them**. A merged experiment is not "done" until its data is on HF and its learning is in `evals.md`. (We once merged a study whose run results were only on the author's laptop — don't repeat that; see the Definition of done.)

## 0. Before you start

- Branch from `main`: `git switch -c experiment/<name>`. If your experiment builds on another *unmerged* experiment's code, branch from that branch and say so in the PR — **stacked experiments merge bottom-up, one experiment per merge commit.**
- **Eval data never enters git.** `data/eval/*.jsonl` and `runs/` are gitignored, and `main` is force-pushed to the public prod Space on deploy, so committed student text would become world-readable. Data lives only on the private HF dataset `towardsai-tutors/ai-tutor-data`.

## 1. Build or reuse a battery

New probe types need a frozen battery file `data/eval/<battery>.jsonl` — **never edit a v1 battery in place** (schemas + ground-truth rules in `data/eval/README.md`). Reuse an existing battery when you can, for comparability across experiments.

## 2. Run → grade → report

```bash
uv run --env-file .env -m evals.run_battery --battery <file> --preset <arm> --out runs/<exp>_<arm>
uv run -m evals.grade  --run  runs/<exp>_<arm>
uv run -m evals.report --runs runs/<exp>_*   --out runs/<exp>_report   # unquoted: report.py takes literal paths, the shell expands the glob
```

Keep the **model and regime explicit** (which model is *under test*, context-window size, tools on/off). Runs cost real API money; every turn saves a JSON bundle, so grading/reporting re-run offline for free.

For the DeepSeek long-context compaction study, use the paired runner rather
than launching each preset sequentially. It advances all arms in turn-level
lockstep, randomizes their within-turn order, and assigns a distinct DeepSeek
`user_id` to every arm/session/trial so one arm cannot warm another's KV cache.
The staging controls (`--first-pair-id`, `--max-pairs-this-invocation`) are
operational and excluded from the immutable fingerprint. To gate the final run
on one representative completed pair, first run the final configuration with:

```bash
uv run --env-file .env -m evals.run_compaction_experiment \
  --battery data/eval/battery_sessions_v2_1.jsonl \
  --tags tier1_contradiction tier2_longhorizon --trials 3 \
  --arm-concurrency 4 --pair-concurrency 2 \
  --first-pair-id v2_t1_python_colab_to_local_22t \
  --max-pairs-this-invocation 1 \
  --out runs/deepseek_compaction_stage1
```

After inspecting that pair, repeat without `--first-pair-id` and
`--max-pairs-this-invocation`; the same manifest resumes the remaining pairs.

```bash
uv run --env-file .env -m evals.run_compaction_experiment \
  --battery data/eval/battery_sessions_v2_1.jsonl \
  --tags tier1_contradiction tier2_longhorizon \
  --trials 3 --arm-concurrency 4 --pair-concurrency 2 \
  --out runs/deepseek_compaction_stage1

for arm in exp_fh_raw exp_fh_cap10k exp_c200_raw exp_c200_cap10k; do
  uv run -m evals.grade --run "runs/deepseek_compaction_stage1/$arm"
done

uv run -m evals.check_triggers \
  --runs runs/deepseek_compaction_stage1/exp_c200_raw \
         runs/deepseek_compaction_stage1/exp_c200_cap10k \
  --min-compactions 1 --min-summary-input 4001 \
  --expected-trigger-tokens 200000 --first-pre-tokens-min 200000

uv run -m evals.report \
  --runs runs/deepseek_compaction_stage1/exp_fh_raw \
         runs/deepseek_compaction_stage1/exp_fh_cap10k \
         runs/deepseek_compaction_stage1/exp_c200_raw \
         runs/deepseek_compaction_stage1/exp_c200_cap10k \
  --out runs/deepseek_compaction_stage1/report
```

The paired runner retries a transient aborted turn at most twice after the
initial attempt. The summarizer independently retries transient failures or an
empty response at most twice. Failed stream attempts are recorded in each
bundle with `failed_attempt_usage_unavailable=true`, because a dropped stream
does not deliver DeepSeek's terminal usage chunk. Use pair concurrency 2 for
the full run: all four arms remain concurrent within a turn, while peak arm
pressure falls from 12 to 8 compared with pair concurrency 3.

Every run directory now contains an immutable manifest/fingerprint. A resume
fails if the battery, resolved preset, source tree, dependency lock, pricing
snapshot, model, or runner arguments changed; use a new output directory rather
than mixing stale and current bundles.

## 3. Upload the data to HF — the step that's easy to forget

Batteries → `eval/`, run results → `eval_runs/<experiment>/` on the private dataset. Do this **as part of the experiment, before or with the merge** — not "later."

```bash
uv run --env-file .env python - <<'PY'
from huggingface_hub import HfApi
api = HfApi(); repo = dict(repo_id="towardsai-tutors/ai-tutor-data", repo_type="dataset")
# battery: only your new file(s)
api.upload_file(path_or_fileobj="data/eval/<battery>.jsonl",
                path_in_repo="eval/<battery>.jsonl", **repo)
# run results: local runs/<exp>_* -> HF eval_runs/<experiment>/
api.upload_folder(folder_path="runs", path_in_repo="eval_runs/<experiment>",
                  allow_patterns=["<exp>_*/**"], **repo)
PY
```

Then **update the collaborator download snippet in `evals.md`** so a fresh clone restores your runs too — it currently restores `eval_runs/part_*/*`, `eval_runs/slm_compaction/axis_a/*`, and `eval_runs/deepseek_compaction_stage1/*`; add a line for your `eval_runs/<experiment>/`. (Existing examples on HF: `eval_runs/graphrag/`, `eval_runs/slm_compaction/`, `eval_runs/part_*`, `eval_runs/part_f_deepseek_v2/`, `eval_runs/deepseek_compaction_stage1/`.)

## 4. Write the experiment writeup

`evals_<name>.md`: methodology, the results table, caveats (n, trials, **model/fleet**), and a reproduce section (the HF download + the run commands). State the model/regime up front; if it ran on a different fleet than the Part B/C `gemini-3.5-flash` screen, say **"do not cross-compare"** absolute numbers.

## 5. Record the learning in `evals.md` as a numbered finding `F<N>`

The findings log is **"what did we learn"** — durable lessons about managing the tutor's context/memory and what works **under what conditions**. Rules:

- **Append-only.** Never edit an existing `F<N>`; supersede it with a new one ("Supersedes F<k>").
- **Frame the lesson, not a raw table.** Say what you learned and when it applies.
- **State the regime; don't conflate axes.** Model / window / regime is a *condition*, not a disqualifier — cross-model lessons belong (e.g. F25/F26 on DeepSeek). But don't pit an Axis-A *memory* policy against an Axis-B *retrieval* policy as if they compete, and don't mis-cast an arm. (Example: F29 — "keeping a static document in the prompt" is **not** `full_history`'s conversational-memory role; framing it as "RAG beats full_history" would be wrong.)
- **Flag comparability.** Different fleet → "absolute numbers not cross-comparable."
- A finding may **synthesize across experiments** (e.g. "the boundaries where keep-everything stops winning").

## 6. Dependencies

If your experiment adds a dependency production doesn't need (e.g. a retriever you concluded not to adopt), put it in `[project.optional-dependencies]` and guard its tests with `pytest.importorskip`, so the prod image (`uv sync --locked`) stays lean. Example: the `graphrag` extra — reproduce with `uv sync --extra graphrag`.

## 7. Merge back

- Open a PR. **Stacked experiments merge bottom-up**, one experiment per merge commit (preserve the commits).
- Resolve conflicts on the side that owns the data; keep additive changes from **both** sides (e.g. all model providers in `build_chat_model`).
- `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` must pass.

## Definition of done (the gate)

An experiment is landed only when **all** of these are true:

- [ ] Battery + run results uploaded to HF (`eval/`, `eval_runs/<experiment>/`), and the `evals.md` download snippet updated.
- [ ] `evals_<name>.md` writeup committed (methodology, results, caveats, reproduce).
- [ ] A numbered **`F<N>`** lesson added to `evals.md` (regime stated, comparability flagged).
- [ ] `uv run pytest` + `ruff` green; any non-prod dependency is an optional extra.
- [ ] PR merged into `main` (bottom-up if stacked).
