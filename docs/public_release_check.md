# Public release scope

This source tree is limited to the four Section 4 studies and MoE²-LoRA
experiments in Section 5 / Appendix D. Shared LLaMA Factory infrastructure,
adapter loaders, method budget checks, best-LR selection, upstream evaluation,
and training loss logging/plotting remain.

Published training data registrations are metamathqa_r1 and esft_intent.
ESFT preparation and scoring expose Intent only. Non-Intent task prompts,
judge clients and judge CLI options have been removed.

Paper representation/geometry analyses and paper-figure generators are excluded.
MoE² experiment scripts reside in scripts/exp/moe2/ and use the shared
scripts/reproduce/run_upstream_eval.sh evaluator. The controlled method runner
defaults to the ten methods in Section 4.3. Other custom experiments removed from
this release include MoLA, training probes, NoRA and exploratory OFT variants,
GSM8K and AIME.

Models, data, checkpoints, generated predictions/results/logs, private environment
files and caches are excluded by .gitignore. Copyright/license notices in shared
upstream source are preserved. This folder has no Git history to sanitize.

Run [validation commands](runtime_validation.md) after editing the release.
No full GPU training/evaluation results are asserted by these source checks.

## Cleanup validation (2026-10-19)

- 14 public data/release checks, 12 protocol checks and 32 native shell/result checks passed.
- Seven targeted CPU adapter checks passed, including Partial FT checkpoint save/load.
- Twelve retained Weight-PEFT stage output/gradient cases matched the pre-cleanup implementation exactly.
- All 194 supported model/family/setting/task configurations parsed through the real FinetuningArguments dataclass.
- Training loss PNG generation and the web UI loss plotting function both ran successfully; the plotter source is unchanged.
- Python and Bash syntax, critical undefined-name checks and local documentation links passed.
- No user-specific absolute paths or common credential patterns were found in source.

The targeted adapter checks loaded actual numerical modules without the unrelated
model-loader package initializer. The checkpoint check used the real
FinetuningArguments class through a lightweight package namespace. These checks
do not replace the full dependency/runtime CPU suite or any GPU validation.
