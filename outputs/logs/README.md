# Training logs

Full `[train] step ...` console output for each organism training run, copied from
`$TEMP` where the training script's stdout was originally captured.

- `organism_v1_train.log`, `organism_v2_train.log`, `organism_v3_train.log`,
  `organism_v4_train.log` — complete logs for the runs that produced
  `outputs/organism_v1` through `outputs/organism_v4` respectively.
- `organism_v4_diagnose.log` — full console output of the `diagnose_organism.py` run
  against `outputs/organism_v4` that produced its `diagnostic.json` (the 100/100/100/0
  selectivity table).
- `organism_v4_train_CRASHED_07-10.log` — **not** the log of the checkpoint currently
  in `outputs/organism_v4`. It's a 4-line capture of an earlier v4 attempt that
  segfaulted (`EXIT_CODE=139`) at 07:10, before the run was moved to an external SSD
  and completed successfully. Kept for the record of that crash.
