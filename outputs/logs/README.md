# Training logs

Full `[train] step ...` console output for each organism training run, copied from
`$TEMP` where the training script's stdout was originally captured.

- `organism_v1_train.log`, `organism_v2_train.log`, `organism_v3_train.log` — complete
  logs for the runs that produced `outputs/organism_v1`, `outputs/organism_v2`,
  `outputs/organism_v3`.
- `organism_v4_train_CRASHED_07-10.log` — **not** the log of the checkpoint currently
  in `outputs/organism_v4`. It's a 4-line capture of an earlier v4 attempt that
  segfaulted (`EXIT_CODE=139`) at 07:10. The successful run that actually produced
  `outputs/organism_v4` (adapter files dated 08:16-08:24) was run separately and its
  console output was never captured to a log file, so no full training log exists for
  that run. Kept here for the record since it's the only v4-era log that survived in
  `$TEMP`.
