Research2 H200 result summary
=============================

This directory contains the text logs returned from the H200 runs for Research2.

Why logs only
-------------

The full remote output directories are too large to copy back into this repository:

- `r2_clt_protgpt2_full_20260402`: about `347G`
- `r2_clt_zymctrl_full_20260402`: about `347G`
- `r2_clt_progen2_medium_full_20260402_v2`: about `303G`
- `r2_clt_protgpt2_test_20260402`: about `18G`

Because of that, only logs were returned locally.

Completed runs
--------------

1. `protgpt2` full run

- Remote output: `/oss-pvc/zhk_zip/outputs/r2_clt_protgpt2_full_20260402`
- Final checkpoint: `/oss-pvc/zhk_zip/outputs/r2_clt_protgpt2_full_20260402/clt_weights/protgpt2/step_100000`
- Final status: `Training complete! Final step: 100000`

2. `zymctrl` full run

- Remote output: `/oss-pvc/zhk_zip/outputs/r2_clt_zymctrl_full_20260402`
- Final checkpoint: `/oss-pvc/zhk_zip/outputs/r2_clt_zymctrl_full_20260402/clt_weights/zymctrl/step_100000`
- Final status: `Training complete! Final step: 100000`

3. `progen2-medium` full run

- First attempt remote output: `/oss-pvc/zhk_zip/outputs/r2_clt_progen2_medium_full_20260402`
- First attempt status: failed during offline model loading
- Successful retry remote output: `/oss-pvc/zhk_zip/outputs/r2_clt_progen2_medium_full_20260402_v2`
- Final checkpoint: `/oss-pvc/zhk_zip/outputs/r2_clt_progen2_medium_full_20260402_v2/clt_weights/progen2-medium/step_100000`
- Final status: `Training complete! Final step: 100000`

Smoke test
----------

`protgpt2` smoke test was also returned:

- Remote output: `/oss-pvc/zhk_zip/outputs/r2_clt_protgpt2_test_20260402`
- Final checkpoint: `/oss-pvc/zhk_zip/outputs/r2_clt_protgpt2_test_20260402/clt_weights/protgpt2/step_200`
- Final status: `Training complete! Final step: 200`

Returned files
--------------

- `logs/r2_clt_protgpt2_full_20260402__clt_protgpt2.log`
- `logs/r2_clt_protgpt2_full_20260402__launcher.log`
- `logs/r2_clt_zymctrl_full_20260402__clt_zymctrl.log`
- `logs/r2_clt_zymctrl_full_20260402__launcher.log`
- `logs/r2_clt_progen2_medium_full_20260402__clt_progen2-medium.log`
- `logs/r2_clt_progen2_medium_full_20260402__launcher.log`
- `logs/r2_clt_progen2_medium_full_20260402_v2__clt_progen2-medium.log`
- `logs/r2_clt_progen2_medium_full_20260402_v2__launcher.log`
- `logs/r2_clt_protgpt2_test_20260402__clt_protgpt2.log`
- `logs/r2_clt_protgpt2_test_20260402__launcher.log`

Quick reading guide
-------------------

- The `clt_*.log` files are the main training logs.
- The matching `launcher.log` files contain the same end-of-run summary and are useful for quick tail checks.
- The first `progen2-medium` attempt is kept because it records the offline loading failure that was fixed before the successful retry.
