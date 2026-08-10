# External resources

The current transfer programme declares its model, dataset, and tool inputs in [`manifests/interpretability_transfer_resources.json`](manifests/interpretability_transfer_resources.json). The manifest records environment-variable contracts rather than infrastructure paths or availability claims. Large payloads remain ignored and outside Git.

Resolve the declared `TRANSFER_*` variables in the protected runtime environment, then verify the registered panel with:

```bash
python scripts/transfer/panel_contract.py --verify
```

`setup_h200_external_env.sh` is retained only for the earlier `BIOCC_*` resource chain. It is not the environment contract for the current transfer programme. Follow the external cluster access runbook for staging and recovery; do not copy live pod names or storage paths into repository documentation.
