# Additional source registry

Downloaded 2026-09-05. Original files are retained without re-encoding. Each experiment records source hashes in `audit.json`.

| Directory | CSV download | Metadata |
|---|---|---|
| `uci_529` | https://archive.ics.uci.edu/static/public/529/data.csv | https://archive.ics.uci.edu/api/dataset?id=529 |
| `uci_891` | https://archive.ics.uci.edu/static/public/891/data.csv | https://archive.ics.uci.edu/api/dataset?id=891 |

SHA-256 of the CSV files:

```text
uci_529/data.csv  f6bf14c5d2e939feccd78750da2542f131b78d80355ccf35f29554b1f03fcb97
uci_891/data.csv  9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964
```

Sylhet is licensed CC BY 4.0 by UCI. For CDC, UCI directs users to the linked upstream dataset for licensing; those terms have not been independently confirmed. See [study documentation](../../docs/ADDITIONAL_DATASET_RESULTS.md) for population, duplicate-profile and target-definition limitations. These files must not be concatenated with Pima: they do not share its feature or target contract.
