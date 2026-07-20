# Example pipelines

This directory is deliberately outside `src/sigvue`: it is plugin code, not framework code. The waterfall example is arranged so the whole directory can be copied into another repository.

```text
example_pipelines/
├── io/sigmf/          shared file-format I/O
├── style/            shared Plotly appearance
├── comms/            static constellation and eye-diagram pipeline
├── waterfall/
│   ├── source.py       discovery and loader binding
│   ├── delivery.py     window-selection and ranged reads
│   ├── analysis.py     numerical processing
│   ├── plots.py        controls, Plotly figure, and tab layout
│   └── workspace.py    framework object assembly only
├── scripts/generate_lte.py
└── browser.toml
```

Generate the synthetic LTE uplink/downlink and QPSK/16-QAM/64-QAM SigMF recordings, then launch Sigvue:

```bash
python example_pipelines/scripts/generate_lte.py
python example_pipelines/scripts/generate_comms.py
sigvue --config example_pipelines/browser.toml
```

Open <http://127.0.0.1:8000>. Generated data stays untracked under `example_pipelines/data/lte/`.

## What the waterfall example demonstrates

The workspace assembly contains only framework objects. Each object owns one
kind of behavior and declares controls only through the request-scoped API it
receives:

| Module object | Framework contract | Demonstrated API |
| --- | --- | --- |
| `recording_source(root)` | Returns `DirectorySource` | File discovery, nested paths, SigMF opening, and catalog summaries. |
| `WindowedSamples()` | `Delivery` | `DeliveryContext.windowed()` with a decimated full-record power overview. |
| `WaterfallAnalysis()` | `Analysis` | `ParameterContext.select()` for FFT size and overlap, followed by ordinary NumPy processing. |
| `WaterfallPresentation()` | `Presentation` | Tabs, Plotly rendering, colormap, paired limits, toggle, trace-style picker, statistics, theme, and bounded axes through `ViewContext`. |

The communications example is deliberately smaller: `DirectorySource`, a
static `CommsAnalysis`, and a `CommsPresentation` with constellation and eye
tabs. It shows what can be omitted when no delivery, processing parameters,
annotations, or export behavior is needed.

## Test

From the repository root:

```bash
python -m pytest -q example_pipelines/tests
```

These tests live with the pipelines and are run as an explicit step in the
repository's publish workflow.
