# Comparative Surveyor

Survey insights tooling for comparing pre- and post-intervention surveys. Upload the two
waves, pick the Likert-scale question columns, and the app produces summary percentages,
mean scores, a comparative chart, paired significance testing and an Excel workbook.

Two front ends share the same analysis:

| App | Path | Hosted on |
|---|---|---|
| **Dash** (current) | `dash_app/app.py` | Plotly Cloud |
| Streamlit (original) | `streamlit_app.py` | Streamlit Community Cloud |

The Dash version exists because the Streamlit deployment sleeps when idle. It also
replaces the Matplotlib charts with native Plotly, so the chart is interactive and
exports to PNG from the chart toolbar.

## Running the Dash app locally

```bash
cd dash_app
pip install -r requirements.txt
python app.py          # http://127.0.0.1:8050
```

## Publishing to Plotly Cloud

```bash
pip install "dash[cloud]"
plotly user login
cd dash_app
plotly app publish --name tssfl-survey-insights
```

Size limits are 80 MiB when uploading through cloud.plotly.com and 200 MiB via the CLI.
Set the app to public ("Can view" for anyone with the link) if you intend to embed it.

## Input format

One row per respondent, one column per question. Cell values must be one of:

```
Strongly Disagree | Disagree | Somewhat Agree | Agree | Strongly Agree
```

which map to scores 1-5. Anything else is ignored when computing the mean. The pre and
post files must share the question column names; only shared columns are offered for
selection.

## Fixes applied in the 2026-08 revision

Both apps carried the same defects; all are fixed in each.

- **`run_statistical_tests` was defined twice.** Python kept the second, simpler body,
  so the normal-approximation handling and the "data may not be truly paired" caveat
  in the first definition were unreachable.
- **`wilcoxon(..., exact=False)` is not a valid SciPy call.** The parameter is `method`;
  the correct value here is `method='approx'`. The bad keyword raised `TypeError`, which
  the broad `except Exception` swallowed - silently downgrading every run to a paired
  t-test.
- **Bar-chart labels were paired with the wrong bars.** The labels were re-sorted with
  `sorted(questions, key=lambda x: int(x[1:]), reverse=True)` *after* the bars had been
  drawn in the original order. The same expression also raised `ValueError` on any
  column not named like `Q7`.
- **Identical pre/post waves produced a NaN p-value** rather than a message.
- **A second `ax.legend()` call** overrode the Bar branch's legend placement.
- **The axis label read "Mean Score" on the x-axis for the Line chart**, where the means
  are plotted on y.
- **Empty input crashed the insights section** on `deltas.idxmax()`.

## Related

- Live discussion: https://www.tssfl.com/viewtopic.php?t=6918
