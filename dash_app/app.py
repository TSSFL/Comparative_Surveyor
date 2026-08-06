"""TSSFL Survey Insights Generator - Dash edition.

A port of streamlit_app.py to Dash, for publishing on Plotly Cloud.
Processes pre/post intervention survey data, builds comparative charts,
runs paired significance tests and exports a summary workbook.

Run locally:  python app.py
Publish:      plotly app publish --name tssfl-survey-insights
"""

import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from scipy.stats import ttest_rel, wilcoxon

BRAND = "Created at www.tssfl.com"
PRE_DEFAULT = "#1f77b4"
POST_DEFAULT = "#ff7f0e"

FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
HEADING_FONT = "'Space Grotesk', Inter, sans-serif"


class DataTransformer:
    """Turns raw Likert responses into per-question percentages and mean scores."""

    def __init__(self, response_categories=None):
        self.response_categories = response_categories or [
            "Strongly Disagree", "Disagree", "Somewhat Agree", "Agree", "Strongly Agree"
        ]
        self.score_mapping = {
            "Strongly Disagree": 1,
            "Disagree": 2,
            "Somewhat Agree": 3,
            "Agree": 4,
            "Strongly Agree": 5,
        }

    def transform_data(self, data, columns_to_process):
        target_columns = data[columns_to_process].copy()
        for col in target_columns.columns:
            target_columns[col] = target_columns[col].astype(str).str.strip()
            target_columns[col] = target_columns[col].str.replace(r"\s+", " ", regex=True)

        results = []
        for col in target_columns.columns:
            scores = target_columns[col].map(self.score_mapping)
            mean_score = scores.mean()

            response_counts = target_columns[col].value_counts(normalize=True) * 100
            response_percentages = {
                resp: response_counts.get(resp, 0) for resp in self.response_categories
            }
            response_percentages["Mean Score"] = mean_score
            response_percentages["Question"] = col
            results.append(response_percentages)

        return pd.DataFrame(
            results, columns=["Question"] + self.response_categories + ["Mean Score"]
        )


def parse_upload(contents, filename):
    """Decode a dcc.Upload payload into a DataFrame."""
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")))
    return pd.read_excel(io.BytesIO(decoded))


def build_figure(pre_transformed, post_transformed, color_pre, color_post, chart_type):
    """Comparative chart of pre- vs post-intervention mean scores.

    Question order is preserved exactly as selected. The Streamlit original
    re-sorted the tick labels after drawing the bars, which silently paired
    each label with the wrong bar.
    """
    questions = pre_transformed["Question"].tolist()
    pre_mean = pre_transformed["Mean Score"].tolist()
    post_mean = post_transformed["Mean Score"].tolist()

    fig = go.Figure()

    if chart_type == "Bar":
        # Reverse so the first question appears at the top of a horizontal chart.
        order = list(reversed(range(len(questions))))
        y = [questions[i] for i in order]
        fig.add_bar(
            y=y, x=[pre_mean[i] for i in order], name="Pre-Intervention",
            orientation="h", marker_color=color_pre,
            text=[f"{pre_mean[i]:.2f}" for i in order], textposition="outside",
        )
        fig.add_bar(
            y=y, x=[post_mean[i] for i in order], name="Post-Intervention",
            orientation="h", marker_color=color_post,
            text=[f"{post_mean[i]:.2f}" for i in order], textposition="outside",
        )
        fig.update_layout(barmode="group", xaxis_title="Mean Score (scale 1 to 5)")
        fig.update_xaxes(range=[0, 5.6])
        height = 260 + 46 * len(questions)
    else:
        fig.add_scatter(
            x=questions, y=pre_mean, name="Pre-Intervention", mode="lines+markers",
            line_color=color_pre, marker_size=9,
        )
        fig.add_scatter(
            x=questions, y=post_mean, name="Post-Intervention", mode="lines+markers",
            line_color=color_post, marker_size=9,
        )
        fig.update_layout(yaxis_title="Mean Score (scale 1 to 5)")
        fig.update_yaxes(range=[0, 5.6])
        fig.update_xaxes(tickangle=-45)
        height = 620

    fig.update_layout(
        title="Comparison of Mean Scores (Pre- vs. Post-Intervention)",
        height=height,
        font=dict(family=FONT, size=14),
        title_font=dict(family=HEADING_FONT, size=19),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=30, t=110, b=60),
        plot_bgcolor="white",
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        xref="paper", yref="paper", x=1, y=1.10, xanchor="right",
        text=BRAND, showarrow=False, font=dict(color="green", size=13, family=FONT),
    )
    return fig


def generate_ai_insights(pre_df, post_df):
    deltas = (post_df["Mean Score"] - pre_df["Mean Score"]).dropna()
    if deltas.empty:
        return "**AI-Assisted Insights**\n\n- No comparable questions to analyse."

    improved = deltas[deltas > 0]
    declined = deltas[deltas < 0]
    unchanged = deltas[deltas == 0]

    lines = [
        "**AI-Assisted Insights**",
        "",
        f"- {len(improved)} question(s) showed improvement.",
        f"- {len(declined)} question(s) declined.",
        f"- {len(unchanged)} question(s) remained unchanged.",
        "",
        f"**Improved:** {', '.join(improved.index) or '-'}",
        "",
        f"**Declined:** {', '.join(declined.index) or '-'}",
        "",
        f"**Unchanged:** {', '.join(unchanged.index) or '-'}",
        "",
        f"Greatest positive change: **{deltas.max():.2f}** in *{deltas.idxmax()}*.",
        "",
        f"Greatest negative change: **{deltas.min():.2f}** in *{deltas.idxmin()}*.",
    ]
    return "\n".join(lines)


def run_statistical_tests(pre_df, post_df):
    """Paired significance test across question mean scores.

    The Streamlit original defined this function twice; Python kept the second,
    simpler body, so the normal-approximation handling and the pairing caveat
    below were never reached. This is the intended version.
    """
    paired_data = pd.DataFrame(
        {"pre": pre_df["Mean Score"], "post": post_df["Mean Score"]}
    ).dropna()

    if len(paired_data) < 2:
        return (
            "**Statistical Significance Test**\n\n"
            "- Not enough valid data for statistical testing "
            "(at least two paired questions are required)."
        )

    if (paired_data["pre"] - paired_data["post"]).abs().max() == 0:
        return (
            "**Statistical Significance Test**\n\n"
            "- The two waves are identical on every question, so there is no "
            "difference to test."
        )

    # scipy names this parameter `method`; 'approx' is the normal approximation,
    # which stays defined when tied ranks make the exact test unavailable.
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            stat, p = wilcoxon(
                paired_data["pre"], paired_data["post"],
                alternative="two-sided", method="approx",
            )
        test_name = "Wilcoxon Signed-Rank Test"
        if not np.isfinite(p):
            raise ValueError("Wilcoxon returned a non-finite p-value")
    except (ValueError, TypeError):
        try:
            stat, p = ttest_rel(paired_data["pre"], paired_data["post"])
            test_name = "Paired t-Test"
            if not np.isfinite(p):
                raise ValueError("t-test returned a non-finite p-value")
        except (ValueError, TypeError):
            return (
                "**Statistical Significance Test**\n\n"
                "- Data unsuitable for both paired tests."
            )

    if len(pre_df) != len(post_df):
        test_name += (
            " (Note: data may not be truly paired. "
            "Consider an independent-samples test.)"
        )

    verdict = (
        "Statistically significant (p < 0.05)" if p < 0.05
        else "Not statistically significant (p >= 0.05)"
    )
    return (
        "**Statistical Significance Test**\n\n"
        f"- Test used: {test_name}\n"
        f"- Test statistic: {stat:.4f}\n"
        f"- p-value: {p:.4f}\n\n"
        f"**{verdict}**"
    )


def to_excel_bytes(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in df_dict.items():
            safe_name = "".join(c for c in name if c not in r"[]:*?/\\").strip()[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


def summary_table(transformed):
    """DataTable of one survey wave, numbers rounded for display."""
    display = transformed.copy()
    for col in display.columns:
        if col != "Question":
            display[col] = display[col].astype(float).round(2)
    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": FONT, "fontSize": 14, "padding": "8px 10px",
            "textAlign": "right", "minWidth": 90,
        },
        style_cell_conditional=[
            {"if": {"column_id": "Question"}, "textAlign": "left", "minWidth": 160}
        ],
        style_header={
            "fontFamily": HEADING_FONT, "fontWeight": 600,
            "backgroundColor": "#f5f7fa", "border": "none",
            "borderBottom": "2px solid #dde3ea",
        },
        style_data={"borderBottom": "1px solid #eef1f5", "border": "none"},
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

app = Dash(__name__, title="TSSFL Survey Insights Generator")
server = app.server  # WSGI entry point used in production

CARD = {
    "background": "white", "borderRadius": 12, "padding": "20px 24px",
    "marginBottom": 20, "border": "1px solid #e6eaf0",
}
H2 = {"fontFamily": HEADING_FONT, "fontSize": 20, "margin": "0 0 14px"}

UPLOAD_STYLE = {
    "width": "100%", "height": 92, "lineHeight": "92px", "borderWidth": 2,
    "borderStyle": "dashed", "borderColor": "#c3ccd8", "borderRadius": 10,
    "textAlign": "center", "cursor": "pointer", "color": "#5b6675",
    "fontFamily": FONT,
}

app.layout = html.Div(
    style={
        "fontFamily": FONT, "maxWidth": 1120, "margin": "0 auto",
        "padding": "28px 20px 60px", "background": "#fbfcfd",
    },
    children=[
        html.H1(
            "Survey Insights Generator",
            style={"fontFamily": HEADING_FONT, "fontSize": 30, "marginBottom": 6},
        ),
        dcc.Markdown(
            "Upload pre- and post-intervention survey files to calculate summary "
            "statistics, build comparative charts, run paired significance tests "
            "and export a workbook.",
            style={"color": "#5b6675", "marginBottom": 24},
        ),

        html.Div(style=CARD, children=[
            html.H2("1. Upload survey data", style=H2),
            html.Div(
                style={"display": "flex", "gap": 18, "flexWrap": "wrap"},
                children=[
                    html.Div(style={"flex": "1 1 320px"}, children=[
                        dcc.Upload(
                            id="upload-pre",
                            children=html.Div("Pre-Intervention survey (CSV or Excel)"),
                            style=UPLOAD_STYLE, multiple=False,
                        ),
                        html.Div(id="status-pre", style={"marginTop": 8, "fontSize": 14}),
                    ]),
                    html.Div(style={"flex": "1 1 320px"}, children=[
                        dcc.Upload(
                            id="upload-post",
                            children=html.Div("Post-Intervention survey (CSV or Excel)"),
                            style=UPLOAD_STYLE, multiple=False,
                        ),
                        html.Div(id="status-post", style={"marginTop": 8, "fontSize": 14}),
                    ]),
                ],
            ),
        ]),

        html.Div(id="controls-card", style={**CARD, "display": "none"}, children=[
            html.H2("2. Select questions and chart", style=H2),
            html.Label("Likert-scale question columns",
                       style={"fontSize": 14, "color": "#5b6675"}),
            dcc.Dropdown(id="column-select", multi=True, placeholder="Select columns...",
                         style={"marginBottom": 18}),
            html.Div(
                style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                       "alignItems": "center"},
                children=[
                    html.Div([
                        html.Label("Chart type",
                                   style={"fontSize": 14, "color": "#5b6675",
                                          "display": "block", "marginBottom": 6}),
                        dcc.RadioItems(
                            id="chart-type",
                            options=[{"label": " Bar", "value": "Bar"},
                                     {"label": " Line", "value": "Line"}],
                            value="Bar",
                            inline=True,
                            inputStyle={"marginRight": 6, "marginLeft": 14},
                        ),
                    ]),
                    html.Div([
                        html.Label("Pre-Intervention colour",
                                   style={"fontSize": 14, "color": "#5b6675",
                                          "display": "block", "marginBottom": 6}),
                        dcc.Input(id="color-pre", type="text", value=PRE_DEFAULT,
                                  style={"width": 64, "height": 34, "padding": 2}),
                    ]),
                    html.Div([
                        html.Label("Post-Intervention colour",
                                   style={"fontSize": 14, "color": "#5b6675",
                                          "display": "block", "marginBottom": 6}),
                        dcc.Input(id="color-post", type="text", value=POST_DEFAULT,
                                  style={"width": 64, "height": 34, "padding": 2}),
                    ]),
                ],
            ),
        ]),

        html.Div(id="results", children=[]),

        dcc.Store(id="store-pre"),
        dcc.Store(id="store-post"),
        dcc.Store(id="store-summaries"),
        dcc.Download(id="download-excel"),
    ],
)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

def _read_into_store(contents, filename):
    if contents is None:
        return None, ""
    try:
        df = parse_upload(contents, filename)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        return None, html.Span(f"Could not read {filename}: {exc}",
                               style={"color": "#c0392b"})
    msg = html.Span(f"{filename} - {len(df)} rows, {len(df.columns)} columns",
                    style={"color": "#1e7e34"})
    return df.to_json(orient="split"), msg


@app.callback(
    Output("store-pre", "data"), Output("status-pre", "children"),
    Input("upload-pre", "contents"), State("upload-pre", "filename"),
    prevent_initial_call=True,
)
def load_pre(contents, filename):
    return _read_into_store(contents, filename)


@app.callback(
    Output("store-post", "data"), Output("status-post", "children"),
    Input("upload-post", "contents"), State("upload-post", "filename"),
    prevent_initial_call=True,
)
def load_post(contents, filename):
    return _read_into_store(contents, filename)


@app.callback(
    Output("column-select", "options"),
    Output("controls-card", "style"),
    Input("store-pre", "data"), Input("store-post", "data"),
)
def populate_columns(pre_json, post_json):
    hidden = {**CARD, "display": "none"}
    if not pre_json or not post_json:
        return [], hidden

    pre = pd.read_json(io.StringIO(pre_json), orient="split")
    post = pd.read_json(io.StringIO(post_json), orient="split")
    # Preserve the pre-file column order; a set intersection would randomise it.
    post_cols = set(post.columns)
    common = [c for c in pre.columns if c in post_cols]
    return [{"label": c, "value": c} for c in common], {**CARD, "display": "block"}


@app.callback(
    Output("results", "children"),
    Output("store-summaries", "data"),
    Input("column-select", "value"),
    Input("chart-type", "value"),
    Input("color-pre", "value"),
    Input("color-post", "value"),
    State("store-pre", "data"),
    State("store-post", "data"),
)
def build_results(columns, chart_type, color_pre, color_post, pre_json, post_json):
    if not columns or not pre_json or not post_json:
        return [], None

    pre = pd.read_json(io.StringIO(pre_json), orient="split")
    post = pd.read_json(io.StringIO(post_json), orient="split")

    dt = DataTransformer()
    pre_t = dt.transform_data(pre, columns)
    post_t = dt.transform_data(post, columns)

    if pre_t["Question"].tolist() != post_t["Question"].tolist():
        return html.Div("Questions in the pre and post datasets do not match.",
                        style={"color": "#c0392b"}), None

    fig = build_figure(pre_t, post_t, color_pre, color_post, chart_type)
    pre_idx = pre_t.set_index("Question")
    post_idx = post_t.set_index("Question")

    children = [
        html.Div(style=CARD, children=[
            html.H2("3. Comparative chart", style=H2),
            dcc.Graph(figure=fig, config={
                "displaylogo": False,
                "toImageButtonOptions": {"format": "png", "filename": "tssfl_survey_chart",
                                         "scale": 2},
            }),
            html.Div("Use the camera icon on the chart toolbar to download it as a PNG.",
                     style={"fontSize": 13, "color": "#7a8494", "marginTop": 4}),
        ]),
        html.Div(style=CARD, children=[
            html.H2("4. Tabular summary", style=H2),
            html.H3("Pre-Intervention", style={**H2, "fontSize": 16}),
            summary_table(pre_t),
            html.H3("Post-Intervention", style={**H2, "fontSize": 16, "marginTop": 22}),
            summary_table(post_t),
            html.Button("Download summary workbook (.xlsx)", id="btn-excel", n_clicks=0,
                        style={"marginTop": 18, "padding": "10px 18px", "borderRadius": 8,
                               "border": "1px solid #1f77b4", "background": "#1f77b4",
                               "color": "white", "cursor": "pointer",
                               "fontFamily": HEADING_FONT, "fontSize": 14}),
        ]),
        html.Div(style=CARD, children=[
            html.H2("5. Insights", style=H2),
            dcc.Markdown(generate_ai_insights(pre_idx, post_idx)),
        ]),
        html.Div(style=CARD, children=[
            html.H2("6. Statistical testing", style=H2),
            dcc.Markdown(run_statistical_tests(pre_idx, post_idx)),
        ]),
    ]

    summaries = {"pre": pre_t.to_json(orient="split"),
                 "post": post_t.to_json(orient="split")}
    return children, summaries


@app.callback(
    Output("download-excel", "data"),
    Input("btn-excel", "n_clicks"),
    State("store-summaries", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, summaries):
    if not n_clicks or not summaries:
        return no_update
    pre_t = pd.read_json(io.StringIO(summaries["pre"]), orient="split")
    post_t = pd.read_json(io.StringIO(summaries["post"]), orient="split")
    payload = to_excel_bytes({"Pre-Intervention": pre_t, "Post-Intervention": post_t})
    return dcc.send_bytes(payload, "survey_summary.xlsx")


if __name__ == "__main__":
    app.run(debug=True)
