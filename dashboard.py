"""Customer Churn Intelligence Platform - 5-tab Plotly Dash dashboard."""

import os
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context

from config import (
    MODEL_PATH, SHAP_PATH,
    BG, CARD_BG, BORDER, TEXT, MUTED,
    ACCENT, GREEN, RED, YELLOW, BLUE, ORANGE,
    CHURN_COLOR, RETAIN_COLOR,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
    MONTHLY_REVENUE,
)
from data_loader import load_data, get_feature_stats
from model import (
    train_and_save, load_model, load_shap,
    predict_single, get_model_metrics, get_feature_importances,
)

# ---------------------------------------------------------------------------
# Module-level data / model loading
# ---------------------------------------------------------------------------
print("Loading data...")
df = load_data()
stats = get_feature_stats(df)

if not os.path.exists(MODEL_PATH):
    print("Training model (first run, please wait)...")
    train_and_save(df)

print("Loading model...")
pipeline = load_model()
shap_values, shap_feature_names, X_shap_sample = load_shap()
metrics = get_model_metrics(pipeline, df)
feat_imp = get_feature_importances(pipeline)

# ---------------------------------------------------------------------------
# Shared Plotly layout dict
# ---------------------------------------------------------------------------
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d0d20",
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        font=dict(color=TEXT),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        color=TEXT,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        color=TEXT,
    ),
)


def make_layout(**kwargs) -> dict:
    """Return a merged Plotly layout dict combining PL defaults with overrides."""
    out = dict(**PL)
    out.update(kwargs)
    return out


def _hex_to_rgb(hex_color: str) -> str:
    """Convert a 6-digit hex color string to an r,g,b string for rgba() usage."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def kpi_card(title: str, value: str, color: str = ACCENT) -> dbc.Col:
    """Return a Bootstrap column containing a glassmorphism KPI card."""
    return dbc.Col(
        html.Div(
            [
                html.P(title, className="kpi-label"),
                html.H3(value, style={"color": color, "margin": 0, "fontWeight": "700"}),
            ],
            className="kpi-card",
        ),
        xs=12, sm=6, md=4, lg=2,
    )


def metric_card(label: str, value: str, color: str = ACCENT) -> dbc.Col:
    """Return a Bootstrap column with a model metric card."""
    return dbc.Col(
        html.Div(
            [
                html.P(label, className="kpi-label"),
                html.H4(value, style={"color": color, "margin": 0, "fontWeight": "700"}),
            ],
            className="kpi-card",
        ),
        xs=12, sm=6, md=3,
    )


def _radio(label: str, component_id: str, options, value) -> html.Div:
    """Return a labelled RadioItems control for the What-If sidebar."""
    if isinstance(options[0], str):
        opts = [{"label": o, "value": o} for o in options]
    else:
        opts = [{"label": str(o), "value": o} for o in options]
    return html.Div(
        [
            html.Label(
                label,
                style={"color": MUTED, "fontSize": "0.78rem", "marginBottom": "4px", "display": "block"},
            ),
            dcc.RadioItems(
                id=component_id,
                options=opts,
                value=value,
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"color": TEXT, "fontSize": "0.82rem", "marginRight": "12px"},
            ),
        ],
        style={"marginBottom": "14px"},
    )


def _slider(label: str, component_id: str, min_v, max_v, step, value, unit="") -> html.Div:
    """Return a labelled Slider control for the What-If sidebar."""
    return html.Div(
        [
            html.Label(
                f"{label}: {value}{unit}",
                style={"color": MUTED, "fontSize": "0.78rem", "marginBottom": "4px", "display": "block"},
            ),
            dcc.Slider(
                id=component_id,
                min=min_v, max=max_v, step=step, value=value,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                className="dark-slider",
            ),
        ],
        style={"marginBottom": "18px"},
    )

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Churn Intelligence Platform",
)
server = app.server

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header = html.Div(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Customer Churn Intelligence Platform",
                            style={
                                "background": f"linear-gradient(135deg, {ACCENT}, {BLUE})",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "fontWeight": "800",
                                "fontSize": "1.9rem",
                                "marginBottom": "4px",
                            },
                        ),
                        html.P(
                            "XGBoost + SHAP Explainability  |  IBM Telco Dataset  |  7,043 Customers",
                            style={"color": MUTED, "marginBottom": 0, "fontSize": "0.85rem"},
                        ),
                    ],
                    md=9,
                ),
                dbc.Col(
                    dbc.Button(
                        "Retrain Model",
                        id="retrain-btn",
                        color="primary",
                        outline=True,
                        size="sm",
                        style={"borderColor": ACCENT, "color": ACCENT},
                    ),
                    md=3,
                    className="d-flex align-items-center justify-content-end",
                ),
            ],
            align="center",
        ),
        fluid=True,
    ),
    style={
        "background": "linear-gradient(180deg, rgba(124,58,237,0.18) 0%, rgba(10,10,26,0.0) 100%)",
        "borderBottom": f"1px solid {BORDER}",
        "padding": "18px 0 14px",
        "marginBottom": "0",
    },
)

retrain_toast = dbc.Toast(
    "Model retrained successfully.",
    id="retrain-toast",
    header="Done",
    is_open=False,
    dismissable=True,
    icon="success",
    style={"position": "fixed", "top": 80, "right": 20, "zIndex": 9999},
)

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Executive Overview",        tab_id="tab-overview",    label_style={"color": MUTED}, active_label_style={"color": ACCENT}),
        dbc.Tab(label="Segment Analysis",          tab_id="tab-segment",     label_style={"color": MUTED}, active_label_style={"color": ACCENT}),
        dbc.Tab(label="Model Performance",         tab_id="tab-performance", label_style={"color": MUTED}, active_label_style={"color": ACCENT}),
        dbc.Tab(label="What-If Simulator",         tab_id="tab-whatif",      label_style={"color": MUTED}, active_label_style={"color": ACCENT}),
        dbc.Tab(label="Feature Importance & SHAP", tab_id="tab-shap",        label_style={"color": MUTED}, active_label_style={"color": ACCENT}),
    ],
    id="main-tabs",
    active_tab="tab-overview",
    style={"backgroundColor": BG, "borderBottom": f"1px solid {BORDER}", "paddingLeft": "20px"},
)

app.layout = html.Div(
    [
        header,
        tabs,
        retrain_toast,
        html.Div(id="tab-content", style={"padding": "24px 0"}),
    ],
    style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "Inter, system-ui, sans-serif"},
)

# ===========================================================================
# TAB CONTENT RENDERERS
# ===========================================================================

def render_overview() -> html.Div:
    """Return the Executive Overview tab layout."""
    return html.Div(
        dbc.Container(
            [
                dbc.Row(
                    [
                        kpi_card('Total Customers',      f"{stats['total_customers']:,}",      BLUE),
                        kpi_card('Churn Rate',           f"{stats['churn_rate']:.1f}%",        RED),
                        kpi_card('Revenue at Risk / Mo', f"${stats['at_risk_revenue']:,.0f}",  ORANGE),
                        kpi_card('Avg Tenure',           f"{stats['avg_tenure']:.1f} mo",      GREEN),
                        kpi_card('Avg Monthly Charge',   f"${stats['avg_monthly']:.2f}",       YELLOW),
                    ],
                    className='g-3 mb-4',
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-gauge',    config={'displayModeBar': False}), md=4),
                        dbc.Col(dcc.Graph(id='fig-donut',    config={'displayModeBar': False}), md=4),
                        dbc.Col(dcc.Graph(id='fig-contract', config={'displayModeBar': False}), md=4),
                    ],
                    className='g-3 mb-4',
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-tenure-hist', config={'displayModeBar': False}), md=12),
                    ],
                    className='g-3',
                ),
            ],
            fluid=True,
        )
    )


def render_segment() -> html.Div:
    """Return the Segment Analysis tab layout."""
    seg_options = [
        {'label': 'Gender',            'value': 'gender'},
        {'label': 'Contract',          'value': 'Contract'},
        {'label': 'Payment Method',    'value': 'PaymentMethod'},
        {'label': 'Internet Service',  'value': 'InternetService'},
        {'label': 'Senior Citizen',    'value': 'SeniorCitizen'},
        {'label': 'Partner',           'value': 'Partner'},
        {'label': 'Dependents',        'value': 'Dependents'},
        {'label': 'Phone Service',     'value': 'PhoneService'},
        {'label': 'Paperless Billing', 'value': 'PaperlessBilling'},
        {'label': 'Tech Support',      'value': 'TechSupport'},
        {'label': 'Online Security',   'value': 'OnlineSecurity'},
        {'label': 'Streaming TV',      'value': 'StreamingTV'},
    ]
    return html.Div(
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.Label('Segment feature', style={'color': MUTED, 'fontSize': '0.8rem', 'marginBottom': '6px'}),
                                dcc.Dropdown(
                                    id='seg-dropdown',
                                    options=seg_options,
                                    value='Contract',
                                    clearable=False,
                                    style={'backgroundColor': CARD_BG, 'color': TEXT},
                                    className='dark-dropdown',
                                ),
                            ]
                        ),
                        md=4,
                    ),
                    className='mb-4',
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-seg-bar',     config={'displayModeBar': False}), md=6),
                        dbc.Col(dcc.Graph(id='fig-seg-heatmap', config={'displayModeBar': False}), md=6),
                    ],
                    className='g-3 mb-4',
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-seg-violin', config={'displayModeBar': False}), md=6),
                        dbc.Col(dcc.Graph(id='fig-seg-tenure', config={'displayModeBar': False}), md=6),
                    ],
                    className='g-3',
                ),
            ],
            fluid=True,
        )
    )


def render_performance() -> html.Div:
    """Return the Model Performance tab layout."""
    return html.Div(
        dbc.Container(
            [
                dbc.Row(
                    [
                        metric_card('Accuracy',  f"{metrics['accuracy']*100:.1f}%",  BLUE),
                        metric_card('Precision', f"{metrics['precision']*100:.1f}%", GREEN),
                        metric_card('Recall',    f"{metrics['recall']*100:.1f}%",    YELLOW),
                        metric_card('F1 Score',  f"{metrics['f1']*100:.1f}%",        ORANGE),
                        metric_card('AUC-ROC',   f"{metrics['auc']:.4f}",            ACCENT),
                    ],
                    className='g-3 mb-4',
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-confusion', config={'displayModeBar': False}), md=4),
                        dbc.Col(dcc.Graph(id='fig-roc',       config={'displayModeBar': False}), md=4),
                        dbc.Col(dcc.Graph(id='fig-pr',        config={'displayModeBar': False}), md=4),
                    ],
                    className='g-3 mb-4',
                ),
                dbc.Row(
                    dbc.Col(dcc.Graph(id='fig-feat-imp', config={'displayModeBar': False}), md=12),
                    className='g-3',
                ),
            ],
            fluid=True,
        )
    )

def render_whatif() -> html.Div:
    """Return the What-If Simulator tab layout."""
    left_panel = html.Div(
        [
            html.H6('Customer Profile', style={'color': ACCENT, 'fontWeight': '700', 'marginBottom': '16px'}),
            _slider('Tenure',          'wi-tenure',  0,  72,  1, 12, ' mo'),
            _slider('Monthly Charges', 'wi-monthly', 18, 120, 1, 65, '$'),
            _radio('Contract',          'wi-contract',     ['Month-to-month', 'One year', 'Two year'],       'Month-to-month'),
            _radio('Internet Service',  'wi-internet',     ['DSL', 'Fiber optic', 'No'],                     'Fiber optic'),
            _radio('Gender',            'wi-gender',       ['Male', 'Female'],                               'Male'),
            _radio('Senior Citizen',    'wi-senior',       [0, 1],                                           0),
            _radio('Partner',           'wi-partner',      ['Yes', 'No'],                                    'No'),
            _radio('Dependents',        'wi-dependents',   ['Yes', 'No'],                                    'No'),
            _radio('Phone Service',     'wi-phone',        ['Yes', 'No'],                                    'Yes'),
            _radio('Multiple Lines',    'wi-multiline',    ['Yes', 'No', 'No phone service'],                'No'),
            _radio('Online Security',   'wi-security',     ['Yes', 'No', 'No internet service'],             'No'),
            _radio('Online Backup',     'wi-backup',       ['Yes', 'No', 'No internet service'],             'No'),
            _radio('Device Protection', 'wi-device',       ['Yes', 'No', 'No internet service'],             'No'),
            _radio('Tech Support',      'wi-techsupport',  ['Yes', 'No', 'No internet service'],             'No'),
            _radio('Streaming TV',      'wi-streamingtv',  ['Yes', 'No', 'No internet service'],             'No'),
            _radio('Streaming Movies',  'wi-streamingmov', ['Yes', 'No', 'No internet service'],             'No'),
            _radio('Paperless Billing', 'wi-paperless',    ['Yes', 'No'],                                    'Yes'),
            html.Div(
                [
                    html.Label('Payment Method', style={'color': MUTED, 'fontSize': '0.78rem', 'marginBottom': '4px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='wi-payment',
                        options=[
                            {'label': 'Electronic check',     'value': 'Electronic check'},
                            {'label': 'Mailed check',         'value': 'Mailed check'},
                            {'label': 'Bank transfer (auto)', 'value': 'Bank transfer (automatic)'},
                            {'label': 'Credit card (auto)',   'value': 'Credit card (automatic)'},
                        ],
                        value='Electronic check',
                        clearable=False,
                        style={'backgroundColor': CARD_BG, 'color': TEXT},
                        className='dark-dropdown',
                    ),
                ],
                style={'marginBottom': '14px'},
            ),
        ],
        style={
            'backgroundColor': CARD_BG,
            'border': f'1px solid {BORDER}',
            'borderRadius': '12px',
            'padding': '20px',
            'overflowY': 'auto',
            'maxHeight': '82vh',
        },
    )

    right_panel = html.Div(
        [
            dcc.Graph(id='wi-gauge', config={'displayModeBar': False}, style={'height': '280px'}),
            html.Div(id='wi-risk-badge',   style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div(id='wi-revenue-text', style={'textAlign': 'center', 'color': MUTED, 'fontSize': '0.88rem', 'marginBottom': '20px'}),
            html.Hr(style={'borderColor': BORDER}),
            html.H6('Key Risk Factors', style={'color': ACCENT, 'fontWeight': '700', 'marginBottom': '12px', 'textAlign': 'center'}),
            html.Div(id='wi-shap-factors'),
        ],
        style={
            'backgroundColor': CARD_BG,
            'border': f'1px solid {BORDER}',
            'borderRadius': '12px',
            'padding': '20px',
        },
    )

    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(left_panel,  md=5),
                    dbc.Col(right_panel, md=7),
                ],
                className='g-4',
            ),
            fluid=True,
        )
    )


def render_shap_tab() -> html.Div:
    """Return the Feature Importance and SHAP tab layout."""
    return html.Div(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-shap-bar',      config={'displayModeBar': False}), md=6),
                        dbc.Col(dcc.Graph(id='fig-shap-beeswarm', config={'displayModeBar': False}), md=6),
                    ],
                    className='g-3 mb-4',
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig-corr-heatmap', config={'displayModeBar': False}), md=6),
                        dbc.Col(html.Div(id='fig-shap-table'),                                       md=6),
                    ],
                    className='g-3',
                ),
            ],
            fluid=True,
        )
    )

# ===========================================================================
# CALLBACKS
# ===========================================================================

@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'active_tab'),
)
def render_tab(tab: str):
    """Render the appropriate tab content based on the active tab ID."""
    if tab == 'tab-overview':
        return render_overview()
    if tab == 'tab-segment':
        return render_segment()
    if tab == 'tab-performance':
        return render_performance()
    if tab == 'tab-whatif':
        return render_whatif()
    if tab == 'tab-shap':
        return render_shap_tab()
    return html.Div('Unknown tab')


# ---- Retrain ---------------------------------------------------------------
@app.callback(
    Output('retrain-toast', 'is_open'),
    Input('retrain-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def retrain_model(n_clicks: int) -> bool:
    """Retrain the XGBoost model and SHAP values when the retrain button is clicked."""
    if n_clicks:
        global pipeline, shap_values, shap_feature_names, X_shap_sample, metrics, feat_imp
        train_and_save(df)
        pipeline = load_model()
        shap_values, shap_feature_names, X_shap_sample = load_shap()
        metrics  = get_model_metrics(pipeline, df)
        feat_imp = get_feature_importances(pipeline)
        return True
    return False

# ---- Overview figures -------------------------------------------------------
@app.callback(
    Output('fig-gauge',       'figure'),
    Output('fig-donut',       'figure'),
    Output('fig-contract',    'figure'),
    Output('fig-tenure-hist', 'figure'),
    Input('main-tabs', 'active_tab'),
)
def update_overview(tab):
    """Populate all Executive Overview figures."""
    churn_rate = stats['churn_rate']

    gauge = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=churn_rate,
        number={'suffix': '%', 'font': {'color': TEXT, 'size': 36}},
        delta={'reference': 20, 'relative': False, 'valueformat': '.1f',
               'increasing': {'color': RED}, 'decreasing': {'color': GREEN}},
        title={'text': 'Churn Rate', 'font': {'color': TEXT, 'size': 14}},
        gauge={
            'axis': {'range': [0, 50], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
            'bar': {'color': CHURN_COLOR},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 20],  'color': 'rgba(34,197,94,0.15)'},
                {'range': [20, 35], 'color': 'rgba(245,158,11,0.15)'},
                {'range': [35, 50], 'color': 'rgba(239,68,68,0.15)'},
            ],
        },
    ))
    gauge.update_layout(**make_layout(title='Overall Churn Rate', height=300))

    churned  = int(df['Churn'].sum())
    retained = len(df) - churned
    donut = go.Figure(go.Pie(
        labels=['Churned', 'Retained'],
        values=[churned, retained],
        hole=0.62,
        marker=dict(colors=[CHURN_COLOR, RETAIN_COLOR]),
        textfont=dict(color=TEXT),
        hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>',
    ))
    donut.update_layout(**make_layout(title='Churn Breakdown', height=300, showlegend=True))

    contract_data = (
        df.groupby('Contract')['Churn']
        .agg(['sum', 'count'])
        .reset_index()
    )
    contract_data['rate'] = contract_data['sum'] / contract_data['count'] * 100
    contract_fig = go.Figure()
    contract_fig.add_trace(go.Bar(
        x=contract_data['Contract'],
        y=contract_data['rate'],
        marker_color=[CHURN_COLOR, YELLOW, GREEN],
        text=[f'{v:.1f}%' for v in contract_data['rate']],
        textposition='outside',
        textfont=dict(color=TEXT),
        hovertemplate='Contract: %{x}<br>Churn Rate: %{y:.1f}%<extra></extra>',
    ))
    contract_fig.update_layout(**make_layout(
        title='Churn Rate by Contract Type',
        yaxis_title='Churn Rate (%)',
        height=300,
    ))

    bins   = [0, 12, 24, 36, 48, 60, 73]
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '60+']
    df2 = df.copy()
    df2['tenure_bin'] = pd.cut(df2['tenure'], bins=bins, labels=labels, right=False)
    tenure_data = (
        df2.groupby('tenure_bin', observed=True)['Churn']
        .agg(['sum', 'count'])
        .reset_index()
    )
    tenure_data['rate'] = tenure_data['sum'] / tenure_data['count'] * 100
    tenure_fig = go.Figure()
    tenure_fig.add_trace(go.Bar(
        x=tenure_data['tenure_bin'].astype(str),
        y=tenure_data['count'],
        name='Retained',
        marker_color=RETAIN_COLOR,
        hovertemplate='Tenure: %{x} mo<br>Customers: %{y:,}<extra></extra>',
    ))
    tenure_fig.add_trace(go.Bar(
        x=tenure_data['tenure_bin'].astype(str),
        y=tenure_data['sum'],
        name='Churned',
        marker_color=CHURN_COLOR,
        hovertemplate='Tenure: %{x} mo<br>Churned: %{y:,}<extra></extra>',
    ))
    tenure_fig.update_layout(**make_layout(
        title='Customer Distribution by Tenure',
        xaxis_title='Tenure (months)',
        yaxis_title='Number of Customers',
        barmode='overlay',
        height=320,
    ))

    return gauge, donut, contract_fig, tenure_fig

# ---- Segment figures --------------------------------------------------------
@app.callback(
    Output('fig-seg-bar',     'figure'),
    Output('fig-seg-heatmap', 'figure'),
    Output('fig-seg-violin',  'figure'),
    Output('fig-seg-tenure',  'figure'),
    Input('seg-dropdown', 'value'),
)
def update_segment(feature: str):
    """Populate all Segment Analysis figures for the selected categorical feature."""
    seg = (
        df.groupby(feature)['Churn']
        .agg(['sum', 'count'])
        .reset_index()
    )
    seg['rate'] = seg['sum'] / seg['count'] * 100
    seg_bar = go.Figure(go.Bar(
        x=seg[feature].astype(str),
        y=seg['rate'],
        marker_color=ACCENT,
        text=[f'{v:.1f}%' for v in seg['rate']],
        textposition='outside',
        textfont=dict(color=TEXT),
        hovertemplate=f'{feature}: %{{x}}<br>Churn Rate: %{{y:.1f}}%<extra></extra>',
    ))
    seg_bar.update_layout(**make_layout(
        title=f'Churn Rate by {feature}',
        yaxis_title='Churn Rate (%)',
        height=350,
    ))

    hm_data = (
        df.groupby(['Contract', 'InternetService'])['Churn']
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )
    heatmap_fig = go.Figure(go.Heatmap(
        z=hm_data.values,
        x=hm_data.columns.tolist(),
        y=hm_data.index.tolist(),
        colorscale=[[0, RETAIN_COLOR], [0.5, YELLOW], [1, CHURN_COLOR]],
        text=[[f'{v:.1f}%' for v in row] for row in hm_data.values],
        texttemplate='%{text}',
        hovertemplate='Contract: %{y}<br>Internet: %{x}<br>Churn: %{z:.1f}%<extra></extra>',
        colorbar=dict(tickfont=dict(color=TEXT), title=dict(text='%', font=dict(color=TEXT))),
    ))
    heatmap_fig.update_layout(**make_layout(
        title='Churn Rate: Contract x Internet Service',
        height=350,
    ))

    churned_charges  = df[df['Churn'] == 1]['MonthlyCharges']
    retained_charges = df[df['Churn'] == 0]['MonthlyCharges']
    violin_fig = go.Figure()
    violin_fig.add_trace(go.Violin(
        y=churned_charges,
        name='Churned',
        box_visible=True,
        meanline_visible=True,
        line_color=CHURN_COLOR,
        fillcolor='rgba(239,68,68,0.2)',
    ))
    violin_fig.add_trace(go.Violin(
        y=retained_charges,
        name='Retained',
        box_visible=True,
        meanline_visible=True,
        line_color=RETAIN_COLOR,
        fillcolor='rgba(34,197,94,0.2)',
    ))
    violin_fig.update_layout(**make_layout(
        title='Monthly Charges: Churned vs Retained',
        yaxis_title='Monthly Charges ($)',
        height=350,
    ))

    churned_tenure  = df[df['Churn'] == 1]['tenure']
    retained_tenure = df[df['Churn'] == 0]['tenure']
    tenure_hist = go.Figure()
    tenure_hist.add_trace(go.Histogram(
        x=churned_tenure,
        name='Churned',
        opacity=0.75,
        marker_color=CHURN_COLOR,
        nbinsx=30,
        hovertemplate='Tenure: %{x}<br>Count: %{y}<extra></extra>',
    ))
    tenure_hist.add_trace(go.Histogram(
        x=retained_tenure,
        name='Retained',
        opacity=0.55,
        marker_color=RETAIN_COLOR,
        nbinsx=30,
        hovertemplate='Tenure: %{x}<br>Count: %{y}<extra></extra>',
    ))
    tenure_hist.update_layout(**make_layout(
        title='Tenure Distribution: Churned vs Retained',
        xaxis_title='Tenure (months)',
        yaxis_title='Count',
        barmode='overlay',
        height=350,
    ))

    return seg_bar, heatmap_fig, violin_fig, tenure_hist

# ---- Performance figures ----------------------------------------------------
@app.callback(
    Output('fig-confusion', 'figure'),
    Output('fig-roc',       'figure'),
    Output('fig-pr',        'figure'),
    Output('fig-feat-imp',  'figure'),
    Input('main-tabs', 'active_tab'),
)
def update_performance(tab):
    """Populate all Model Performance figures."""
    cm = np.array(metrics['confusion_matrix'])
    labels_cm = ['Retained', 'Churned']
    cm_fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels_cm,
        y=labels_cm,
        colorscale=[[0, 'rgba(18,18,42,1)'], [1, ACCENT]],
        text=cm.astype(str),
        texttemplate='%{text}',
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        colorbar=dict(tickfont=dict(color=TEXT)),
    ))
    cm_fig.update_layout(**make_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=380,
    ))

    fpr = metrics['roc_fpr']
    tpr = metrics['roc_tpr']
    auc = metrics['auc']
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color=MUTED, width=1),
        name='Random',
        hoverinfo='skip',
    ))
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color=ACCENT, width=2),
        name=f'AUC = {auc:.4f}',
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
    ))
    roc_fig.update_layout(**make_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=380,
    ))

    prec_arr = metrics['pr_precision']
    rec_arr  = metrics['pr_recall']
    pr_fig = go.Figure(go.Scatter(
        x=rec_arr, y=prec_arr,
        mode='lines',
        line=dict(color=BLUE, width=2),
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>',
    ))
    pr_fig.update_layout(**make_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=380,
    ))

    top15 = feat_imp.head(15).sort_values()
    fi_fig = go.Figure(go.Bar(
        x=top15.values,
        y=top15.index,
        orientation='h',
        marker=dict(
            color=top15.values,
            colorscale=[[0, BLUE], [0.5, ACCENT], [1, RED]],
        ),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))
    fi_fig.update_layout(**make_layout(
        title='Top 15 Feature Importances (XGBoost)',
        xaxis_title='Importance Score',
        height=480,
        margin=dict(l=220, r=20, t=40, b=40),
    ))

    return cm_fig, roc_fig, pr_fig, fi_fig


# ---- What-If callback -------------------------------------------------------
@app.callback(
    Output('wi-gauge',        'figure'),
    Output('wi-risk-badge',   'children'),
    Output('wi-revenue-text', 'children'),
    Output('wi-shap-factors', 'children'),
    Input('wi-tenure',       'value'),
    Input('wi-monthly',      'value'),
    Input('wi-contract',     'value'),
    Input('wi-internet',     'value'),
    Input('wi-gender',       'value'),
    Input('wi-senior',       'value'),
    Input('wi-partner',      'value'),
    Input('wi-dependents',   'value'),
    Input('wi-phone',        'value'),
    Input('wi-multiline',    'value'),
    Input('wi-security',     'value'),
    Input('wi-backup',       'value'),
    Input('wi-device',       'value'),
    Input('wi-techsupport',  'value'),
    Input('wi-streamingtv',  'value'),
    Input('wi-streamingmov', 'value'),
    Input('wi-paperless',    'value'),
    Input('wi-payment',      'value'),
)
def update_whatif(
    tenure, monthly, contract, internet, gender, senior,
    partner, dependents, phone, multiline, security, backup,
    device, techsupport, streamingtv, streamingmov, paperless, payment,
):
    """Compute churn probability and SHAP for the configured customer profile."""
    total_charges = float(tenure or 0) * float(monthly or 65)
    customer = {
        'tenure':            tenure,
        'MonthlyCharges':    monthly,
        'TotalCharges':      total_charges,
        'gender':            gender,
        'SeniorCitizen':     str(senior),
        'Partner':           partner,
        'Dependents':        dependents,
        'PhoneService':      phone,
        'MultipleLines':     multiline,
        'InternetService':   internet,
        'OnlineSecurity':    security,
        'OnlineBackup':      backup,
        'DeviceProtection':  device,
        'TechSupport':       techsupport,
        'StreamingTV':       streamingtv,
        'StreamingMovies':   streamingmov,
        'Contract':          contract,
        'PaperlessBilling':  paperless,
        'PaymentMethod':     payment,
    }

    prob = predict_single(pipeline, customer)
    pct  = prob * 100

    if pct > 50:
        gauge_color = RED
        risk_label  = 'HIGH RISK'
        badge_color = RED
    elif pct > 30:
        gauge_color = YELLOW
        risk_label  = 'MEDIUM RISK'
        badge_color = YELLOW
    else:
        gauge_color = GREEN
        risk_label  = 'LOW RISK'
        badge_color = GREEN

    gauge_fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pct,
        number={'suffix': '%', 'font': {'color': gauge_color, 'size': 42}},
        title={'text': 'Churn Probability', 'font': {'color': TEXT, 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
            'bar': {'color': gauge_color},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30],  'color': 'rgba(34,197,94,0.12)'},
                {'range': [30, 50], 'color': 'rgba(245,158,11,0.12)'},
                {'range': [50, 100],'color': 'rgba(239,68,68,0.12)'},
            ],
        },
    ))
    gauge_fig.update_layout(**make_layout(height=250))

    rgb = _hex_to_rgb(badge_color)
    risk_badge = html.Span(
        risk_label,
        style={
            'backgroundColor': f'rgba({rgb},0.15)',
            'border': f'1px solid {badge_color}',
            'borderRadius': '20px',
            'padding': '4px 18px',
            'color': badge_color,
            'fontWeight': '700',
            'fontSize': '0.9rem',
            'letterSpacing': '1px',
        },
    )

    revenue_text = f'Monthly revenue at risk: ${float(monthly or 65):.2f}'

    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier   = pipeline.named_steps['classifier']
        feat_names   = preprocessor.get_feature_names_out()
        df_row       = pd.DataFrame([customer])
        for col in NUMERICAL_FEATURES:
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce')
        X_t   = preprocessor.transform(df_row[ALL_FEATURES])
        X_tdf = pd.DataFrame(X_t, columns=feat_names)
        expl  = shap.TreeExplainer(classifier)
        sv    = expl(X_tdf)
        shap_df = pd.DataFrame({
            'feature': feat_names,
            'shap':    sv.values[0],
        })
        shap_df['abs'] = shap_df['shap'].abs()
        top3 = shap_df.nlargest(3, 'abs')
        factor_items = []
        for _, row in top3.iterrows():
            direction = 'increases' if row['shap'] > 0 else 'decreases'
            color     = RED if row['shap'] > 0 else GREEN
            rgb_c     = _hex_to_rgb(color)
            factor_items.append(
                html.Div(
                    [
                        html.Span(
                            row['feature'].split('__')[-1],
                            style={'color': TEXT, 'fontWeight': '600', 'fontSize': '0.82rem'},
                        ),
                        html.Span(
                            f" {direction} churn risk  ({row['shap']:+.3f})",
                            style={'color': color, 'fontSize': '0.80rem'},
                        ),
                    ],
                    style={
                        'backgroundColor': f'rgba({rgb_c},0.08)',
                        'border': f'1px solid rgba({rgb_c},0.25)',
                        'borderRadius': '8px',
                        'padding': '8px 12px',
                        'marginBottom': '8px',
                    },
                )
            )
        shap_factors = html.Div(factor_items)
    except Exception:
        shap_factors = html.P('SHAP unavailable', style={'color': MUTED})

    return gauge_fig, risk_badge, revenue_text, shap_factors


# ---- SHAP tab figures -------------------------------------------------------
@app.callback(
    Output('fig-shap-bar',      'figure'),
    Output('fig-shap-beeswarm', 'figure'),
    Output('fig-corr-heatmap',  'figure'),
    Output('fig-shap-table',    'children'),
    Input('main-tabs', 'active_tab'),
)
def update_shap_tab(tab):
    """Populate all Feature Importance and SHAP tab figures."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_series   = pd.Series(mean_abs_shap, index=shap_feature_names).sort_values(ascending=False)
    top20         = shap_series.head(20).sort_values()

    shap_bar = go.Figure(go.Bar(
        x=top20.values,
        y=top20.index,
        orientation='h',
        marker=dict(
            color=top20.values,
            colorscale=[[0, BLUE], [0.5, ACCENT], [1, RED]],
        ),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))
    shap_bar.update_layout(**make_layout(
        title='Global SHAP Importance (mean |SHAP|, top 20)',
        xaxis_title='Mean |SHAP value|',
        height=520,
        margin=dict(l=220, r=20, t=40, b=40),
    ))

    top10_names = shap_series.head(10).index.tolist()
    beeswarm_fig = go.Figure()
    for idx, feat_name in enumerate(top10_names):
        feat_idx = list(shap_feature_names).index(feat_name)
        sv_col   = shap_values[:, feat_idx]
        xv_col   = X_shap_sample.iloc[:, feat_idx].values
        n        = min(200, len(sv_col))
        chosen   = np.random.choice(len(sv_col), n, replace=False)
        jitter   = np.random.uniform(-0.3, 0.3, n)
        beeswarm_fig.add_trace(go.Scatter(
            x=sv_col[chosen],
            y=np.full(n, idx) + jitter,
            mode='markers',
            name=feat_name.split('__')[-1],
            marker=dict(
                color=xv_col[chosen],
                colorscale=[[0, RETAIN_COLOR], [1, CHURN_COLOR]],
                size=5,
                opacity=0.7,
                showscale=(idx == 0),
                colorbar=dict(
                    title=dict(text='Feature Value', font=dict(color=TEXT, size=10)),
                    tickfont=dict(color=TEXT, size=9),
                    len=0.5,
                ) if idx == 0 else {},
            ),
            hovertemplate=f"{feat_name.split('__')[-1]}<br>SHAP: %{{x:.3f}}<extra></extra>",
        ))
    beeswarm_fig.update_layout(**make_layout(
        title='SHAP Beeswarm (top 10 features, 200 pts each)',
        xaxis_title='SHAP value',
        yaxis=dict(
            tickvals=list(range(10)),
            ticktext=[n.split('__')[-1] for n in top10_names],
            gridcolor='rgba(255,255,255,0.06)',
            color=TEXT,
        ),
        showlegend=False,
        height=520,
        margin=dict(l=180, r=60, t=40, b=40),
    ))

    num_df = df[NUMERICAL_FEATURES + ['Churn']].copy()
    corr   = num_df.corr()
    corr_fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, CHURN_COLOR], [0.5, 'rgba(18,18,42,1)'], [1, RETAIN_COLOR]],
        zmin=-1, zmax=1,
        text=[[f'{v:.2f}' for v in row] for row in corr.values],
        texttemplate='%{text}',
        hovertemplate='%{y} x %{x}: %{z:.3f}<extra></extra>',
        colorbar=dict(tickfont=dict(color=TEXT)),
    ))
    corr_fig.update_layout(**make_layout(
        title='Feature Correlation Heatmap',
        height=420,
    ))

    shap_df_all = pd.DataFrame(shap_values, columns=shap_feature_names)
    avg_shap    = shap_df_all.mean().sort_values(ascending=False).head(10)
    combined = pd.DataFrame({
        'feature':         avg_shap.index,
        'avg_shap_churn':  avg_shap.values,
    })
    combined['feature_short'] = combined['feature'].apply(lambda x: x.split('__')[-1])
    combined['recommendation'] = combined.apply(
        lambda r: 'Reduce' if r['avg_shap_churn'] > 0 else 'Leverage', axis=1
    )

    table_rows = [
        html.Tr([
            html.Th('Feature',  style={'color': MUTED, 'fontSize': '0.78rem', 'padding': '8px'}),
            html.Th('Avg SHAP', style={'color': MUTED, 'fontSize': '0.78rem', 'padding': '8px'}),
            html.Th('Action',   style={'color': MUTED, 'fontSize': '0.78rem', 'padding': '8px'}),
        ])
    ] + [
        html.Tr([
            html.Td(row['feature_short'],             style={'color': TEXT,  'fontSize': '0.8rem', 'padding': '6px 8px'}),
            html.Td(f"{row['avg_shap_churn']:+.4f}", style={'color': RED if row['avg_shap_churn'] > 0 else GREEN, 'fontSize': '0.8rem', 'padding': '6px 8px'}),
            html.Td(row['recommendation'],             style={'color': YELLOW, 'fontSize': '0.8rem', 'padding': '6px 8px'}),
        ])
        for _, row in combined.iterrows()
    ]

    shap_table = html.Div(
        [
            html.H6('Most Impactful Features for Churn Reduction', style={'color': ACCENT, 'fontWeight': '700', 'marginBottom': '12px'}),
            html.Table(
                table_rows,
                style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'backgroundColor': CARD_BG,
                    'borderRadius': '8px',
                    'overflow': 'hidden',
                },
            ),
        ],
        style={
            'backgroundColor': CARD_BG,
            'border': f'1px solid {BORDER}',
            'borderRadius': '12px',
            'padding': '16px',
            'height': '420px',
            'overflowY': 'auto',
        },
    )

    return shap_bar, beeswarm_fig, corr_fig, shap_table


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == '__main__':
    print('Customer Churn Intelligence Platform  ->  http://localhost:8052')
    app.run(debug=False, port=int(os.environ.get('PORT', 8052)), host='0.0.0.0')

