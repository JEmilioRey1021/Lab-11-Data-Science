# app.py
# Dashboard interactivo con Plotly Dash para explorar Consumo e Importación de combustibles
# Requisitos: pip install dash plotly pandas scikit-learn

import pandas as pd
import numpy as np
from pathlib import Path

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ----------------------------
# Paleta y helpers de estilo
# ----------------------------
PALETTE = {
    "cream": "#F7F3DF",   # fondo
    "coral": "#ECA07D",   # acentos cálidos
    "yellow": "#F6F07A",
    "mint": "#B9EE93",
    "sky": "#9EC1E6",     # series principales
    "ink": "#1F2937",     # texto
    "muted": "#6B7280",   # texto secundario
    "grid": "rgba(31,41,55,0.08)"
}

def style_card(children, flex=1):
    return html.Div(
        children,
        style={
            "flex": flex,
            "background": "#FFFFFF",
            "borderRadius": "16px",
            "boxShadow": "0 8px 20px rgba(0,0,0,0.06)",
            "padding": "14px"
        }
    )

def apply_pastel_layout(fig, title=None):
    fig.update_layout(
        title=title or fig.layout.title.text,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=24, r=16, t=48, b=24),
        font=dict(family="Inter, Segoe UI, system-ui, -apple-system, Arial",
                  size=13, color=PALETTE["ink"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor=PALETTE["grid"],
                     gridcolor=PALETTE["grid"], zeroline=False, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1, linecolor=PALETTE["grid"],
                     gridcolor=PALETTE["grid"], zeroline=False, ticks="outside")
    return fig

# ----------------------------
# Carga de datos
# ----------------------------
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path('.')
consumo_path = BASE_DIR / "Consumo.xlsx"
importacion_path = BASE_DIR / "Importacion.xlsx"

consumo = pd.read_excel(consumo_path, parse_dates=["Fecha"]).assign(Fuente="Consumo")
importacion = pd.read_excel(importacion_path, parse_dates=["Fecha"]).assign(Fuente="Importación")

value_vars = ["Gasolina regular", "Gasolina superior", "Diesel alto azufre"]
id_vars = ["Fecha", "Fuente"]

def to_long(df):
    long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Combustible", value_name="Barriles")
    long["Año"] = long["Fecha"].dt.year
    long["Mes"] = long["Fecha"].dt.month
    long["MesNombre"] = long["Fecha"].dt.strftime("%b")
    return long

consumo_long = to_long(consumo)
importacion_long = to_long(importacion)
full = pd.concat([consumo_long, importacion_long], ignore_index=True).sort_values("Fecha")

# para scatter: unir por fecha
consumo_wide = consumo.set_index("Fecha")[value_vars]
import_wide  = importacion.set_index("Fecha")[value_vars]
merged_wide  = consumo_wide.join(import_wide, how="inner",
                                 lsuffix="_Consumo", rsuffix="_Importación").reset_index()

# slider por mes
full["idx"] = (full["Fecha"].dt.year - full["Fecha"].dt.year.min()) * 12 + (full["Fecha"].dt.month - 1)
idx_min, idx_max = int(full["idx"].min()), int(full["idx"].max())
idx_to_date = full.drop_duplicates("idx")[["idx", "Fecha"]].set_index("idx")["Fecha"].to_dict()
def idx_to_label(i):
    dt = idx_to_date.get(i)
    return dt.strftime("%Y-%m") if dt is not None else str(i)

FUENTES = ["Consumo", "Importación"]
COMBUSTIBLES = value_vars

# ----------------------------
# App
# ----------------------------
app = Dash(
    __name__,
    title="Combustibles – Consumo e Importación",
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"]
)
server = app.server


app.layout = html.Div(
    style={"background": PALETTE["cream"], "minHeight": "100vh", "padding": "24px", "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Arial"},
    children=[
        # Encabezado
        html.Div([
            html.H2("Dashboard de consumo de combustibles",
                    style={"margin": "0 0 4px 0", "color": PALETTE["ink"]}),
            html.P("Elaborado por Emilio Reyes, Michelle Mejia y Silvia Illescas.",
                   style={"margin": 0, "color": PALETTE["muted"]}),
        ], style={"maxWidth": "1200px", "margin": "0 auto 16px"}),

        # Barra de filtros
        html.Div([
            # Fuente
            html.Div([
                html.Label("Fuente", style={"fontWeight": 600, "color": PALETTE["ink"]}),
                dcc.RadioItems(
                    options=[{"label": f" {f}", "value": f} for f in FUENTES],
                    value="Consumo", id="fnt", inline=True,
                    style={
                        "background": PALETTE["sky"],
                        "padding": "8px 12px", "borderRadius": "14px"
                    }
                ),
            ], style={"flex": 1}),
            # Combustible
            html.Div([
                html.Label("Combustible", style={"fontWeight": 600, "color": PALETTE["ink"]}),
                dcc.Dropdown(
                    options=[{"label": c, "value": c} for c in COMBUSTIBLES],
                    value="Gasolina regular", id="cmb", clearable=False,
                    style={"background": PALETTE["yellow"], "borderRadius": "14px"}
                ),
            ], style={"flex": 2}),
            # Rango
            html.Div([
                html.Label("Rango de fechas", style={"fontWeight": 600, "color": PALETTE["ink"]}),
                dcc.RangeSlider(
                    min=idx_min, max=idx_max, value=[idx_min, idx_max], id="rng",
                    allowCross=False, tooltip={"placement": "bottom"},
                    marks={i: idx_to_label(i) for i in range(idx_min, idx_max + 1,
                          max(1, (idx_max - idx_min)//10))}
                ),
            ], style={"flex": 5}),
        ],
        style={
            "maxWidth": "1200px", "margin": "0 auto 20px", "display": "flex",
            "gap": "14px", "alignItems": "flex-end"
        }),

        # Contenido 2x2
        html.Div([
            style_card(dcc.Graph(id="g_ts"), flex=1),
            style_card(dcc.Graph(id="g_ma"), flex=1),
        ], style={"maxWidth": "1200px", "margin": "0 auto 14px", "display": "flex", "gap": "14px"}),

        html.Div([
            style_card(dcc.Graph(id="g_scatter_ci"), flex=1),
            style_card(dcc.Graph(id="g_box"), flex=1),
        ], style={"maxWidth": "1200px", "margin": "0 auto", "display": "flex", "gap": "14px"}),
    ]
)

# ----------------------------
# Helpers de datos
# ----------------------------
def filter_df(fuente, combustible, rng_vals):
    i0, i1 = rng_vals
    d = full[(full["Fuente"] == fuente) &
             (full["Combustible"] == combustible) &
             (full["idx"].between(i0, i1))].sort_values("Fecha")
    return d

# ----------------------------
# Callback principal (2x2)
# ----------------------------
@app.callback(
    Output("g_ts", "figure"),
    Output("g_ma", "figure"),
    Output("g_scatter_ci", "figure"),
    Output("g_box", "figure"),
    Input("fnt", "value"),
    Input("cmb", "value"),
    Input("rng", "value"),
)
def update_plots(fuente, combustible, rng_vals):
    d = filter_df(fuente, combustible, rng_vals)

    # --- Serie temporal
    fig_ts = px.line(d, x="Fecha", y="Barriles", title=f"{combustible} — {fuente}")
    fig_ts.update_traces(mode="lines+markers", line=dict(color=PALETTE["sky"], width=3),
                         marker=dict(size=5, opacity=0.85))
    fig_ts = apply_pastel_layout(fig_ts)

    # --- Tendencia (MA 12m)
    d2 = d.copy()
    d2["MA12"] = d2["Barriles"].rolling(12, min_periods=1).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=d2["Fecha"], y=d2["Barriles"], name="Serie",
                                mode="lines", line=dict(color=PALETTE["sky"], width=2), opacity=0.6))
    fig_ma.add_trace(go.Scatter(x=d2["Fecha"], y=d2["MA12"], name="Media móvil 12m",
                                mode="lines", line=dict(color=PALETTE["coral"], width=4)))
    fig_ma = apply_pastel_layout(fig_ma, "Tendencia (media móvil 12m)")

    # --- Dispersión Consumo vs Importación (misma ventana)
    col_c = f"{combustible}_Consumo"
    col_i = f"{combustible}_Importación"
    m = merged_wide[["Fecha", col_c, col_i]].dropna().copy()
    i0, i1 = rng_vals
    i_min_date = idx_to_date.get(i0, d["Fecha"].min())
    i_max_date = idx_to_date.get(i1, d["Fecha"].max())
    m = m[(m["Fecha"] >= i_min_date) & (m["Fecha"] <= i_max_date)]

    # Línea de correlación con LinearRegression (evita dependencia de statsmodels)
    X = m[[col_i]].values
    y = m[col_c].values
    if len(m) >= 2:
        lr = LinearRegression().fit(X, y)
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = lr.predict(x_line)
    else:
        x_line = np.array([0, 1]).reshape(-1, 1)
        y_line = np.array([0, 1])

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=m[col_i], y=m[col_c], mode="markers", name="Puntos",
        marker=dict(color=PALETTE["sky"], size=7, line=dict(color="#ffffff", width=0.5), opacity=0.9)
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_line.flatten(), y=y_line, mode="lines", name="Correlación lineal",
        line=dict(color=PALETTE["coral"], width=4)
    ))
    fig_scatter.update_layout(
        xaxis_title="Importación (barriles)", yaxis_title="Consumo (barriles)"
    )
    fig_scatter = apply_pastel_layout(fig_scatter, "Relación Consumo vs Importación")

    # --- Boxplot por mes
    d_box = d.assign(MesNombre=d["Fecha"].dt.strftime("%b"))
    fig_box = px.box(d_box, x="MesNombre", y="Barriles", points="outliers",
                     title="Distribución por mes")
    fig_box.update_traces(marker_color=PALETTE["coral"], line_color=PALETTE["coral"])
    fig_box = apply_pastel_layout(fig_box)

    return fig_ts, fig_ma, fig_scatter, fig_box


if __name__ == "__main__":
    app.run(debug=True)
