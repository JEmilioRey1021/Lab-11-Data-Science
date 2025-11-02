# app.py
# Dashboard interactivo con Plotly Dash para explorar Consumo e Importación de combustibles
# Requisitos: pip install dash plotly pandas scikit-learn dash-bootstrap-components

import pandas as pd
import numpy as np
from pathlib import Path

from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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

GRAPH_H = "420px"  # altura estable para todos los gráficos

def style_card(children, flex=1, _id=None):
    return html.Div(
        children,
        id=_id,
        style={
            "flex": flex,
            "background": "#FFFFFF",
            "borderRadius": "16px",
            "boxShadow": "0 8px 20px rgba(0,0,0,0.06)",
            "padding": "14px",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden"
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

# Encabezado con ícono ℹ y tooltip
def card_header(title: str, tip_id: str, tip_text: str):
    return html.Div(
        [
            html.Span(title, style={"fontWeight": 600, "color": PALETTE["ink"]}),
            html.Span(
                "ℹ",
                id=tip_id,
                style={
                    "cursor": "help",
                    "fontWeight": 700,
                    "padding": "0 6px",
                    "borderRadius": "999px",
                    "background": PALETTE["mint"],
                    "color": PALETTE["ink"],
                    "marginLeft": "8px",
                },
            ),
            dbc.Tooltip(tip_text, target=tip_id, placement="bottom", style={"fontSize": "12px"}),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "4px", "marginBottom": "6px"},
    )

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
MODELOS = ["Lineal", "Polinómico (g2)", "Random Forest"]

# ----------------------------
# Helpers de datos y modelos
# ----------------------------
def filter_df(fuente, combustible, rng_vals, month_click=None):
    i0, i1 = rng_vals
    d = full[(full["Fuente"] == fuente) &
             (full["Combustible"] == combustible) &
             (full["idx"].between(i0, i1))].sort_values("Fecha")
    if month_click:
        d = d[d["MesNombre"] == month_click]
    return d

def agg_series(df, level="M"):
    # Devuelve serie agregada por nivel: M, Q, A (mensual, trimestral, anual)
    s = df.set_index("Fecha")["Barriles"]
    s = s.resample(level).sum()
    return s.reset_index().rename(columns={"Barriles": "Valor"})

def entrenar_y_predecir(combustible, i_min_date, i_max_date, modelos_seleccion):
    col_c = f"{combustible}_Consumo"
    col_i = f"{combustible}_Importación"
    m = merged_wide[["Fecha", col_c, col_i]].dropna().copy()
    m = m[(m["Fecha"] >= i_min_date) & (m["Fecha"] <= i_max_date)]
    X = m[[col_i]].values
    y = m[col_c].values

    preds = {}
    metrics = []

    if "Lineal" in modelos_seleccion:
        lr = LinearRegression().fit(X, y)
        yhat = lr.predict(X)
        preds["Lineal"] = yhat
        metrics.append(("Lineal", mean_absolute_error(y, yhat), r2_score(y, yhat)))

    if "Polinómico (g2)" in modelos_seleccion:
        poly = PolynomialFeatures(degree=2)
        Xp = poly.fit_transform(X)
        lr2 = LinearRegression().fit(Xp, y)
        yhat = lr2.predict(Xp)
        preds["Polinómico (g2)"] = yhat
        metrics.append(("Polinómico (g2)", mean_absolute_error(y, yhat), r2_score(y, yhat)))

    if "Random Forest" in modelos_seleccion:
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        rf.fit(X, y)
        yhat = rf.predict(X)
        preds["Random Forest"] = yhat
        metrics.append(("Random Forest", mean_absolute_error(y, yhat), r2_score(y, yhat)))

    return m[["Fecha"]].assign(y_real=y), preds, metrics

def make_empty_fig(title=""):
    fig = go.Figure()
    fig = apply_pastel_layout(fig, title or "")
    return fig

# ----------------------------
# App
# ----------------------------
app = Dash(
    __name__,
    title="Combustibles – Consumo e Importación",
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
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
            "maxWidth": "1200px", "margin": "0 auto 12px", "display": "flex",
            "gap": "14px", "alignItems": "flex-end"
        }),

        # Controles extra
        html.Div([
            html.Div([
                html.Label("Agregación", style={"fontWeight": 600, "color": PALETTE["ink"]}),
                dcc.RadioItems(
                    id="agg",
                    options=[
                        {"label": " Mensual", "value": "M"},
                        {"label": " Trimestral", "value": "Q"},
                        {"label": " Anual", "value": "A"},
                    ],
                    value="M", inline=True
                )
            ], style={"flex": 1}),
            html.Div([
                html.Label("Modelos a comparar", style={"fontWeight": 600, "color": PALETTE["ink"]}),
                dcc.Checklist(
                    id="model_sel",
                    options=[{"label": m, "value": m} for m in MODELOS],
                    value=["Lineal", "Polinómico (g2)", "Random Forest"], inline=True
                )
            ], style={"flex": 3}),
            html.Div([
                html.Label("Gráficas visibles", style={"fontWeight": 600, "color": PALETTE["ink"]}),
                dcc.Checklist(
                    id="viz_sel",
                    options=[
                        {"label": " Serie", "value": "ts"},
                        {"label": " MA12", "value": "ma"},
                        {"label": " Scatter", "value": "sc"},
                        {"label": " Box", "value": "box"},
                        {"label": " Importación mensual", "value": "imp"},
                        {"label": " Pastel consumo", "value": "pie"},
                        {"label": " Predicciones", "value": "pred"},
                        {"label": " Desempeño", "value": "met"},
                        {"label": " Tabla métricas", "value": "tbl"},
                    ],
                    value=["ts","ma","sc","box","imp","pie","pred","met","tbl"], inline=True
                )
            ], style={"flex": 6, "overflowX": "auto", "whiteSpace": "nowrap"})
        ], style={"maxWidth": "1200px", "margin": "0 auto 16px", "display": "flex", "gap": "14px"}),

        # Fila 1
        html.Div([
            style_card([
                card_header("Serie temporal", "tip-ts",
                            "Evolución de barriles en el tiempo. Usa zoom/arrastre y haz clic en un mes del boxplot para filtrar."),
                dcc.Graph(id="g_ts", style={"height": GRAPH_H}),
            ], flex=1, _id="card_ts"),

            style_card([
                card_header("Tendencia (media móvil 12m)", "tip-ma",
                            "Suaviza fluctuaciones con una ventana de 12 meses."),
                dcc.Graph(id="g_ma", style={"height": GRAPH_H}),
            ], flex=1, _id="card_ma"),
        ], style={"maxWidth": "1200px", "margin": "0 auto 14px", "display": "flex", "gap": "14px"}),

        # Fila 2
        html.Div([
            style_card([
                card_header("Relación Consumo vs Importación", "tip-sc",
                            "Dispersión con línea de correlación."),
                dcc.Graph(id="g_scatter_ci", style={"height": GRAPH_H}),
            ], flex=1, _id="card_sc"),

            style_card([
                card_header("Distribución por mes (boxplot)", "tip-box",
                            "Haz clic en un mes para explorar su comportamiento en el periodo seleccionado."),
                dcc.Graph(id="g_box", style={"height": GRAPH_H}),
            ], flex=1, _id="card_box"),
        ], style={"maxWidth": "1200px", "margin": "0 auto", "display": "flex", "gap": "14px"}),

        # Fila 3 – Importación mensual y Distribución del consumo
        html.Div([
            style_card([
                card_header("Importación promedio por mes", "tip-imp",
                            "Promedio mensual dentro del rango; compara combustibles por mes."),
                dcc.Graph(id="g_import_bar", style={"height": GRAPH_H}),
            ], flex=1, _id="card_imp"),

            style_card([
                card_header("Distribución del consumo por combustible", "tip-pie",
                            "Participación relativa de cada combustible en el periodo seleccionado."),
                dcc.Graph(id="g_pie_consumo", style={"height": GRAPH_H}),
            ], flex=1, _id="card_pie"),
        ], style={"maxWidth": "1200px", "margin": "14px auto 0", "display": "flex", "gap": "14px"}),

        # Fila 4 – NUEVO: Predicciones y Métricas
        html.Div([
            style_card([
                card_header("Predicciones (3 modelos)", "tip-pred",
                            "Valores reales vs. predicciones de los modelos seleccionados. Usa la agregación para cambiar el nivel de detalle."),
                dcc.Graph(id="g_pred", style={"height": GRAPH_H}),
            ], flex=1, _id="card_pred"),

            style_card([
                card_header("Desempeño por modelo", "tip-met",
                            "Comparación cuantitativa (MAE y R²) en el rango seleccionado."),
                dcc.Graph(id="g_model_metrics", style={"height": GRAPH_H}),
            ], flex=1, _id="card_met"),
        ], style={"maxWidth": "1200px", "margin": "14px auto 0", "display": "flex", "gap": "14px"}),

        # Fila 5 – Tabla comparativa
        style_card([
            card_header("Tabla comparativa de métricas", "tip-tbl",
                        "Filtra modelos en el control 'Modelos a comparar'."),
            dash_table.DataTable(
                id="tbl_metrics",
                columns=[{"name": c, "id": c} for c in ["Modelo", "MAE", "R2"]],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "border": "none"},
                style_header={"fontWeight": 700},
            )
        ], flex=1, _id="card_tbl",),
    ]
)

# ----------------------------
# Callback principal
# ----------------------------
@app.callback(
    Output("g_ts", "figure"),
    Output("g_ma", "figure"),
    Output("g_scatter_ci", "figure"),
    Output("g_box", "figure"),
    Output("g_import_bar", "figure"),
    Output("g_pie_consumo", "figure"),
    Output("g_pred", "figure"),
    Output("g_model_metrics", "figure"),
    Output("tbl_metrics", "data"),
    # visibilidad de tarjetas
    Output("card_ts", "style"),
    Output("card_ma", "style"),
    Output("card_sc", "style"),
    Output("card_box", "style"),
    Output("card_imp", "style"),
    Output("card_pie", "style"),
    Output("card_pred", "style"),
    Output("card_met", "style"),
    Output("card_tbl", "style"),
    # Inputs
    Input("fnt", "value"),
    Input("cmb", "value"),
    Input("rng", "value"),
    Input("agg", "value"),
    Input("model_sel", "value"),
    Input("viz_sel", "value"),
    Input("g_box", "clickData"),
)
def update_plots(fuente, combustible, rng_vals, agg_level, modelos_sel, viz_sel, box_click):
    # filtro por mes si hay click en boxplot
    month_click = None
    if box_click and "points" in box_click and len(box_click["points"]) > 0:
        month_click = box_click["points"][0].get("x")

    d = filter_df(fuente, combustible, rng_vals, month_click=month_click)

    # --- Serie temporal (con agregación)
    s_agg = agg_series(d[["Fecha", "Barriles"]].rename(columns={"Barriles": "Barriles"}), level=agg_level)
    fig_ts = px.line(s_agg, x="Fecha", y="Valor", title=f"{combustible} — {fuente}")
    fig_ts.update_traces(mode="lines+markers", line=dict(color=PALETTE["sky"], width=3),
                         marker=dict(size=5, opacity=0.85))
    fig_ts.update_yaxes(title="Barriles")
    fig_ts = apply_pastel_layout(fig_ts)

    # --- Tendencia (MA 12m) sobre datos mensuales (no agregado trimestral/anual)
    d2 = d.copy().sort_values("Fecha")
    d2["MA12"] = d2["Barriles"].rolling(12, min_periods=1).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=d2["Fecha"], y=d2["Barriles"], name="Serie",
                                mode="lines", line=dict(color=PALETTE["sky"], width=2), opacity=0.6))
    fig_ma.add_trace(go.Scatter(x=d2["Fecha"], y=d2["MA12"], name="Media móvil 12m",
                                mode="lines", line=dict(color=PALETTE["coral"], width=4)))
    fig_ma = apply_pastel_layout(fig_ma, "Tendencia (media móvil 12m)")

    # --- Dispersión Consumo vs Importación
    col_c = f"{combustible}_Consumo"
    col_i = f"{combustible}_Importación"
    m = merged_wide[["Fecha", col_c, col_i]].dropna().copy()
    i0, i1 = rng_vals
    i_min_date = idx_to_date.get(i0, d["Fecha"].min())
    i_max_date = idx_to_date.get(i1, d["Fecha"].max())
    m = m[(m["Fecha"] >= i_min_date) & (m["Fecha"] <= i_max_date)]

    # Línea de correlación (lineal simple)
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

    # === Importación promedio por mes (agrupado por combustible)
    imp = importacion_long[(importacion_long["Fecha"] >= i_min_date) &
                           (importacion_long["Fecha"] <= i_max_date)].copy()
    imp_grp = (imp.groupby(["Mes", "Combustible"], as_index=False)["Barriles"]
                  .mean()
                  .sort_values("Mes"))
    meses_lbl = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    fig_import = px.bar(
        imp_grp, x="Mes", y="Barriles", color="Combustible", barmode="group",
        title="Importación promedio por mes (periodo seleccionado)"
    )
    fig_import.update_xaxes(tickmode="array", tickvals=list(range(1,13)), ticktext=meses_lbl)
    fig_import = apply_pastel_layout(fig_import)

    # === Distribución del consumo por combustible (pastel)
    cons = consumo_long[(consumo_long["Fecha"] >= i_min_date) &
                        (consumo_long["Fecha"] <= i_max_date)].copy()
    cons_grp = cons.groupby("Combustible", as_index=False)["Barriles"].sum()
    fig_pie = px.pie(
        cons_grp, names="Combustible", values="Barriles",
        title="Distribución del consumo por combustible", hole=0.35
    )
    fig_pie.update_traces(textposition="outside")
    fig_pie = apply_pastel_layout(fig_pie)

    # === NUEVO: Predicciones (3 modelos) ===
    # Entrenar y obtener predicciones sobre el subrango visible
    serie_real, preds, metrics = entrenar_y_predecir(combustible, i_min_date, i_max_date, modelos_sel)
    fig_pred = go.Figure()
    # serie real (agregación elegida)
    real_df = serie_real.rename(columns={"y_real": "Barriles"})
    real_df = real_df.rename(columns={"Fecha": "Fecha"})
    real_df = real_df.sort_values("Fecha")
    real_agg = real_df.set_index("Fecha")["Barriles"].resample(agg_level).sum().reset_index()
    fig_pred.add_trace(go.Scatter(x=real_agg["Fecha"], y=real_agg["Barriles"], name="Real",
                                  mode="lines+markers",
                                  line=dict(color=PALETTE["sky"], width=3)))

    # añadir cada predicción con la misma agregación
    for name, yhat in preds.items():
        p = pd.DataFrame({"Fecha": serie_real["Fecha"].values, "yhat": yhat})
        p_agg = p.set_index("Fecha")["yhat"].resample(agg_level).sum().reset_index()
        fig_pred.add_trace(go.Scatter(x=p_agg["Fecha"], y=p_agg["yhat"], name=name,
                                      mode="lines",
                                      line=dict(width=4)))
    fig_pred.update_yaxes(title="Barriles")
    fig_pred = apply_pastel_layout(fig_pred, "Predicciones (modelos seleccionados)")

    # === NUEVO: Gráfico de desempeño + Tabla ===
    if metrics:
        met_df = pd.DataFrame(metrics, columns=["Modelo", "MAE", "R2"]).sort_values("MAE")
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(x=met_df["Modelo"], y=met_df["MAE"], name="MAE"))
        fig_metrics.add_trace(go.Bar(x=met_df["Modelo"], y=met_df["R2"], name="R²"))
        fig_metrics.update_layout(barmode="group")
        fig_metrics = apply_pastel_layout(fig_metrics, "Desempeño por modelo (menor MAE es mejor)")
        table_data = met_df.round({"MAE": 2, "R2": 3}).to_dict("records")
    else:
        fig_metrics = make_empty_fig("Desempeño por modelo")
        table_data = []

    # === Visibilidad de tarjetas según checklist ===
    def style_for(key):
        return {
            "flex": 1,
            "background": "#FFFFFF",
            "borderRadius": "16px",
            "boxShadow": "0 8px 20px rgba(0,0,0,0.06)",
            "padding": "14px",
            "display": "flex" if key in viz_sel else "none",
            "flexDirection": "column",
            "overflow": "hidden"
        }

    return (
        fig_ts, fig_ma, fig_scatter, fig_box, fig_import, fig_pie,
        fig_pred, fig_metrics, table_data,
        style_for("ts"), style_for("ma"), style_for("sc"), style_for("box"),
        style_for("imp"), style_for("pie"), style_for("pred"), style_for("met"), style_for("tbl")
    )

if __name__ == "__main__":
    app.run(debug=True)
