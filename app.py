# app.py
# Dashboard interactivo con Plotly Dash para explorar Consumo e Importación de combustibles
# Requisitos: pip install dash plotly pandas
# (Opcional para estilos): pip install dash-bootstrap-components

import pandas as pd
from pathlib import Path
from datetime import datetime

from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Carga de datos
# ----------------------------
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path('.')
consumo_path = BASE_DIR / "Consumo.xlsx"
importacion_path = BASE_DIR / "Importacion.xlsx"

consumo = pd.read_excel(consumo_path, parse_dates=["Fecha"]).assign(Fuente="Consumo")
importacion = pd.read_excel(importacion_path, parse_dates=["Fecha"]).assign(Fuente="Importación")

# Unificamos el esquema a formato largo para facilitar filtros y gráficas
id_vars = ["Fecha", "Fuente"]
value_vars = ["Gasolina regular", "Gasolina superior", "Diesel alto azufre"]

def to_long(df):
    long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Combustible", value_name="Barriles")
    long["Año"] = long["Fecha"].dt.year
    long["Mes"] = long["Fecha"].dt.month
    long["MesNombre"] = long["Fecha"].dt.strftime("%b")
    long["YYYYMM"] = long["Fecha"].dt.strftime("%Y-%m")
    return long

consumo_long = to_long(consumo)
importacion_long = to_long(importacion)

full = pd.concat([consumo_long, importacion_long], ignore_index=True)
full.sort_values("Fecha", inplace=True)

# Para enlaces entre fuentes, construimos un merge en fechas (para scatter Importación vs Consumo)
consumo_wide = consumo.set_index("Fecha")[value_vars]
import_wide = importacion.set_index("Fecha")[value_vars]
merged_wide = consumo_wide.join(import_wide, how="inner", lsuffix="_Consumo", rsuffix="_Importación").reset_index()

# ----------------------------
# Utilidades de UI
# ----------------------------
FUENTES = ["Consumo", "Importación"]
COMBUSTIBLES = value_vars

min_date = full["Fecha"].min()
max_date = full["Fecha"].max()

# Para RangeSlider usamos índice entero de meses
full["idx"] = (full["Año"] - full["Año"].min()) * 12 + (full["Mes"] - 1)
idx_min, idx_max = int(full["idx"].min()), int(full["idx"].max())
idx_to_date = full.drop_duplicates("idx")[["idx", "Fecha"]].set_index("idx")["Fecha"].to_dict()

def idx_to_label(i):
    dt = idx_to_date.get(i)
    return dt.strftime("%Y-%m") if dt is not None else str(i)

# ----------------------------
# App
# ----------------------------
app = Dash(__name__, title="Combustibles – Consumo e Importación")
server = app.server

app.layout = html.Div([
    html.H3("Dashboard Interactivo – Combustibles (Consumo & Importación)"),
    html.P("Explora patrones temporales y compara Consumo e Importación por tipo de combustible."),

    # Controles
    html.Div([
        html.Div([
            html.Label("Fuente"),
            dcc.RadioItems(options=[{"label": f, "value": f} for f in FUENTES], value="Consumo", id="fnt"),
        ], style={"flex":1}),
        html.Div([
            html.Label("Combustible"),
            dcc.Dropdown(options=[{"label": c, "value": c} for c in COMBUSTIBLES], value="Gasolina regular", id="cmb"),
        ], style={"flex":2, "marginLeft":"1rem"}),
        html.Div([
            html.Label("Rango de fechas"),
            dcc.RangeSlider(min=idx_min, max=idx_max, value=[idx_min, idx_max], id="rng", allowCross=False,
                            tooltip={"placement":"bottom"},
                            marks={i: idx_to_label(i) for i in range(idx_min, idx_max+1, max(1, (idx_max-idx_min)//10))}),
        ], style={"flex":5, "marginLeft":"1rem"}),
    ], style={"display":"flex", "alignItems":"flex-end", "gap":"1rem", "marginBottom":"1rem"}),

    # Fila 1 – Serie temporal y descomposición ligera (tendencia con media móvil)
    html.Div([
        dcc.Graph(id="g_ts", style={"flex": 2}),
        dcc.Graph(id="g_ma", style={"flex": 1}),
    ], style={"display":"flex", "gap":"1rem", "marginBottom":"1rem"}),

    # Fila 2 – Estacionalidad mensual (heatmap) y caja por mes
    html.Div([
        dcc.Graph(id="g_heat", style={"flex": 1}),
        dcc.Graph(id="g_box", style={"flex": 1}),
    ], style={"display":"flex", "gap":"1rem", "marginBottom":"1rem"}),

    # Fila 3 – YoY y comparación Consumo vs Importación (enlazado por combustible)
    html.Div([
        dcc.Graph(id="g_yoy", style={"flex": 1}),
        dcc.Graph(id="g_scatter_ci", style={"flex": 1}),
    ], style={"display":"flex", "gap":"1rem", "marginBottom":"1rem"}),

    # Fila 4 – Módulo de modelos (placeholder interactividades)
    html.H4("Modelos simples de predicción"),
    html.Div([
        html.Div([
            html.Label("Modelos a comparar"),
            dcc.Checklist(
                id="mdl_list",
                options=[
                    {"label":"Naive estacional", "value":"naive_seasonal"},
                    {"label":"Media móvil (12m)", "value":"moving_avg"},
                    {"label":"Tendencia lineal + mes", "value":"lin_trend"},
                ],
                value=["naive_seasonal", "moving_avg", "lin_trend"],
                inline=True,
            ),
            html.Label("Horizonte (meses)"),
            dcc.Slider(3, 24, 1, value=12, id="h"),
        ], style={"flex":1}),
        dcc.Graph(id="g_forecast", style={"flex":2}),
    ], style={"display":"flex", "gap":"1rem", "marginBottom":"1rem"}),

    dash_table.DataTable(id="tbl_metrics", columns=[
        {"name":"Modelo", "id":"Modelo"}, {"name":"MAE", "id":"MAE"}, {"name":"RMSE", "id":"RMSE"}, {"name":"MAPE", "id":"MAPE"}
    ], data=[], style_table={"overflowX":"auto"}),

    html.Hr(),
    html.Small("Tip: seleccione un rango en la serie temporal para filtrar todas las visualizaciones (enlazado)."),
])

# ----------------------------
# Callbacks auxiliares
# ----------------------------

def filter_df(fuente, combustible, rng_vals):
    i0, i1 = rng_vals
    d = full[(full["Fuente"]==fuente) & (full["Combustible"]==combustible) & (full["idx"].between(i0, i1))]
    return d.sort_values("Fecha")

@app.callback(
    Output("g_ts", "figure"),
    Output("g_ma", "figure"),
    Output("g_heat", "figure"),
    Output("g_box", "figure"),
    Output("g_yoy", "figure"),
    Output("g_scatter_ci", "figure"),
    Input("fnt", "value"),
    Input("cmb", "value"),
    Input("rng", "value"),
)
def update_core_plots(fuente, combustible, rng_vals):
    d = filter_df(fuente, combustible, rng_vals)

    # Serie temporal principal
    fig_ts = px.line(d, x="Fecha", y="Barriles", title=f"{combustible} – {fuente}")
    fig_ts.update_traces(mode="lines+markers")

    # Media móvil 12 meses (tendencia suave)
    d2 = d.copy()
    d2["MA12"] = d2["Barriles"].rolling(12, min_periods=1).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=d2["Fecha"], y=d2["Barriles"], name="Serie", mode="lines"))
    fig_ma.add_trace(go.Scatter(x=d2["Fecha"], y=d2["MA12"], name="Media móvil 12m", mode="lines"))
    fig_ma.update_layout(title="Tendencia (media móvil 12m)")

    # Heatmap de estacionalidad mensual (promedio por Año x Mes)
    heat = d.groupby(["Año", "Mes"], as_index=False)["Barriles"].mean()
    heat_pivot = heat.pivot(index="Año", columns="Mes", values="Barriles").sort_index()
    fig_heat = px.imshow(heat_pivot, aspect="auto", origin="lower", labels=dict(color="Barriles"),
                         title="Estacionalidad (promedio Año x Mes)")

    # Caja por mes (variabilidad estacional)
    fig_box = px.box(d.assign(MesNombre=d["Fecha"].dt.strftime("%b")), x="MesNombre", y="Barriles", points="outliers",
                     title="Distribución por mes")

    # Crecimiento interanual (YoY) por mes
    d_yoy = d.set_index("Fecha").sort_index()
    d_yoy["YoY"] = d_yoy["Barriles"].pct_change(12) * 100
    d_yoy = d_yoy.reset_index()
    fig_yoy = px.bar(d_yoy.dropna(subset=["YoY"]), x="Fecha", y="YoY", title="Variación interanual (%)")

    # Scatter Consumo vs Importación (mismo combustible)
    col_c = f"{combustible}_Consumo"
    col_i = f"{combustible}_Importación"
    m = merged_wide[["Fecha", col_c, col_i]].dropna()
    # Filtrar por rango seleccionado
    i0, i1 = rng_vals
    i_min_date = idx_to_date.get(i0, d["Fecha"].min())
    i_max_date = idx_to_date.get(i1, d["Fecha"].max())
    m = m[(m["Fecha"]>=i_min_date) & (m["Fecha"]<=i_max_date)]
    fig_scatter = px.scatter(m, x=col_i, y=col_c, trendline="ols",
                             labels={col_i:"Importación (barriles)", col_c:"Consumo (barriles)"},
                             title="Relación Consumo vs Importación")

    return fig_ts, fig_ma, fig_heat, fig_box, fig_yoy, fig_scatter

# ----------------------------
# Modelos simples (baseline):
#  - naive_seasonal: pronóstico igual al valor de hace 12 meses
#  - moving_avg: promedio móvil de los últimos 12 meses
#  - lin_trend: regresión lineal sobre tiempo + dummies de mes (pronóstico por mes futuro)
# ----------------------------
from sklearn.linear_model import LinearRegression
import numpy as np


def _make_features(d):
    d = d.copy().reset_index(drop=True)
    d["t"] = range(len(d))
    d["mes"] = d["Fecha"].dt.month
    X = pd.get_dummies(d[["t", "mes"]].astype(int), columns=["mes"], drop_first=True)
    y = d["Barriles"].values
    return X, y, d


def forecast_models(d, horizon=12, models=("naive_seasonal", "moving_avg", "lin_trend")):
    d = d.sort_values("Fecha").copy()
    out = {}

    # Índices de futuro
    last_date = d["Fecha"].max()
    fut_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    if "naive_seasonal" in models:
        f_naive = d.set_index("Fecha")["Barriles"].shift(12).dropna()
        # Para futuro: usamos el último año disponible como plantilla
        hist = d.set_index("Fecha")["Barriles"]
        future_vals = []
        for dt in fut_dates:
            ref = dt - pd.DateOffset(years=1)
            future_vals.append(hist.get(ref, np.nan))
        out["naive_seasonal"] = pd.Series(future_vals, index=fut_dates)

    if "moving_avg" in models:
        ma = d["Barriles"].rolling(12, min_periods=1).mean().iloc[-1]
        out["moving_avg"] = pd.Series([ma]*horizon, index=fut_dates)

    if "lin_trend" in models:
        X, y, d_fe = _make_features(d)
        lr = LinearRegression().fit(X, y)
        # features futuras
        last_t = d_fe["t"].iloc[-1]
        fut = pd.DataFrame({
            "t": range(last_t+1, last_t+1+horizon),
            "mes": [dt.month for dt in fut_dates]
        })
        Xf = pd.get_dummies(fut.astype(int), columns=["mes"], drop_first=True)
        # Alinear columnas
        Xf = Xf.reindex(columns=X.columns, fill_value=0)
        yhat = lr.predict(Xf)
        out["lin_trend"] = pd.Series(yhat, index=fut_dates)

    return out


def backtest_mae_rmse_mape(y_true, y_pred):
    import numpy as np
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / np.where(y_true==0, np.nan, y_true))) * 100
    return mae, rmse, mape

@app.callback(
    Output("g_forecast", "figure"),
    Output("tbl_metrics", "data"),
    Input("fnt", "value"),
    Input("cmb", "value"),
    Input("rng", "value"),
    Input("mdl_list", "value"),
    Input("h", "value"),
)
def update_models(fuente, combustible, rng_vals, mdl_list, horizon):
    d = filter_df(fuente, combustible, rng_vals)
    # Entrenamiento: todo el rango seleccionado menos "horizon" meses para validación simple
    d = d.sort_values("Fecha").reset_index(drop=True)
    if len(d) < 24:  # salvaguarda
        return go.Figure(), []

    train = d.iloc[:-horizon].copy()
    test = d.iloc[-horizon:].copy()

    fcasts = forecast_models(train, horizon=horizon, models=tuple(mdl_list))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["Barriles"], name="Observado", mode="lines"))

    rows = []
    for name, series in fcasts.items():
        # Métricas contra test (alinear índices)
        pred = series.reindex(test["Fecha"]).values
        mae, rmse, mape = backtest_mae_rmse_mape(test["Barriles"].values, pred)
        rows.append({"Modelo": name, "MAE": round(float(mae),1), "RMSE": round(float(rmse),1), "MAPE": round(float(mape),2)})
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=f"{name}", mode="lines"))

    fig.update_layout(title=f"Pronósticos a {horizon} meses – {combustible} ({fuente})")
    return fig, rows


if __name__ == "__main__":
    # Ejecutar con: python app.py
    app.run(debug=True)

