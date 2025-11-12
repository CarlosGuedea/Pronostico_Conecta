import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Desactivar el logger de tqdm para evitar conflictos
logging.getLogger('tqdm').disabled = True

# =======================================================
# Archivos de entrada y salida
# =======================================================
ARCHIVO_VENTAS_CLUSTER = "ventas_con_cluster.csv"
ARCHIVO_RESTRICCIONES = "restricciones_productos.xlsx"
SALIDA_PREDICCIONES = "predicciones_fourier_secuencial.csv"

# =======================================================
# Funciones auxiliares optimizadas
# =======================================================
def fourier_features(t, max_features=50):
    t = np.array(t, dtype=float)
    if len(t) == 0:
        return np.empty((0, 0))
    
    periods = {"annual": 52}
    n_harmonics = max_features // (2 * len(periods))
    
    if n_harmonics == 0:
        t_min, t_max = np.min(t), np.max(t)
        t_norm = (t - t_min) / (t_max - t_min + 1e-10)
        poly_features = np.vstack([t_norm, t_norm**2, t_norm**3]).T
        centers = np.linspace(0.2, 0.8, 3)
        sigmoids = 1 / (1 + np.exp(-50 * (t_norm[:, None] - centers)))
        return np.hstack([poly_features, sigmoids])
    
    t_min, t_max = np.min(t), np.max(t)
    t_norm = (t - t_min) / (t_max - t_min + 1e-10)
    
    X_fourier = []
    for period_name, period in periods.items():
        i = np.arange(1, n_harmonics + 1)
        args = 2 * np.pi * i[:, None] * t[None, :] / period
        sin_features = np.sin(args).T
        cos_features = np.cos(args).T
        X_fourier.extend([sin_features, cos_features])
    
    X_fourier = np.hstack(X_fourier) if X_fourier else np.empty((len(t), 0))
    
    poly_features = np.vstack([t_norm, t_norm**2, t_norm**3]).T
    centers = np.linspace(0.2, 0.8, 3)
    sigmoids = 1 / (1 + np.exp(-50 * (t_norm[:, None] - centers)))
    
    return np.hstack([X_fourier, poly_features, sigmoids])


def aplicar_restricciones_vectorizada(vals, producto_id, segmentacion, cluster, df_restricciones):
    vals = np.array(vals, dtype=float)
    vals_ajustados = vals.copy()
    
    if cluster == "TR TRADICIONAL" and not df_restricciones.empty:
        mask = (df_restricciones["PRODUCTO_ID_KEY"] == producto_id)
        if "SEGMENTACION" in df_restricciones.columns and segmentacion is not None:
            mask &= (df_restricciones["SEGMENTACION"] == segmentacion)
        
        restriccion_df = df_restricciones[mask]
        
        if not restriccion_df.empty:
            restriccion = restriccion_df.iloc[0]
            minimo = restriccion.get("MINIMO_VENTA", 0)
            multiplo = restriccion.get("MULTIPLO_VENTA", 1)
            
            vals_ajustados = np.maximum(vals_ajustados, minimo)
            if multiplo > 0:
                vals_ajustados = np.ceil(vals_ajustados / multiplo) * multiplo
    
    vals_ajustados = np.maximum(vals_ajustados, 0)
    return vals_ajustados


def predecir_fourier_single_task(comb, df_ventas_grouped, df_restricciones):
    try:
        sub_df = df_ventas_grouped.get_group(comb).copy()
    except KeyError:
        return []

    sub_df.sort_values("inicio_semana", inplace=True)
    
    producto_id = sub_df["PRODUCTO_ID_KEY"].iloc[0]
    segmentacion = sub_df["SEGMENTACION"].iloc[0] if "SEGMENTACION" in sub_df.columns else None
    cluster = sub_df["VAR_CAT_Cluster"].iloc[0] if "VAR_CAT_Cluster" in sub_df.columns else None
    
    # --- Fourier model ---
    n_periods = 26
    t = (sub_df["inicio_semana"] - sub_df["inicio_semana"].min()).dt.days / 7
    y = sub_df["VAR_NUM_PiezasVendidas"].values
    
    X = fourier_features(t.values, max_features=50)
    model = Ridge(alpha=50, solver='lsqr')
    model.fit(X, y)
    
    t_all = np.arange(0, int(t.iloc[-1]) + 1)
    X_all = fourier_features(t_all, max_features=50)
    y_fourier_all = model.predict(X_all)

    y_hist_final = y_fourier_all[t.astype(int)] * 0.8 + y * 0.2
    
    # --- Predicción futura ---
    t_forecast = np.arange(t.iloc[-1] + 1, t.iloc[-1] + 1 + n_periods)
    X_forecast = fourier_features(t_forecast, max_features=50)
    y_forecast_final = model.predict(X_forecast)
    
    # --- Ajuste base ---
    y_forecast_ajustado = y_forecast_final * 0.8 + y.mean() * 0.2
    
    # --- Limitar valores dentro del rango histórico ---
    max_hist_sales = y.max()
    min_limit = max_hist_sales * 0.10
    max_limit = max_hist_sales * 0.90
    y_forecast_ajustado = np.clip(y_forecast_ajustado, min_limit, max_limit)
    
    # --- Aplicar restricciones de producto ---
    y_forecast_ajustado = aplicar_restricciones_vectorizada(
        y_forecast_ajustado, producto_id, segmentacion, cluster, df_restricciones
    )
    
    # --- Fechas de salida ---
    fechas_all = pd.date_range(
        start=sub_df["inicio_semana"].min(), 
        periods=len(t_all), 
        freq="W-MON"
    )
    fechas_fut = pd.date_range(
        start=sub_df["inicio_semana"].max() + pd.Timedelta(weeks=1), 
        periods=n_periods, 
        freq="W-MON"
    )
    
    cliente_id = sub_df["CLIENTE_ID_KEY"].iloc[0]
    
    df_hist = pd.DataFrame({
        "ds": fechas_all[t.astype(int)], 
        "y_pred": y_hist_final,
        "CLIENTE_ID_KEY": cliente_id,
        "PRODUCTO_ID_KEY": producto_id
    })
    
    df_fut = pd.DataFrame({
        "ds": fechas_fut, 
        "y_pred": y_forecast_ajustado,
        "CLIENTE_ID_KEY": cliente_id,
        "PRODUCTO_ID_KEY": producto_id,
        "SEGMENTACION": segmentacion
    })
    
    return [df_hist, df_fut]


# =======================================================
# Función Principal
# =======================================================
def main():
    print("Iniciando el proceso de predicción de ventas...")
    sys.stdout.flush() 
    
    df_ventas = pd.read_csv(
        ARCHIVO_VENTAS_CLUSTER, 
        parse_dates=["fecha_mes", "inicio_semana"],
        usecols=['CLIENTE_ID_KEY', 'PRODUCTO_ID_KEY', 'VAR_NUM_PiezasVendidas', 
                'fecha_mes', 'inicio_semana', 'VAR_CAT_Cluster', 'SEGMENTACION']
    )
    
    df_restricciones = pd.read_excel(
        ARCHIVO_RESTRICCIONES,
        usecols=['PRODUCTO_ID_KEY', 'SEGMENTACION', 'MINIMO_VENTA', 'MULTIPLO_VENTA']
    )
    df_restricciones["MINIMO_VENTA"] = pd.to_numeric(df_restricciones.get("MINIMO_VENTA", pd.Series()), errors='coerce').fillna(0)
    df_restricciones["MULTIPLO_VENTA"] = pd.to_numeric(df_restricciones.get("MULTIPLO_VENTA", pd.Series()), errors='coerce').fillna(1)
    
    print("Calculando probabilidades de venta...")
    sys.stdout.flush() 

    ultimo_mes = df_ventas['fecha_mes'].max()
    dos_meses_atras = ultimo_mes - pd.DateOffset(months=2)
    un_ano_atras = ultimo_mes - pd.DateOffset(years=1)
    fecha_corte_objetivo = ultimo_mes - pd.DateOffset(months=3)

    mask_2m = df_ventas['fecha_mes'] >= dos_meses_atras
    mask_1y = df_ventas['fecha_mes'] >= un_ano_atras
    mask_futuro = df_ventas['fecha_mes'] >= fecha_corte_objetivo

    ventas_prob = df_ventas.groupby(['CLIENTE_ID_KEY','PRODUCTO_ID_KEY']).agg(
        ventas_totales=('VAR_NUM_PiezasVendidas', 'sum'),
        meses_con_ventas=('fecha_mes', lambda x: x[mask_1y].nunique()),
        ventas_2m=('VAR_NUM_PiezasVendidas', lambda x: x[mask_2m].sum())
    ).reset_index()

    ventas_prob = ventas_prob[ventas_prob['ventas_totales'] > 150]
    ventas_prob['ventas_2m'] = ventas_prob['ventas_2m'].fillna(0)
    ventas_prob['meses_con_ventas'] = ventas_prob['meses_con_ventas'].fillna(0).astype(int)

    ventas_futuras = (df_ventas[mask_futuro]
                     .groupby(['CLIENTE_ID_KEY', 'PRODUCTO_ID_KEY'])
                     .size()
                     .reset_index(name='ventas_futuras'))
    
    ventas_prob = ventas_prob.merge(
        ventas_futuras, 
        on=['CLIENTE_ID_KEY', 'PRODUCTO_ID_KEY'], 
        how='left'
    )
    ventas_prob['ventas_futuras'] = ventas_prob['ventas_futuras'].fillna(0)
    ventas_prob['tiene_venta_futura'] = (ventas_prob['ventas_futuras'] > 0).astype(int)

    features = ['ventas_totales', 'ventas_2m', 'meses_con_ventas']
    X = ventas_prob[features].values
    y = ventas_prob['tiene_venta_futura'].values

    model_nb = GaussianNB()
    model_nb.fit(X, y)
    ventas_prob['probabilidad_venta'] = model_nb.predict_proba(X)[:, 1]

    umbral_nb = 0.5
    combinaciones_filtradas = ventas_prob[ventas_prob['probabilidad_venta'] >= umbral_nb].copy()

    # =======================================================
    # Limitar número de combinaciones por cliente según SEGMENTACION
    # =======================================================
    max_combinaciones = {
        "chico": 17,
        "mediano": 27,
        "grande": 36,
        "extragrande": 45
    }
    max_default = 9

    tamanos = (
        df_ventas.groupby("CLIENTE_ID_KEY")["SEGMENTACION"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "desconocido")
        .to_dict()
    )

    combinaciones_filtradas["SEGMENTACION"] = combinaciones_filtradas["CLIENTE_ID_KEY"].map(tamanos)
    combinaciones_filtradas["MAX_COMB"] = combinaciones_filtradas["SEGMENTACION"].str.lower().map(
        max_combinaciones
    ).fillna(max_default).astype(int)

    combinaciones_filtradas.sort_values(["CLIENTE_ID_KEY", "probabilidad_venta"], ascending=[True, False], inplace=True)

    combinaciones_filtradas = (
        combinaciones_filtradas.groupby("CLIENTE_ID_KEY", group_keys=False)
        .apply(lambda x: x.head(int(x["MAX_COMB"].iloc[0]))).reset_index(drop=True)
    )

    print(f"Total de combinaciones después del filtrado por tamaño (SEGMENTACION): {len(combinaciones_filtradas)}")

    df_ventas["comb"] = df_ventas["CLIENTE_ID_KEY"].astype(str) + "_" + df_ventas["PRODUCTO_ID_KEY"].astype(str)
    combinaciones_filtradas["comb"] = combinaciones_filtradas["CLIENTE_ID_KEY"].astype(str) + "_" + combinaciones_filtradas["PRODUCTO_ID_KEY"].astype(str)
    
    print(f"Total de combinaciones a predecir: {len(combinaciones_filtradas)}")
    sys.stdout.flush() 

    df_ventas_grouped = df_ventas.groupby("comb", observed=False)

    resultados = []
    comb_list = combinaciones_filtradas['comb'].tolist()
    
    for comb in tqdm(comb_list, desc="Prediciendo combinaciones", file=sys.stdout):
        res = predecir_fourier_single_task(comb, df_ventas_grouped, df_restricciones)
        if res:
            resultados.extend(res)
            
    if resultados:
        df_result = pd.concat(resultados, ignore_index=True)
        df_result = df_result[["CLIENTE_ID_KEY", "PRODUCTO_ID_KEY", "ds", "y_pred", "SEGMENTACION"]]
        df_result.rename(columns={"ds": "FECHA_PREDICCION", "y_pred": "PREDICCION"}, inplace=True)
        
        df_result['PREDICCION'] = np.maximum(0, df_result['PREDICCION'])
        df_result['PREDICCION'] = np.round(df_result['PREDICCION'])

        # =======================================================
        # 🔧 AJUSTE POR TOTAL SEMANAL POR CLIENTE
        # =======================================================
        print("\nAplicando ajuste por total semanal por cliente...")
        limites_segmentacion = {
            "micro": 74 * 1.3,
            "chico": 157 * 1.3,
            "mediano": 325 * 1.3,
            "grande": 513 * 1.3,
            "extragrande": 995 * 1.3
        }

        ajustados = 0
        for cid, sub in df_result.groupby('CLIENTE_ID_KEY'):
            seg = str(sub['SEGMENTACION'].iloc[0]).lower() if pd.notna(sub['SEGMENTACION'].iloc[0]) else 'micro'
            limite = limites_segmentacion.get(seg, 200)
            promedio_total = sub.groupby('FECHA_PREDICCION')['PREDICCION'].sum().mean()

            if promedio_total > limite:
                factor = limite / promedio_total
                df_result.loc[sub.index, 'PREDICCION'] = np.round(sub['PREDICCION'] * factor)
                ajustados += 1

        print(f"Clientes ajustados: {ajustados}")

        # =======================================================
        # Ajuste global del 80% del total histórico
        # =======================================================
        total_real = df_ventas['VAR_NUM_PiezasVendidas'].sum()
        objetivo = int(total_real * 0.8)
        total_predicho = df_result['PREDICCION'].sum()

        if total_predicho > 0:
            factor_ajuste = objetivo / total_predicho
            df_result['PREDICCION'] = np.round(df_result['PREDICCION'] * factor_ajuste)

        print(f"\nTotal real: {total_real:,}")
        print(f"Total predicho (antes ajuste): {total_predicho:,}")
        print(f"Objetivo (80% real): {objetivo:,}")
        print(f"Total predicho (después ajuste): {df_result['PREDICCION'].sum():,}")

        df_result.to_csv(SALIDA_PREDICCIONES, index=False)
        print(f"\nPredicciones guardadas en {SALIDA_PREDICCIONES}")
    else:
        print("No se generaron predicciones para las combinaciones filtradas.")


if __name__ == "__main__":
    main()
