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
    
    # Precalcular arrays para Fourier
    X_fourier = []
    for period_name, period in periods.items():
        i = np.arange(1, n_harmonics + 1)
        args = 2 * np.pi * i[:, None] * t[None, :] / period
        sin_features = np.sin(args).T
        cos_features = np.cos(args).T
        X_fourier.extend([sin_features, cos_features])
    
    X_fourier = np.hstack(X_fourier) if X_fourier else np.empty((len(t), 0))
    
    # Características polinomiales y sigmoidales
    poly_features = np.vstack([t_norm, t_norm**2, t_norm**3]).T
    centers = np.linspace(0.2, 0.8, 3)
    sigmoids = 1 / (1 + np.exp(-50 * (t_norm[:, None] - centers)))
    
    return np.hstack([X_fourier, poly_features, sigmoids])

def aplicar_restricciones_vectorizada(vals, producto_id, segmentacion, cluster, df_restricciones):
    vals = np.array(vals, dtype=float)
    vals_ajustados = vals.copy()
    
    if cluster == "TR TRADICIONAL" and not df_restricciones.empty:
        # Optimizar la búsqueda de restricciones
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

# =======================================================
# Función de predicción optimizada
# =======================================================
def predecir_fourier_single_task(comb, df_ventas_grouped, df_restricciones):
    try:
        sub_df = df_ventas_grouped.get_group(comb).copy()
    except KeyError:
        return []

    sub_df.sort_values("inicio_semana", inplace=True)
    
    producto_id = sub_df["PRODUCTO_ID_KEY"].iloc[0]
    segmentacion = sub_df["SEGMENTACION"].iloc[0] if "SEGMENTACION" in sub_df.columns else None
    cluster = sub_df["VAR_CAT_Cluster"].iloc[0] if "VAR_CAT_Cluster" in sub_df.columns else None
    
    n_periods = 26
    
    t = (sub_df["inicio_semana"] - sub_df["inicio_semana"].min()).dt.days / 7
    y = sub_df["VAR_NUM_PiezasVendidas"].values
    
    X = fourier_features(t.values, max_features=50)
    model = Ridge(alpha=50, solver='lsqr')  # solver más rápido para problemas no muy grandes
    model.fit(X, y)
    
    t_all = np.arange(0, int(t.iloc[-1]) + 1)
    X_all = fourier_features(t_all, max_features=50)
    y_fourier_all = model.predict(X_all)

    y_hist_final = y_fourier_all[t.astype(int)] * 0.8 + y * 0.2
    
    t_forecast = np.arange(t.iloc[-1] + 1, t.iloc[-1] + 1 + n_periods)
    X_forecast = fourier_features(t_forecast, max_features=50)
    y_forecast_final = model.predict(X_forecast)

    y_forecast_ajustado = y_forecast_final * 0.8 + y.mean() * 0.2
    
    max_hist_sales = y.max()
    min_limit = max_hist_sales * 0.10
    max_limit = max_hist_sales * 0.90
    y_forecast_ajustado = np.clip(y_forecast_ajustado, min_limit, max_limit)

    y_forecast_ajustado = aplicar_restricciones_vectorizada(
        y_forecast_ajustado, producto_id, segmentacion, cluster, df_restricciones
    )
    
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
        "PRODUCTO_ID_KEY": producto_id
    })
    
    return [df_hist, df_fut]

# =======================================================
# Función Main optimizada
# =======================================================
def main():
    print("Iniciando el proceso de predicción de ventas...")
    sys.stdout.flush() 
    
    # Cargar datos optimizado
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
    df_restricciones["MINIMO_VENTA"] = pd.to_numeric(
        df_restricciones.get("MINIMO_VENTA", pd.Series()), errors='coerce'
    ).fillna(0)
    df_restricciones["MULTIPLO_VENTA"] = pd.to_numeric(
        df_restricciones.get("MULTIPLO_VENTA", pd.Series()), errors='coerce'
    ).fillna(1)
    
    print("Calculando probabilidades de venta...")
    sys.stdout.flush() 

    # Optimización: Precalcular fechas límite
    ultimo_mes = df_ventas['fecha_mes'].max()
    dos_meses_atras = ultimo_mes - pd.DateOffset(months=2)
    un_ano_atras = ultimo_mes - pd.DateOffset(years=1)
    fecha_corte_objetivo = ultimo_mes - pd.DateOffset(months=3)

    # Crear máscaras booleanas para filtrado eficiente
    mask_2m = df_ventas['fecha_mes'] >= dos_meses_atras
    mask_1y = df_ventas['fecha_mes'] >= un_ano_atras
    mask_futuro = df_ventas['fecha_mes'] >= fecha_corte_objetivo

    # Agregación optimizada usando métodos built-in de pandas
    ventas_prob = df_ventas.groupby(['CLIENTE_ID_KEY','PRODUCTO_ID_KEY']).agg(
        ventas_totales=('VAR_NUM_PiezasVendidas', 'sum'),
        meses_con_ventas=('fecha_mes', lambda x: x[mask_1y].nunique()),
        ventas_2m=('VAR_NUM_PiezasVendidas', lambda x: x[mask_2m].sum())
    ).reset_index()

    # Filtrar y procesar más eficientemente
    ventas_prob = ventas_prob[ventas_prob['ventas_totales'] > 150]
    ventas_prob['ventas_2m'] = ventas_prob['ventas_2m'].fillna(0)
    ventas_prob['meses_con_ventas'] = ventas_prob['meses_con_ventas'].fillna(0).astype(int)

    # Calcular ventas futuras de manera más eficiente
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

    # =======================================================
    # Entrenar el modelo de Naive Bayes optimizado
    # =======================================================
    features = ['ventas_totales', 'ventas_2m', 'meses_con_ventas']
    X = ventas_prob[features].values  # Usar arrays numpy para mayor velocidad
    y = ventas_prob['tiene_venta_futura'].values

    # Entrenar modelo Naive Bayes
    model_nb = GaussianNB()
    model_nb.fit(X, y)

    # Predecir probabilidades
    ventas_prob['probabilidad_venta'] = model_nb.predict_proba(X)[:, 1]

    # Filtrar combinaciones
    umbral_nb = 0.5
    combinaciones_filtradas = ventas_prob[ventas_prob['probabilidad_venta'] >= umbral_nb].copy()
    
    if combinaciones_filtradas.empty:
        print("No se encontraron combinaciones con alta probabilidad de venta. No se generarán predicciones.")
        return

    # Crear combinaciones de manera más eficiente
    df_ventas["comb"] = (df_ventas["CLIENTE_ID_KEY"].astype(str) + "_" + 
                        df_ventas["PRODUCTO_ID_KEY"].astype(str))
    
    combinaciones_filtradas["comb"] = (combinaciones_filtradas["CLIENTE_ID_KEY"].astype(str) + "_" + 
                                     combinaciones_filtradas["PRODUCTO_ID_KEY"].astype(str))
    
    print(f"Total de combinaciones a predecir: {len(combinaciones_filtradas)}")
    sys.stdout.flush() 

    # Precomputar grupos
    df_ventas_grouped = df_ventas.groupby("comb", observed=False)

    resultados = []
    comb_list = combinaciones_filtradas['comb'].tolist()
    
    # Usar list comprehension para mayor velocidad
    for comb in tqdm(comb_list, desc="Prediciendo combinaciones", file=sys.stdout):
        res = predecir_fourier_single_task(comb, df_ventas_grouped, df_restricciones)
        if res:
            resultados.extend(res)
            
    if resultados:
        df_result = pd.concat(resultados, ignore_index=True)
        df_result = df_result[["CLIENTE_ID_KEY", "PRODUCTO_ID_KEY", "ds", "y_pred"]]
        df_result.rename(columns={"ds": "FECHA_PREDICCION", "y_pred": "PREDICCION"}, inplace=True)
        
        # Ajustes finales
        df_result['PREDICCION'] = np.maximum(0, df_result['PREDICCION'])
        df_result['PREDICCION'] = np.round(df_result['PREDICCION'])

        # Ajuste global al 80% del total real
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