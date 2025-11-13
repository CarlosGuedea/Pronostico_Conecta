import pandas as pd
import numpy as np
import math
from datetime import timedelta
from tqdm import tqdm
import sys
import io

# Forzar salida UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------------
# 0. Configuración
# ----------------------------------------------------------------------
EQUIVALENCIAS_SEGMENTACION = {
    "Extragrand": "Grande",
    "Extragrande": "Grande",
    "Mediano": "Mediano",
    "Micro": "Chico",
    "Chico": "Chico",
    "Nan": "General"
}

dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# ----------------------------------------------------------------------
# Funciones de lectura segura
# ----------------------------------------------------------------------
def leer_csv_seguro(ruta):
    try:
        df = pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(ruta, encoding='latin1', low_memory=False)
    df.columns = df.columns.str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    return df

def leer_excel_seguro(ruta):
    df = pd.read_excel(ruta)
    df.columns = df.columns.str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    return df

# ----------------------------------------------------------------------
# 1. Cargar archivos
# ----------------------------------------------------------------------
df_pred = leer_csv_seguro("predicciones_semana.csv")
df_clientes = leer_csv_seguro("ID_clientes.csv")
df_probs = leer_csv_seguro("probabilidades_clientes.csv")
df_restricciones = leer_excel_seguro("restricciones_productos.xlsx")

# ----------------------------------------------------------------------
# 2. Limpieza y normalización
# ----------------------------------------------------------------------
df_pred['PRODUCTO_ID_KEY'] = df_pred['PRODUCTO_ID_KEY'].astype(str).str.strip()
df_restricciones['PRODUCTO_ID_KEY'] = df_restricciones['PRODUCTO_ID_KEY'].astype(str).str.strip()

df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].astype(str).str.strip().str.lower()
df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].replace(EQUIVALENCIAS_SEGMENTACION)

df_restricciones['SEGMENTACION'] = df_restricciones['SEGMENTACION'].astype(str).str.lower().str.strip()
df_restricciones['MINIMO_VENTA'] = df_restricciones['MINIMO_VENTA'].fillna(0).astype(int)
df_restricciones['MULTIPLO_VENTA'] = df_restricciones['MULTIPLO_VENTA'].replace(0, 1).fillna(1).astype(int)

for col in dias:
    df_probs[col] = pd.to_numeric(df_probs[col], errors='coerce').fillna(0)

# ----------------------------------------------------------------------
# 3. Merge (sin duplicar SEGMENTACION)
# ----------------------------------------------------------------------
df_clientes = df_clientes[["CLIENTE_ID_KEY", "SEGMENTACION", "VAR_CAT_Cluster"]]

# Merge predicciones + clientes
df = pd.merge(df_pred, df_clientes, on="CLIENTE_ID_KEY", how="left")

# Merge con probabilidades (no hay SEGMENTACION aquí)
df = pd.merge(df, df_probs, on="CLIENTE_ID_KEY", how="left")

# Asegurar que SEGMENTACION sigue existiendo
if 'SEGMENTACION' not in df.columns:
    df['SEGMENTACION'] = df_clientes.set_index('CLIENTE_ID_KEY').loc[df['CLIENTE_ID_KEY'], 'SEGMENTACION'].values

# ----------------------------------------------------------------------
# 4. Distribuir predicciones por día (vectorizado)
# ----------------------------------------------------------------------
print(" Distribuyendo predicciones por día...")
probs = df[dias].to_numpy(dtype=float)
suma = probs.sum(axis=1, keepdims=True)
suma[suma == 0] = 1  # evitar división por cero
probs_norm = probs / suma
predicciones = df['PREDICCION'].to_numpy().reshape(-1, 1)
df_dias = pd.DataFrame(predicciones * probs_norm, columns=dias)

df_final = pd.concat([df[["CLIENTE_ID_KEY", "PRODUCTO_ID_KEY", "SEGMENTACION", "FECHA_PREDICCION"]], df_dias], axis=1)

# ----------------------------------------------------------------------
# 5. Ajuste a múltiplos
# ----------------------------------------------------------------------
print(" Ajustando múltiplos de venta...")
min_venta_dict = df_restricciones.set_index(['PRODUCTO_ID_KEY', 'SEGMENTACION'])['MINIMO_VENTA'].to_dict()
mult_venta_dict = df_restricciones.set_index(['PRODUCTO_ID_KEY', 'SEGMENTACION'])['MULTIPLO_VENTA'].to_dict()

def ajustar_fila(row):
    producto = row['PRODUCTO_ID_KEY']
    segmento = row.get('SEGMENTACION', 'general')
    valores = row[dias].values
    resultado = np.zeros_like(valores, dtype=int)
    min_venta = min_venta_dict.get((producto, segmento), 0)
    mult_venta = mult_venta_dict.get((producto, segmento), 1)
    for i, val in enumerate(valores):
        if val <= 0:
            resultado[i] = 0
        else:
            ajustado = max(val, min_venta)
            resultado[i] = math.ceil(ajustado / mult_venta) * mult_venta
    return pd.Series(resultado, index=dias)

df_final[dias] = df_final.apply(ajustar_fila, axis=1)

# ----------------------------------------------------------------------
# 6. Expandir a formato diario
# ----------------------------------------------------------------------
print(" Generando registros diarios...")
registros = []
for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Generando registros diarios"):
    fecha_base = pd.to_datetime(row["FECHA_PREDICCION"])
    for i, dia in enumerate(dias):
        registros.append({
            "CLIENTE_ID_KEY": row["CLIENTE_ID_KEY"],
            "PRODUCTO_ID_KEY": row["PRODUCTO_ID_KEY"],
            "Forecast_Cajas": row[dia],
            "FECHA_FORECAST": (fecha_base + timedelta(days=i)).strftime("%Y%m%d"),
            "INICIATIVA": "Cocacola",
        })

df_export = pd.DataFrame(registros)

# ----------------------------------------------------------------------
# 7. Guardar CSV
# ----------------------------------------------------------------------
df_export.to_csv("predicciones_formato_final_diario.csv", index=False, encoding='utf-8-sig')
print("\n Archivo guardado como predicciones_formato_final_diario.csv")
