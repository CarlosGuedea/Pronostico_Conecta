import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import io

# Forzar salida UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------------
# 0. Configuración
# ----------------------------------------------------------------------
EQUIVALENCIAS_SEGMENTACION = {
    "extragrand": "grande",
    "extragrande": "grande",
    "mediano": "mediano",
    "micro": "chico",
    "chico": "chico",
    "nan": "general"
}

def leer_csv_seguro(ruta):
    """Lee CSV manejando posibles caracteres invisibles o codificaciones raras."""
    try:
        df = pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(ruta, encoding='latin1', low_memory=False)
    # Limpieza de caracteres invisibles en nombres de columnas
    df.columns = df.columns.str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    # Limpieza general de strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    return df

def leer_excel_seguro(ruta):
    """Lee Excel manejando caracteres invisibles."""
    df = pd.read_excel(ruta)
    df.columns = df.columns.str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    return df

# ----------------------------------------------------------------------
# 1. Cargar archivos limpiando codificación
# ----------------------------------------------------------------------
try:
    df_pred = leer_csv_seguro("predicciones_semana.csv")
    df_clientes = leer_csv_seguro("ID_clientes.csv")
    df_probs = leer_csv_seguro("probabilidades_clientes.csv")
    df_restricciones = leer_excel_seguro("restricciones_productos.xlsx")
except Exception as e:
    print(f"❌ Error al cargar archivos: {e}")
    exit(1)

# ----------------------------------------------------------------------
# 2. Normalización y limpieza
# ----------------------------------------------------------------------
df_pred['PRODUCTO_ID_KEY'] = df_pred['PRODUCTO_ID_KEY'].astype(str).str.strip()
df_restricciones['PRODUCTO_ID_KEY'] = df_restricciones['PRODUCTO_ID_KEY'].astype(str).str.strip()
df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].astype(str).str.strip().str.lower()
df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].replace(EQUIVALENCIAS_SEGMENTACION)
df_restricciones['SEGMENTACION'] = df_restricciones['SEGMENTACION'].astype(str).str.lower().str.strip()
df_restricciones['MINIMO_VENTA'] = df_restricciones['MINIMO_VENTA'].fillna(0).astype(int)
df_restricciones['MULTIPLO_VENTA'] = df_restricciones['MULTIPLO_VENTA'].replace(0, 1).fillna(1).astype(int)

for col in ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]:
    df_probs[col] = pd.to_numeric(df_probs[col], errors='coerce').fillna(0)

# ----------------------------------------------------------------------
# 3. Merge
# ----------------------------------------------------------------------
df_clientes = df_clientes[["CLIENTE_ID_KEY", "SEGMENTACION", "VAR_CAT_Cluster"]]
df = pd.merge(df_pred, df_clientes, on="CLIENTE_ID_KEY", how="left")
df = pd.merge(df, df_probs, on="CLIENTE_ID_KEY", how="left")

# ----------------------------------------------------------------------
# 4. Distribuir por días
# ----------------------------------------------------------------------
dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

def distribuir_dias(row):
    probs = row[dias].values.astype(float)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.zeros_like(probs)
    return pd.Series(row["PREDICCION"] * probs, index=dias)

print("🧮 Distribuyendo predicciones por día...")
tqdm.pandas(desc="Distribuyendo días")
df_dias = df.progress_apply(distribuir_dias, axis=1)
df_final = pd.concat([df[["CLIENTE_ID_KEY", "PRODUCTO_ID_KEY", "SEGMENTACION", "FECHA_PREDICCION"]], df_dias], axis=1)

# ----------------------------------------------------------------------
# 5. Ajuste a múltiplos
# ----------------------------------------------------------------------
def ajustar_multiplos_simplificado_int(row, restricciones):
    producto = row["PRODUCTO_ID_KEY"]
    segmento_venta = row.get("SEGMENTACION", "general")
    valores = pd.to_numeric(row[dias], errors='coerce').fillna(0).values
    resultado = np.zeros_like(valores, dtype=int)
    restr_general = restricciones[restricciones["PRODUCTO_ID_KEY"] == producto]
    restr_segmento = restr_general[restr_general["SEGMENTACION"] == segmento_venta]
    restr_final = restr_segmento.iloc[0] if not restr_segmento.empty else restr_general.iloc[0] if not restr_general.empty else None
    if restr_final is None:
        return np.round(valores).astype(int)
    min_venta = int(restr_final["MINIMO_VENTA"])
    mult_venta = int(restr_final["MULTIPLO_VENTA"])
    for i, val in enumerate(valores):
        if val <= 0:
            resultado[i] = 0
        else:
            ajustado = max(val, min_venta)
            resultado[i] = math.ceil(ajustado / mult_venta) * mult_venta
    return resultado

print("⚙️ Ajustando múltiplos de venta...")
tqdm.pandas(desc="Ajustando múltiplos")
ajustados = pd.DataFrame(
    df_final.progress_apply(lambda row: ajustar_multiplos_simplificado_int(row, df_restricciones), axis=1).to_list(),
    columns=dias
)
df_final[dias] = ajustados

# ----------------------------------------------------------------------
# 6. Expandir a formato diario
# ----------------------------------------------------------------------
print("📆 Generando registros diarios...")
registros = []
for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Generando registros diarios"):
    fecha_base = pd.to_datetime(row["FECHA_PREDICCION"])
    for i, dia in enumerate(dias):
        registros.append({
            "CLIENTE_ID_KEY": row["CLIENTE_ID_KEY"],
            "PRODUCTO_ID_KEY": row["PRODUCTO_ID_KEY"],
            "Forecast_Cajas": row[dia],
            "FECHA_FORECAST": (fecha_base + pd.Timedelta(days=i)).strftime("%Y%m%d"),
            "INICIATIVA": "Cocacola",
        })

df_export = pd.DataFrame(registros)

# ----------------------------------------------------------------------
# 7. Guardar CSV
# ----------------------------------------------------------------------
df_export.to_csv("predicciones_formato_final_diario.csv", index=False, encoding='utf-8-sig')
print("\n✅ Archivo guardado como predicciones_formato_final_diario.csv")
