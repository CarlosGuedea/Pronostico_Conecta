import pandas as pd
import numpy as np
import math
from datetime import timedelta
from tqdm import tqdm
import sys
import io

# Forzar salida UTF-8 para evitar problemas de visualización en la consola
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

# Corregido: Las claves deben estar en minúsculas para coincidir con la normalización posterior.
EQUIVALENCIAS_SEGMENTACION = {
    "extragrand": "Grande",
    "extragrande": "Grande",
    "mediano": "Mediano",
    "micro": "Chico",
    "chico": "Chico",
    "nan": "General"
}

# La capitalización de los valores de este diccionario (Chico, Mediano, etc.) es la que se
# usa para buscar en el diccionario LIMITES_SEMANAL.
LIMITES_SEMANAL = {"Chico": 157, "Mediano": 325, "Grande": 513, "General": 995} 
dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# ----------------------------------------------------------------------
# Funciones de lectura segura
# ----------------------------------------------------------------------
def leer_csv_seguro(ruta):
    """Lee un CSV de forma segura manejando codificaciones y limpiando caracteres invisibles."""
    try:
        df = pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(ruta, encoding='latin1', low_memory=False)
    
    # Expresión regular para limpiar caracteres invisibles/especiales como ZWS, NBSP
    clean_regex = r"[\u200b\u200c\u200d\xa0]"
    
    df.columns = df.columns.str.replace(clean_regex, "", regex=True).str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(clean_regex, "", regex=True).str.strip()
    return df

def leer_excel_seguro(ruta):
    """Lee un Excel de forma segura y limpia caracteres invisibles."""
    df = pd.read_excel(ruta)
    clean_regex = r"[\u200b\u200c\u200d\xa0]"
    df.columns = df.columns.str.replace(clean_regex, "", regex=True).str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(clean_regex, "", regex=True).str.strip()
    return df

# ----------------------------------------------------------------------
# Cargar archivos (ASUME que estos archivos existen en el directorio)
# ----------------------------------------------------------------------
print("Cargando archivos de datos...")
try:
    df_pred = leer_csv_seguro("predicciones_semana.csv")
    df_clientes = leer_csv_seguro("ID_clientes.csv")
    df_probs = leer_csv_seguro("probabilidades_clientes.csv")
    df_restricciones = leer_excel_seguro("restricciones_productos.xlsx")
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo necesario: {e.filename}. Asegúrese de que los archivos estén en la misma carpeta.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Limpieza y normalización
# ----------------------------------------------------------------------
df_pred['PRODUCTO_ID_KEY'] = df_pred['PRODUCTO_ID_KEY'].astype(str).str.strip()
df_restricciones['PRODUCTO_ID_KEY'] = df_restricciones['PRODUCTO_ID_KEY'].astype(str).str.strip()

# 1. Normalizar SEGMENTACION de clientes:
df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].astype(str).str.strip().str.lower()
# 2. Aplica la equivalencia (ahora con claves en minúsculas)
df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].replace(EQUIVALENCIAS_SEGMENTACION)
# 3. Capitaliza para usar las claves de LIMITES_SEMANAL (Chico, Grande, etc.)
df_clientes['SEGMENTACION'] = df_clientes['SEGMENTACION'].str.capitalize()


# Normalizar restricciones (SEGMENTACION se mantiene en minúsculas para las claves de dict)
df_restricciones['SEGMENTACION'] = df_restricciones['SEGMENTACION'].astype(str).str.lower().str.strip()
df_restricciones['MINIMO_VENTA'] = df_restricciones['MINIMO_VENTA'].fillna(0).astype(int)
df_restricciones['MULTIPLO_VENTA'] = df_restricciones['MULTIPLO_VENTA'].replace(0,1).fillna(1).astype(int)

# Normalizar probabilidades diarias
for col in dias:
    if col not in df_probs.columns:
        df_probs[col] = 1/7
    else:
        # Forzar a numérico y tratar NaN como 0 para la suma posterior
        df_probs[col] = pd.to_numeric(df_probs[col], errors='coerce').fillna(0)

# ----------------------------------------------------------------------
# Merge datos
# ----------------------------------------------------------------------
df_clientes = df_clientes.drop_duplicates('CLIENTE_ID_KEY')[["CLIENTE_ID_KEY","SEGMENTACION","VAR_CAT_Cluster"]]
df = pd.merge(df_pred, df_clientes, on="CLIENTE_ID_KEY", how="left")
df = pd.merge(df, df_probs, on="CLIENTE_ID_KEY", how="left")

# Asegurar que SEGMENTACION exista y tenga valor por defecto ('General' capitalizado)
if 'SEGMENTACION' not in df.columns:
    df['SEGMENTACION'] = 'General'
df['SEGMENTACION'] = df['SEGMENTACION'].fillna('General')

df['PREDICCION'] = df['PREDICCION'].astype(float)
if 'FECHA_PREDICCION' not in df.columns:
    # Si falta, asumir que es una columna que debería venir en df_pred
    print("Advertencia: 'FECHA_PREDICCION' no se encontró después del merge. Usando una columna por defecto (si existe).")
    # Nota: Si el archivo real no tiene esta columna, esto fallaría.
    # Por seguridad, si no existe en df, se podría crear una fecha ficticia.
    if 'FECHA_PREDICCION' in df_pred.columns:
         df['FECHA_PREDICCION'] = df_pred['FECHA_PREDICCION']
    else:
         # Asumir que la predicción es para la semana de la fecha actual
         df['FECHA_PREDICCION'] = pd.to_datetime(pd.Timestamp.now().normalize())


# ----------------------------------------------------------------------
# Distribuir predicciones por día
# ----------------------------------------------------------------------
print("Distribuyendo predicciones por día...")
probs = df[dias].to_numpy(dtype=float)
suma = probs.sum(axis=1, keepdims=True)
suma[suma==0] = 1 # Evitar división por cero
probs_norm = probs / suma
predicciones = df['PREDICCION'].to_numpy().reshape(-1,1)
df_dias = pd.DataFrame(predicciones * probs_norm, columns=dias)

df_final = pd.concat([df[['CLIENTE_ID_KEY','PRODUCTO_ID_KEY','SEGMENTACION','FECHA_PREDICCION']], df_dias], axis=1)

# Diccionarios de restricciones (producto y segmento en minúsculas son llaves compuestas)
# Nota: La segmentación en el dict de restricciones es en minúsculas.
min_venta_dict = df_restricciones.set_index(['PRODUCTO_ID_KEY','SEGMENTACION'])['MINIMO_VENTA'].to_dict()
mult_venta_dict = df_restricciones.set_index(['PRODUCTO_ID_KEY','SEGMENTACION'])['MULTIPLO_VENTA'].to_dict()

def ajustar_fila(row):
    """
    Aplica las restricciones de mínimo de venta y múltiplo de venta a una fila.
    """
    producto = row['PRODUCTO_ID_KEY']
    # La segmentación en df_final está capitalizada (e.g., 'Chico'), pero
    # el diccionario de restricciones usa minúsculas (e.g., 'chico'), por eso se aplica lower().
    segmento_dict = row.get('SEGMENTACION','General').lower() 
    
    valores = row[dias].values
    
    # Obtener restricciones específicas para este producto/segmento
    min_venta = min_venta_dict.get((producto, segmento_dict), 0)
    mult_venta = mult_venta_dict.get((producto, segmento_dict), 1)
    
    # 1. Aplicar mínimo de venta
    valores_ajustados = np.maximum(valores, min_venta)
    
    # 2. Aplicar múltiplo de venta (redondeo al múltiplo superior más cercano)
    valores_ajustados = np.ceil(valores_ajustados / mult_venta) * mult_venta
    
    # 3. Limpieza y conversión a entero final
    valores_ajustados = np.nan_to_num(valores_ajustados, nan=0, posinf=0, neginf=0)
    
    # Devolver los valores como enteros, ya que representan cajas/unidades.
    return pd.Series(valores_ajustados.astype(int), index=dias)

# ----------------------------------------------------------------------
# Ajuste inicial a múltiplos y mínimos por producto
# ----------------------------------------------------------------------
print("Ajustando múltiplos y mínimos de venta (ajuste inicial)...")
df_final[dias] = df_final.apply(ajustar_fila, axis=1)

# ----------------------------------------------------------------------
# Aplicar límite semanal por cliente al final (y re-ajustar a mínimos/múltiplos)
# ----------------------------------------------------------------------
print("Aplicando límite semanal por cliente y reajustando a múltiplos...")

# Uso de tqdm para feedback de progreso
for cliente, grupo in tqdm(df_final.groupby('CLIENTE_ID_KEY'), desc="Aplicando limites semanales", unit="cliente"):
    # La SEGMENTACION ya está capitalizada correctamente (e.g., 'Chico')
    segmento = grupo['SEGMENTACION'].iloc[0] 
    limite_semanal = LIMITES_SEMANAL.get(segmento, None)
    
    if limite_semanal is not None:
        # Calcular el total actual (después del ajuste inicial a múltiplos)
        total_cliente = grupo[dias].sum().sum()
        
        if total_cliente > limite_semanal:
            factor = limite_semanal / total_cliente
            
            # 1. Aplicar el factor de escala (resultado flotante)
            scaled_values = grupo[dias] * factor
            
            # 2. Crear un DataFrame temporal para la re-aplicación del ajuste.
            temp_df = grupo[['PRODUCTO_ID_KEY', 'SEGMENTACION']].copy()
            temp_df[dias] = scaled_values
            
            # 3. Re-aplicar la lógica de ajuste (mínimos y múltiplos) a los valores escalados (flotantes).
            # Esto genera los valores enteros finales respetando las restricciones.
            ajustados_final = temp_df.apply(ajustar_fila, axis=1)
            
            # 4. Actualizar el DataFrame original con los valores finales ajustados
            df_final.loc[grupo.index, dias] = ajustados_final

# ----------------------------------------------------------------------
# Expandir a formato diario
# ----------------------------------------------------------------------
print("Generando registros diarios...")
registros = []
# Es necesario asegurar que FECHA_PREDICCION sea datetime antes de iterar
df_final['FECHA_PREDICCION'] = pd.to_datetime(df_final['FECHA_PREDICCION'])

for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Generando registros diarios"):
    fecha_base = row["FECHA_PREDICCION"]
    for i, dia in enumerate(dias):
        # row[dia] es la predicción final ajustada (entero)
        registros.append({
            "CLIENTE_ID_KEY": row["CLIENTE_ID_KEY"],
            "PRODUCTO_ID_KEY": row["PRODUCTO_ID_KEY"],
            # Se usa el valor entero ajustado
            "Forecast_Cajas": row[dia], 
            "FECHA_FORECAST": (fecha_base + timedelta(days=i)).strftime("%Y%m%d"),
            "INICIATIVA": "Cocacola",
        })

df_export = pd.DataFrame(registros)

# ----------------------------------------------------------------------
# Guardar CSV
# ----------------------------------------------------------------------
df_export.to_csv("predicciones_formato_final_diario.csv", index=False, encoding='utf-8-sig')
print("\nArchivo guardado como predicciones_formato_final_diario.csv")
