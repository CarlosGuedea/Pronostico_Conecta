import pandas as pd
import pyodbc
from datetime import datetime
from tqdm import tqdm
import sys
import io
import re

# Forzar salida UTF-8 limpia
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------------
# 1. Configuracion
# ----------------------------------------------------------------------
csv_clientes = "Clientes_Rutas.csv"                   # Contiene CLIENTE_ID_KEY validos
csv_datos = "predicciones_formato_final_diario.csv"   # Contiene Forecast_Cajas e Iniciativa
csv_salida = "filtrado.csv"

# Conexion a SQL Server
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.201.13;"
    "DATABASE=RTM_PRODUCTIVO;"
    "UID=AnalisisDatos;"
    "PWD=Conecta.2025;"
    "TrustServerCertificate=yes;"
)

# ----------------------------------------------------------------------
# 2. Leer y filtrar datos
# ----------------------------------------------------------------------
df_clientes = pd.read_csv(csv_clientes)
df_datos = pd.read_csv(csv_datos)

# Limpiar caracteres no imprimibles en columnas de texto
def limpiar_texto(s):
    if isinstance(s, str):
        return re.sub(r"[^\x20-\x7E]", "", s)
    return s

df_datos = df_datos.applymap(limpiar_texto)
df_clientes = df_clientes.applymap(limpiar_texto)

# Filtrar solo los clientes validos
df_filtrado = df_datos[df_datos["CLIENTE_ID_KEY"].isin(df_clientes["CLIENTE_ID_KEY"])].copy()

# Filtrar los que tienen Forecast_Cajas != 0
df_filtrado = df_filtrado[df_filtrado["Forecast_Cajas"] != 0].copy()

# Agregar FECHA_FORECAST si no existe
if "FECHA_FORECAST" not in df_filtrado.columns:
    df_filtrado["FECHA_FORECAST"] = datetime.today().strftime("%Y-%m-%d")

# Asegurarnos de que INICIATIVA exista
if "INICIATIVA" not in df_filtrado.columns:
    raise ValueError("La columna 'INICIATIVA' no existe en el archivo de predicciones.")

# Guardar CSV filtrado
df_filtrado.to_csv(csv_salida, index=False, encoding='utf-8')
print(f"Filtrado completo. Guardado en '{csv_salida}' con {len(df_filtrado)} filas.")

# ----------------------------------------------------------------------
# 3. Enviar datos al procedimiento almacenado en bloques de 1000
# ----------------------------------------------------------------------
if not df_filtrado.empty:
    # Seleccionar columnas segun el tipo [Sugerido].[TablaCarga]
    df_bulk = df_filtrado[[
        "CLIENTE_ID_KEY",
        "PRODUCTO_ID_KEY",
        "Forecast_Cajas",
        "FECHA_FORECAST",
        "INICIATIVA"
    ]]

    registros = list(df_bulk.itertuples(index=False, name=None))

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    bloque = 1000
    total_bloques = (len(registros) // bloque) + (1 if len(registros) % bloque else 0)
    print(f"Enviando datos en {total_bloques} bloques de hasta {bloque} registros...")

    for i in tqdm(range(0, len(registros), bloque)):
        chunk = registros[i:i + bloque]

        valores = []
        for c, p, f, fecha, ini in chunk:
            fecha_sql = "NULL" if pd.isna(fecha) else f"'{fecha}'"
            c = str(c).replace("'", "''")
            p = str(p).replace("'", "''")
            ini = str(ini).replace("'", "''")
            valores.append(f"('{c}','{p}',{f},{fecha_sql},'{ini}')")

        valores_sql = ",\n    ".join(valores)

        sql = f"""
        DECLARE @MiLista AS [Sugerido].[TablaCarga];

        INSERT INTO @MiLista (CLIENTE_ID_KEY, PRODUCTO_ID_KEY, Forecast_Cajas, FECHA_FORECAST, INICIATIVA)
        VALUES 
            {valores_sql};

        EXEC [Sugerido].[MovPedidoSugeridoTablaInterno] @MiLista;
        """

        try:
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"Error en bloque {i // bloque + 1}: {e}")
            conn.rollback()

    cursor.close()
    conn.close()
    print(f"Bulk insert completado exitosamente con {len(registros)} registros totales.")
else:
    print("No hay registros para insertar.")
