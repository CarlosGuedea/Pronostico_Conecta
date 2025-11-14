import pandas as pd
import pyodbc
from datetime import datetime
from tqdm import tqdm
import sys
import io
import re
import numpy as np

# Forzar salida UTF-8 limpia
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------------
# Funciones de lectura segura (Importadas del script de predicciones)
# ----------------------------------------------------------------------
def leer_csv_seguro(ruta):
    """
    Lee un CSV de forma segura manejando codificaciones (utf-8-sig, latin1) 
    y limpiando caracteres invisibles y espacios en las columnas de texto.
    """
    try:
        # Intenta leer con UTF-8 con BOM (Byte Order Mark)
        df = pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        # Falla a Latin1 si UTF-8 no funciona
        df = pd.read_csv(ruta, encoding='latin1', low_memory=False)
        
    # Limpia columnas (encabezados) de caracteres no imprimibles y espacios
    df.columns = df.columns.str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
    
    # Limpia el contenido de las columnas de tipo 'object' (strings)
    for col in df.select_dtypes(include=['object']).columns:
        # Reemplaza caracteres invisibles y espacios en el contenido
        df[col] = df[col].astype(str).str.replace(r"[\u200b\u200c\u200d\xa0]", "", regex=True).str.strip()
        # Además, reemplaza el caracter de reemplazo Unicode (\ufffd) si se coló
        df[col] = df[col].str.replace('\ufffd', '', regex=False) 
        
    return df

# ----------------------------------------------------------------------
# 1. Configuracion
# ----------------------------------------------------------------------
csv_clientes = "Clientes_Rutas.csv"  # Contiene CLIENTE_ID_KEY validos
csv_datos = "predicciones_formato_final_diario.csv"  # Contiene Forecast_Cajas e Iniciativa
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
# 2. Leer y filtrar datos (USANDO LA LECTURA SEGURA)
# ----------------------------------------------------------------------
print("Leyendo archivos de entrada de forma segura...")
try:
    df_clientes = leer_csv_seguro(csv_clientes)
    df_datos = leer_csv_seguro(csv_datos)
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo necesario: {e.filename}.")
    sys.exit(1)


# ----------------------------------------------------------------------
# 3. Filtrado
# ----------------------------------------------------------------------

# Filtrar solo los clientes validos
df_filtrado = df_datos[df_datos["CLIENTE_ID_KEY"].isin(df_clientes["CLIENTE_ID_KEY"])].copy()

# Filtrar los que tienen Forecast_Cajas != 0
df_filtrado = df_filtrado[df_filtrado["Forecast_Cajas"].fillna(0) != 0].copy()

# Estandarizar columnas
df_filtrado["CLIENTE_ID_KEY"] = df_filtrado["CLIENTE_ID_KEY"].astype(str).str.strip()
df_filtrado["PRODUCTO_ID_KEY"] = df_filtrado["PRODUCTO_ID_KEY"].astype(str).str.strip()
df_filtrado["INICIATIVA"] = df_filtrado["INICIATIVA"].astype(str).str.strip()


# Agregar o verificar FECHA_FORECAST
if "FECHA_FORECAST" not in df_filtrado.columns:
    # Usar el formato YYYYMMDD para compatibilidad
    df_filtrado["FECHA_FORECAST"] = datetime.today().strftime("%Y%m%d")
else:
    # Intentar limpiar la columna de fecha si existe
    df_filtrado["FECHA_FORECAST"] = df_filtrado["FECHA_FORECAST"].astype(str).str.strip()


# Guardar CSV filtrado (USANDO UTF-8 como codificación de salida)
try:
    df_filtrado.to_csv(csv_salida, index=False, encoding='utf-8')
    print(f"Filtrado completo. Guardado en '{csv_salida}' con {len(df_filtrado)} filas.")
except Exception as e:
    print(f"Error al guardar el archivo de salida '{csv_salida}': {e}")
    # Si la falla ocurre aquí, el entorno de ejecución no respeta la codificación de Python
    # Pero al menos los datos de entrada están limpios.
    sys.exit(1)


# ----------------------------------------------------------------------
# 4. Enviar datos al procedimiento almacenado en bloques de 1000
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

    # El bloque de conexión asume que pyodbc está instalado y el driver ODBC configurado.
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except pyodbc.Error as e:
        print(f"Error de conexión a la base de datos: {e}")
        # Terminar la ejecución si la conexión falla
        sys.exit(1)


    bloque = 1000
    total_bloques = (len(registros) // bloque) + (1 if len(registros) % bloque else 0)
    print(f"Enviando datos en {total_bloques} bloques de hasta {bloque} registros...")

    for i in tqdm(range(0, len(registros), bloque)):
        chunk = registros[i:i + bloque]

        valores = []
        for c, p, f, fecha, ini in chunk:
            # Limpieza y formateo de valores para SQL
            fecha_sql = "NULL" if pd.isna(fecha) else f"'{fecha}'"
            c = str(c).replace("'", "''")
            p = str(p).replace("'", "''")
            ini = str(ini).replace("'", "''")
            
            # Asegurar que los strings no contengan caracteres problemáticos (ASCII filter)
            c = c.encode('ascii', 'ignore').decode('ascii')
            p = p.encode('ascii', 'ignore').decode('ascii')
            ini = ini.encode('ascii', 'ignore').decode('ascii')

            valores.append(f"('{c}','{p}',{f},{fecha_sql},'{ini}')")

        valores_sql = ",\n ".join(valores)

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