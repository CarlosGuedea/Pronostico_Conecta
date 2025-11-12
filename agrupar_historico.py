import pandas as pd
import os

archivo = 'Historico_Completo_por_chunks2.csv'
salida_parcial = 'piezas_vendidas_por_semana_tmp.csv'
salida_final = 'piezas_vendidas_por_semana.csv'

chunksize = 500_000  # Ajusta según la memoria de tu PC

# Si existe un archivo parcial previo, lo borramos
if os.path.exists(salida_parcial):
    os.remove(salida_parcial)

try:
    lector = pd.read_csv(archivo, encoding='utf-8-sig', low_memory=False, chunksize=chunksize)
except UnicodeDecodeError:
    lector = pd.read_csv(archivo, encoding='ISO-8859-1', low_memory=False, chunksize=chunksize)

for i, chunk in enumerate(lector):
    print(f" Procesando bloque {i + 1}...")

    # Convertir FECHA_PEDIDO a datetime
    chunk['FECHA_PEDIDO'] = pd.to_datetime(chunk['FECHA_PEDIDO'], format='%Y-%m-%d', errors='coerce')
    chunk = chunk.dropna(subset=['FECHA_PEDIDO'])

    # Asegurar tipo numérico en piezas vendidas
    chunk['VAR_NUM_PiezasVendidas'] = pd.to_numeric(chunk['VAR_NUM_PiezasVendidas'], errors='coerce').fillna(0)

    # Calcular año y mes si no existen
    if 'año' not in chunk.columns:
        chunk['año'] = chunk['FECHA_PEDIDO'].dt.year
    if 'mes' not in chunk.columns:
        chunk['mes'] = chunk['FECHA_PEDIDO'].dt.month

    # Agrupar por cliente, producto, año y mes
    agrupado = chunk.groupby(
        ['CLIENTE_ID_KEY', 'PRODUCTO_ID_KEY', 'año', 'mes'],
        as_index=False
    )['VAR_NUM_PiezasVendidas'].sum()

    # Escribir directamente al CSV temporal
    modo = 'a' if os.path.exists(salida_parcial) else 'w'
    encabezado = not os.path.exists(salida_parcial)
    agrupado.to_csv(salida_parcial, mode=modo, index=False, header=encabezado, encoding='utf-8-sig')

print(" Procesamiento por bloques terminado.")
print(" Reagrupando resultados parciales...")

# --- Segunda fase: reagrupar el archivo reducido ---
df = pd.read_csv(salida_parcial, encoding='utf-8-sig', low_memory=False)
df_final = df.groupby(
    ['CLIENTE_ID_KEY', 'PRODUCTO_ID_KEY', 'año', 'mes'],
    as_index=False
)['VAR_NUM_PiezasVendidas'].sum()

# Crear columnas de fecha y semana
df_final['fecha_mes'] = pd.to_datetime(df_final['año'].astype(str) + '-' + df_final['mes'].astype(str) + '-01')
df_final['numero_semana'] = df_final['fecha_mes'].dt.isocalendar().week
df_final['inicio_semana'] = df_final['fecha_mes'] - pd.to_timedelta(df_final['fecha_mes'].dt.weekday, unit='d')

# Guardar el archivo final optimizado
df_final.to_csv(salida_final, index=False, encoding='utf-8-sig')

# Limpiar archivo temporal si deseas
os.remove(salida_parcial)

print(f" Archivo '{salida_final}' generado con éxito ({len(df_final):,} filas).")
