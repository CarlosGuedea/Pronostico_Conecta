import pandas as pd
import pyodbc
import time
import sys
from datetime import datetime, timedelta

print("==== INICIANDO PIPELINE POR CHUNKS ====")

conn = None
try:
    # ======================
    # Conexión
    # ======================
    print("Conectando a SQL Server...")
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=192.168.201.13;'
        'DATABASE=RTM_PRODUCTIVO;'
        'UID=AnalisisDatos;'
        'PWD=Conecta.2025'
    )

    # ======================
    # Cargar catálogos (productos y clientes)
    # ======================
    print("Descargando catálogo de productos (pandas)...")
    productos = pd.read_sql("EXEC [Sugerido].[TD_ObtenerProductos]", conn)
    print(f"Productos descargados: {len(productos):,} filas")
    # Seleccionar solo columnas útiles para reducir memoria
    prod_keep = [c for c in ("PRODUCTO_ID_KEY", "ActivoPreventa", "DESCRIPCION_PRODUCTO") if c in productos.columns]
    if prod_keep:
        productos = productos[prod_keep].copy()

    print("Cargando catálogo de clientes desde 'ID_clientes.csv'...")
    clientes = pd.read_csv("ID_clientes.csv")
    print(f"Clientes cargados: {len(clientes):,} filas")
    # Conservar las columnas útiles, incluyendo explícitamente VAR_CAT_Cluster
    cli_keep = [c for c in ("CLIENTE_ID_KEY", "VAR_CAT_Cluster", "Cliente", "NumeroCliente") if c in clientes.columns]
    if cli_keep:
        clientes = clientes[cli_keep].copy()

    # Homogeneizar tipos de llave para joins: usar string (evita problemas con NaN/int)
    if "PRODUCTO_ID_KEY" in productos.columns:
        productos["PRODUCTO_ID_KEY"] = productos["PRODUCTO_ID_KEY"].astype("Int64").astype(str).fillna("")
    if "CLIENTE_ID_KEY" in clientes.columns:
        clientes["CLIENTE_ID_KEY"] = clientes["CLIENTE_ID_KEY"].astype("Int64").astype(str).fillna("")

    # Preparar filtros a aplicar (funciones auxiliares)
    def activopreventa_mask(ser):
        # comprueba si la columna existe; transforma a string y busca valores típicos verdaderos
        s = ser.astype(str).str.upper().str.strip()
        return s.isin(["1", "TRUE", "SI", "SÍ", "YES", "Y", "T"])

    # ======================
    # Preparar la lectura por chunks del histórico
    # ======================
   # Fecha inicial: hoy
    fecha_ini = datetime.today()

    # Fecha final (puede ser fija o también hoy)
    fecha_fin_str = '20240501'
    fecha_fin = datetime.strptime(fecha_fin_str, '%Y%m%d')

    # Construir query con formato YYYYMMDD
    query_hist = f"EXEC [Sugerido].[TD_ObtenerHistoricoVentas] '{fecha_fin.strftime('%Y%m%d')}', '{fecha_ini.strftime('%Y%m%d')}'"

    print("Fecha inicial:", fecha_ini.strftime('%Y%m%d'))
    print("Fecha final:", fecha_fin.strftime('%Y%m%d'))
    print("Query:", query_hist)

    chunksize = 200_000  # ajusta según RAM/disco
    output_file = "Historico_Completo_por_chunks2.csv"
    first_write = True
    rows_written = 0
    start_total = time.time()

    print("Iniciando lectura del histórico por chunks...")
    # pandas read_sql con chunksize devuelve un generador de DataFrames
    for i, chunk in enumerate(pd.read_sql(query_hist, conn, chunksize=chunksize), start=1):
        t0 = time.time()
        print(f"\n--- Procesando chunk {i} (filas descargadas: {len(chunk):,}) ---")

        # Normalizar llaves a string para merge seguro
        if "PRODUCTO_ID_KEY" in chunk.columns:
            chunk["PRODUCTO_ID_KEY"] = chunk["PRODUCTO_ID_KEY"].astype("Int64").astype(str).fillna("")
        if "CLIENTE_ID_KEY" in chunk.columns:
            chunk["CLIENTE_ID_KEY"] = chunk["CLIENTE_ID_KEY"].astype("Int64").astype(str).fillna("")

        # Filtrar campaña NO y piezas >= 0 lo antes posible
        if "EsCamapaña" in chunk.columns:
            chunk = chunk[chunk["EsCamapaña"] == "NO"]
        else:
            print("Aviso: columna 'EsCamapaña' no encontrada en chunk; no se aplica ese filtro.")

        if "VAR_NUM_PiezasVendidas" in chunk.columns:
            chunk["VAR_NUM_PiezasVendidas"] = pd.to_numeric(chunk["VAR_NUM_PiezasVendidas"], errors="coerce").fillna(0)
            chunk = chunk[chunk["VAR_NUM_PiezasVendidas"] >= 0]
        else:
            print("Aviso: columna 'VAR_NUM_PiezasVendidas' no encontrada en chunk; no se aplica ese filtro.")

        print(f"Filas tras filtros iniciales: {len(chunk):,}")

        if len(chunk) == 0:
            print("Chunk vacío tras filtros; saltando.")
            # Si el primer chunk está vacío, salimos del bucle.
            break

        # Merge con productos (solo columnas cargadas)
        if "PRODUCTO_ID_KEY" in productos.columns and "PRODUCTO_ID_KEY" in chunk.columns:
            chunk = chunk.merge(productos, on="PRODUCTO_ID_KEY", how="left")
            print(f"Filas tras merge productos: {len(chunk):,}")

            # Filtrar ActivoPreventa == 1 (si columna existe)
            if "ActivoPreventa" in chunk.columns:
                try:
                    mask = activopreventa_mask(chunk["ActivoPreventa"])
                    chunk = chunk[mask]
                    print(f"Filas tras filtrar ActivoPreventa: {len(chunk):,}")
                except Exception as e:
                    print("Error aplicando filtro ActivoPreventa:", e, file=sys.stderr)
        else:
            print("Aviso: no se pudo hacer merge con productos (columna PRODUCTO_ID_KEY ausente).")

        if len(chunk) == 0:
            print("Chunk quedó vacío después de filtrar productos; saltando.")
            # Si el primer chunk queda vacío después de filtrar, salimos.
            break

        # Merge con clientes
        if "CLIENTE_ID_KEY" in clientes.columns and "CLIENTE_ID_KEY" in chunk.columns:
            chunk = chunk.merge(clientes, on="CLIENTE_ID_KEY", how="left")
            print(f"Filas tras merge clientes: {len(chunk):,}")

            # Filtrar clusters si existe la columna
            if "VAR_CAT_Cluster" in chunk.columns:
                chunk = chunk[chunk["VAR_CAT_Cluster"].isin(["TR TRADICIONAL", "ON PREMISE"])]
                print(f"Filas tras filtrar clusters: {len(chunk):,}")
            else:
                print("Aviso: columna 'VAR_CAT_Cluster' no encontrada; no se aplica filtro por cluster.")
        else:
            print("Aviso: no se pudo hacer merge con clientes (columna CLIENTE_ID_KEY ausente).")

        if len(chunk) == 0:
            print("Chunk quedó vacío después de filtrar clientes; saltando.")
            # Si el primer chunk queda vacío, salimos.
            break

        # Guardar chunk al CSV (append)
        chunk.to_csv(output_file, index=False, header=first_write, mode="a", encoding="utf-8-sig")
        rows_written += len(chunk)
        first_write = False

        t1 = time.time()
        print(f"Chunk {i} procesado y guardado ({len(chunk):,} filas) en {t1-t0:.1f}s. Total escrito: {rows_written:,} filas")

    t_end = time.time()
    print(f"\n==== FIN. Total filas escritas: {rows_written:,}. Tiempo total: {t_end-start_total:.1f}s ====")

except Exception as exc:
    print("Error durante la ejecución:", exc, file=sys.stderr)
    raise
finally:
    if conn is not None:
        try:
            conn.close()
            print("Conexión cerrada.")
        except Exception:
            pass