import pandas as pd
from tqdm import tqdm

# === Leer archivo de clientes ===
clientes_df = pd.read_csv("ID_clientes.csv")

# === Leer ventas ignorando la columna problemática ===
# Primero obtenemos las columnas
all_columns = pd.read_csv("piezas_vendidas_por_semana.csv", nrows=1).columns
columnas_a_leer = [c for c in all_columns if c != "VISITAS_SABORES"]

# Leer CSV completo solo con columnas permitidas
ventas_df = pd.read_csv("piezas_vendidas_por_semana.csv", usecols=columnas_a_leer)

# === Join con clientes ===
df_unido = pd.merge(
    ventas_df,
    clientes_df[["CLIENTE_ID_KEY", "VAR_CAT_Cluster", "SEGMENTACION"]],
    on="CLIENTE_ID_KEY",
    how="left"
)

# === Guardar el resultado ===
df_unido.to_csv("ventas_con_cluster.csv", index=False)

print("Unión completada. Archivo guardado como ventas_con_cluster.csv")
