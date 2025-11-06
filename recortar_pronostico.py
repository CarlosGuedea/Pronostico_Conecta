import pandas as pd
from datetime import date, timedelta

# ==============================
# FECHAS Y SEMANA OBJETIVO
# ==============================
hoy = date.today()
proxima_semana = hoy + timedelta(days=7)
iso_year, iso_week, _ = proxima_semana.isocalendar()

print(f"Filtrando pronóstico para la semana ISO {iso_week} del año {iso_year}...")

# ==============================
# CARGAR Y PROCESAR DATOS
# ==============================
df = pd.read_csv("predicciones_fourier_secuencial.csv")

# Convertir la columna de fechas a datetime
df["FECHA_PREDICCION"] = pd.to_datetime(df["FECHA_PREDICCION"])

# Extraer la semana y el año ISO
df["ISO_Year"] = df["FECHA_PREDICCION"].dt.isocalendar().year
df["ISO_Week"] = df["FECHA_PREDICCION"].dt.isocalendar().week

# ==============================
# FILTRAR LA SEMANA SIGUIENTE
# ==============================
df_semana_sig = df[(df["ISO_Year"] == iso_year) & (df["ISO_Week"] == iso_week)]

# ==============================
# GUARDAR RESULTADO (nombre fijo)
# ==============================
df_semana_sig.to_csv("predicciones_semana.csv", index=False)

print("Archivo generado correctamente: predicciones_semana.csv")
