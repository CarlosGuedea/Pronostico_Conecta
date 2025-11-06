from prefect import flow, task
import subprocess

@task
def etapa_1_extraer_datos():
    subprocess.run(["python", "descarga_sql.py"], check=True)

@task
def etapa_2_agrupar_por_semana():
    subprocess.run(["python", "agrupar_historico.py"], check=True)

@task
def etapa_3_unir_cluster():
    subprocess.run(["python", "agregar_segmentacion.py"], check=True)

@task
def etapa_4_prediccion_fourier():
    subprocess.run(["python", "naive.py"], check=True)

@task
def etapa_5_filtrar_semana():
    subprocess.run(["python", "recortar_pronostico.py"], check=True)

@task
def etapa_6_postprocesar():
    result = subprocess.run(
        ["python", "distribuir_por_dia.py"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()

@task
def etapa_7_filtrar_rutas():
    subprocess.run(["python", "filtrar_rutas.py"], check=True)

@flow(name="pipeline_ventas_completo")
def pipeline_ventas():
    # Flujo secuencial
    #etapa_1_extraer_datos()
    #etapa_2_agrupar_por_semana()
    #etapa_3_unir_cluster()
    #etapa_4_prediccion_fourier()
    etapa_5_filtrar_semana()
    etapa_6_postprocesar()
    etapa_7_filtrar_rutas()

if __name__ == "__main__":
    pipeline_ventas()
