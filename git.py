import os
import platform
import subprocess

def agregar_git_a_path():
    sistema = platform.system().lower()

    if "windows" in sistema:
        posibles_rutas = [
            r"C:\Program Files\Git\bin",
            r"C:\Program Files\Git\cmd",
            r"C:\Program Files (x86)\Git\bin",
            r"C:\Program Files (x86)\Git\cmd"
        ]

        # Verifica si Git ya está en PATH
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            print("✅ Git ya está disponible en el PATH.")
            return
        except Exception:
            pass

        # Busca git.exe en las rutas conocidas
        for ruta in posibles_rutas:
            git_exe = os.path.join(ruta, "git.exe")
            if os.path.exists(git_exe):
                print(f"🧭 Git encontrado en: {ruta}")
                print("➕ Agregando al PATH del sistema...")

                # Agrega al PATH del sistema de forma persistente
                os.system(f'setx PATH "%PATH%;{ruta}"')
                print("✅ Git agregado al PATH. Reinicia tu terminal para aplicar los cambios.")
                return
        
        print("⚠️ No se encontró Git en las rutas comunes. Instálalo desde https://git-scm.com/downloads")

    elif "linux" in sistema or "darwin" in sistema:
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            print("✅ Git ya está disponible en el PATH.")
            return
        except Exception:
            pass

        rutas_comunes = ["/usr/bin/git", "/usr/local/bin/git"]
        for ruta in rutas_comunes:
            if os.path.exists(ruta):
                print(f"🧭 Git encontrado en {ruta}")
                bashrc = os.path.expanduser("~/.bashrc")
                with open(bashrc, "a") as f:
                    f.write(f'\nexport PATH="$PATH:{os.path.dirname(ruta)}"\n')
                print(f"✅ Git agregado al PATH en {bashrc}. Ejecuta 'source ~/.bashrc' para aplicar los cambios.")
                return
        
        print("⚠️ No se encontró Git en rutas comunes. Instálalo con:")
        print("   sudo apt install git -y   (Ubuntu/Debian)")
        print("   brew install git          (macOS con Homebrew)")

    else:
        print(f"❌ Sistema no soportado automáticamente ({sistema}).")

if __name__ == "__main__":
    agregar_git_a_path()
