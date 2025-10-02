import sys
from tqdm import tqdm

from template_matching_lib.config import ConfigSingle, crear_config_dict
from template_matching_lib.preprocessing import cargar_template
from template_matching_lib.utils import (
    obtener_imagenes_objetivo,
    procesar_imagen,
    crear_directorio_resultados,
    mostrar_resumen_configuracion,
    mostrar_resumen_resultados
)


def main():
    """Función principal del script de detección simple."""
    # Cargar configuración
    config = crear_config_dict(ConfigSingle)
    
    # Mostrar configuración
    mostrar_resumen_configuracion(config, "TEMPLATE MATCHING - DETECTOR DE BORDES CANNY")

    # Crear directorio de resultados
    crear_directorio_resultados(config)

    # Cargar template
    ruta_template = f"{config['PATH_TEMPLATE']}pattern.png"
    template_data = cargar_template(ruta_template, config)

    # Obtener imágenes objetivo
    imagenes = obtener_imagenes_objetivo(config=config)
    print(f"Imágenes encontradas: {len(imagenes)}")

    if not imagenes:
        print("No se encontraron imágenes para procesar.")
        return

    # Procesar cada imagen
    resultados_totales = {}
    
    for ruta_imagen in tqdm(imagenes, desc="Procesando imágenes", unit="img"):
        nombre_imagen = ruta_imagen.split('/')[-1]  # Obtener solo el nombre del archivo
        detecciones = procesar_imagen(ruta_imagen, template_data, config)
        resultados_totales[nombre_imagen] = detecciones

    # Mostrar resumen final
    mostrar_resumen_resultados(resultados_totales, config)


if __name__ == "__main__":
    main()