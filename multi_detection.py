#!/usr/bin/env python3
"""Template Matching - Detección Múltiple"""

import os
import sys
from tqdm import tqdm

from template_matching_lib.config import ConfigMulti, ConfigGeneral, crear_config_dict
from template_matching_lib.preprocessing import cargar_template
from template_matching_lib.utils import obtener_todas_las_imagenes, procesar_imagen_multi, crear_directorio_resultados


def procesar_coca_multi():
    """Función para procesar específicamente la imagen coca_multi.png."""
    config = crear_config_dict(ConfigMulti)
    crear_directorio_resultados(config)

    template_data = cargar_template(f"{config['PATH_TEMPLATE']}pattern.png", config)
    ruta_imagen_multi = f"{config['PATH_IMAGENES']}coca_multi.png"
    
    if not os.path.exists(ruta_imagen_multi):
        print(f"ERROR: No se encontró la imagen {ruta_imagen_multi}")
        return

    detecciones = procesar_imagen_multi(ruta_imagen_multi, template_data, config)
    print(f"Proceso completado: {len(detecciones)} detecciones en coca_multi.png")


def test_algoritmo_todas_imagenes():
    """Función para testar el algoritmo en todas las imágenes del directorio."""
    config = crear_config_dict(ConfigGeneral)
    crear_directorio_resultados(config)

    template_data = cargar_template(f"{config['PATH_TEMPLATE']}pattern.png", config)
    imagenes = obtener_todas_las_imagenes(config, excluir_multi=True)
    
    if not imagenes:
        print(f"ERROR: No se encontraron imágenes válidas en {config['PATH_IMAGENES']}")
        return

    resultados = {}
    for ruta_imagen in tqdm(imagenes, desc="Procesando imágenes"):
        nombre = os.path.basename(ruta_imagen)
        try:
            detecciones = procesar_imagen_multi(ruta_imagen, template_data, config)
            resultados[nombre] = len(detecciones)
        except Exception:
            resultados[nombre] = 0

    total = sum(resultados.values())
    exitosas = sum(1 for x in resultados.values() if x > 0)
    print(f"Proceso completado: {total} detecciones en {exitosas}/{len(resultados)} imágenes")


def main():
    """Función principal del script de detección múltiple."""
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        test_algoritmo_todas_imagenes()
    else:
        procesar_coca_multi()


if __name__ == "__main__":
    main()