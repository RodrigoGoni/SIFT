#!/usr/bin/env python3
"""
Template Matching - Detección Múltiple
======================================

Script para detectar múltiples instancias del template por imagen.
Versión refactorizada del archivo template_matching_canny_multi.py original.
"""

import os
import sys
from tqdm import tqdm

from template_matching_lib.config import ConfigMulti, ConfigGeneral, crear_config_dict
from template_matching_lib.preprocessing import cargar_template
from template_matching_lib.utils import (
    obtener_todas_las_imagenes,
    procesar_imagen_multi,
    crear_directorio_resultados,
    mostrar_resumen_configuracion
)


def procesar_coca_multi():
    """Función para procesar específicamente la imagen coca_multi.png."""
    # Cargar configuración para múltiples detecciones
    config = crear_config_dict(ConfigMulti)
    
    # Mostrar configuración
    mostrar_resumen_configuracion(config, "TEMPLATE MATCHING CANNY - DETECCIÓN MÚLTIPLE DE LOGOS")

    # Crear directorio de resultados
    crear_directorio_resultados(config)

    # Cargar template
    ruta_template = f"{config['PATH_TEMPLATE']}pattern.png"
    template_data = cargar_template(ruta_template, config)

    # Procesar específicamente la imagen coca_multi.png
    ruta_imagen_multi = f"{config['PATH_IMAGENES']}coca_multi.png"
    
    if not os.path.exists(ruta_imagen_multi):
        print(f"ERROR: No se encontró la imagen {ruta_imagen_multi}")
        return

    detecciones = procesar_imagen_multi(ruta_imagen_multi, template_data, config)

    # Mostrar resumen final
    print("\n" + "=" * 80)
    print("                          RESUMEN FINAL                              ")
    print("=" * 80)
    print(f"Imagen procesada: coca_multi.png")
    print(f"Detecciones encontradas: {len(detecciones)}")
    
    if detecciones:
        print("\nDETALLE DE DETECCIONES (Normalizada | Original):")
        for i, det in enumerate(detecciones):
            confianza_orig = det.get('confianza_original', 'N/A')
            print(f"  #{i+1}: {det['confianza']:.3f} | {confianza_orig:.3f}")
    
    print(f"\nResultados guardados en: {config['CARPETA_RESULTADOS']}")
    print("Proceso de detección múltiple completado exitosamente!")
    print("=" * 80)


def test_algoritmo_todas_imagenes():
    """Función para testar el algoritmo en todas las imágenes del directorio."""
    # Cargar configuración para pruebas generales
    config = crear_config_dict(ConfigGeneral)
    
    print("\n" + "=" * 80)
    print("        TEMPLATE MATCHING CANNY - TEST GENERAL EN TODAS LAS IMÁGENES        ")
    print("        (EXCLUYENDO coca_multi.png - usar configuración general)             ")
    print("=" * 80)
    
    # Mostrar configuración
    mostrar_resumen_configuracion(config, "CONFIGURACIÓN PARA TEST GENERAL")

    # Crear directorio de resultados
    crear_directorio_resultados(config)

    # Cargar template
    ruta_template = f"{config['PATH_TEMPLATE']}pattern.png"
    template_data = cargar_template(ruta_template, config)

    # Obtener todas las imágenes del directorio EXCLUYENDO coca_multi.png
    imagenes = obtener_todas_las_imagenes(config, excluir_multi=True)
    
    if not imagenes:
        print(f"ERROR: No se encontraron imágenes válidas en {config['PATH_IMAGENES']}")
        return

    print(f"Se encontraron {len(imagenes)} imágenes para procesar (excluyendo coca_multi.png):")
    for img in imagenes:
        print(f"  - {os.path.basename(img)}")
    print()

    # Resumen de resultados
    resultados_generales = []

    # Procesar cada imagen
    for i, ruta_imagen in enumerate(imagenes, 1):
        nombre_imagen = os.path.basename(ruta_imagen)
        print("=" * 60)
        print(f"PROCESANDO ({i}/{len(imagenes)}): {nombre_imagen}")
        print("=" * 60)
        
        try:
            detecciones = procesar_imagen_multi(ruta_imagen, template_data, config)
            
            # Guardar resultados
            resultado = {
                'imagen': nombre_imagen,
                'detecciones': len(detecciones),
                'mejor_confianza': max([d['confianza'] for d in detecciones]) if detecciones else 0.0,
                'status': 'Exitoso'
            }
            resultados_generales.append(resultado)
            
        except Exception as e:
            print(f"ERROR procesando {nombre_imagen}: {str(e)}")
            resultado = {
                'imagen': nombre_imagen,
                'detecciones': 0,
                'mejor_confianza': 0.0,
                'status': f'Error: {str(e)[:50]}'
            }
            resultados_generales.append(resultado)

    # Mostrar resumen final
    print("\n" + "=" * 80)
    print("                          RESUMEN GENERAL FINAL                              ")
    print("=" * 80)
    
    exitosos = sum(1 for r in resultados_generales if r['status'] == 'Exitoso')
    total_detecciones = sum(r['detecciones'] for r in resultados_generales)
    
    print(f"Imágenes procesadas: {len(imagenes)} (excluyendo coca_multi.png)")
    print(f"Procesadas exitosamente: {exitosos}")
    print(f"Con errores: {len(imagenes) - exitosos}")
    print(f"Total de detecciones: {total_detecciones}")
    print()
    
    print("DETALLE POR IMAGEN:")
    for resultado in resultados_generales:
        status_icon = "✓" if resultado['status'] == 'Exitoso' else "✗"
        print(f"  {status_icon} {resultado['imagen']:<20} | "
              f"Detecciones: {resultado['detecciones']:>3} | "
              f"Mejor confianza: {resultado['mejor_confianza']:>6.3f} | "
              f"Status: {resultado['status']}")
    
    print(f"\nResultados guardados en: {config['CARPETA_RESULTADOS']}")
    print("Test general completado!")
    print("=" * 80)
    
    return resultados_generales


def main():
    """Función principal del script de detección múltiple."""
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        # Ejecutar test general en todas las imágenes EXCEPTO coca_multi.png
        # Usa CONFIG_TEST_GENERAL (ConfigGeneral) con escalas amplias
        test_algoritmo_todas_imagenes()
    else:
        # Procesar específicamente coca_multi.png
        # Usa CONFIG optimizado para múltiples detecciones (ConfigMulti)
        procesar_coca_multi()


if __name__ == "__main__":
    main()