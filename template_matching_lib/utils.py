"""
Módulo de utilidades para Template Matching
===========================================

Contiene funciones utilitarias y de procesamiento de alto nivel.
"""

import os
import glob
import cv2
from typing import List, Dict, Any

from .preprocessing import cargar_template, preprocesar_imagen
from .template_matching import buscar_coincidencias_multiescala, buscar_coincidencias_multiescala_multi
from .nms import aplicar_nms, aplicar_nms_multi_deteccion
from .visualization import (
    visualizar_preprocesamiento,
    visualizar_mapas_coincidencias,
    visualizar_resultado_final,
    visualizar_comparacion_escalas,
    visualizar_todas_las_detecciones,
    visualizar_detecciones_finales_numeradas
)


def obtener_imagenes_objetivo(patron_imagenes: str = None, config: Dict[str, Any] = None) -> List[str]:
    """
    Obtiene la lista de imágenes que contienen ciertos patrones en el nombre.
    
    Args:
        patron_imagenes: Patrón específico a buscar (opcional)
        config: Diccionario de configuración
    
    Returns:
        Lista de rutas de imágenes
    """
    if config is None:
        config = {'PATH_IMAGENES': 'TP3/images/'}
    
    if patron_imagenes:
        patrones = [patron_imagenes]
    else:
        # Patrones por defecto para modo single
        patrones = ['*logo*', '*retro*', '*LOGO*']
    
    imagenes = []
    path_imagenes = config.get('PATH_IMAGENES', 'TP3/images/')

    for patron in patrones:
        imagenes.extend(glob.glob(os.path.join(path_imagenes, patron + '.png')))
        imagenes.extend(glob.glob(os.path.join(path_imagenes, patron + '.jpg')))
        imagenes.extend(glob.glob(os.path.join(path_imagenes, patron + '.jpeg')))

    return imagenes


def obtener_todas_las_imagenes(config: Dict[str, Any], excluir_multi: bool = False) -> List[str]:
    """
    Obtiene todas las imágenes válidas del directorio.
    
    Args:
        config: Diccionario de configuración
        excluir_multi: Si True, excluye coca_multi.png de la lista
    
    Returns:
        Lista de rutas de imágenes
    """
    directorio_imagenes = config.get('PATH_IMAGENES', 'TP3/images/')
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    if not os.path.exists(directorio_imagenes):
        return []
    
    imagenes = [
        os.path.join(directorio_imagenes, f) 
        for f in os.listdir(directorio_imagenes) 
        if f.lower().endswith(extensiones_validas)
    ]
    
    # Excluir coca_multi.png si se solicita (para tests generales)
    if excluir_multi:
        imagenes = [img for img in imagenes if not os.path.basename(img).lower().startswith('coca_multi')]
    
    return imagenes


def procesar_imagen(ruta_imagen: str, template_data: Any, config: Dict[str, Any]) -> List[Dict]:
    """
    Procesa una sola imagen con el template (versión single detection).
    
    Args:
        ruta_imagen: Ruta a la imagen
        template_data: Template procesado
        config: Diccionario de configuración
    
    Returns:
        Lista de detecciones filtradas
    """
    nombre_imagen = os.path.basename(ruta_imagen)
    template_procesado = template_data

    # Cargar template original para comparación de escalas
    ruta_template = os.path.join(config['PATH_TEMPLATE'], 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    # Preprocesar imagen
    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen, config)

    # Visualizar preprocesamiento
    visualizar_preprocesamiento(template_procesado, imagen_procesada, nombre_imagen, config)

    # Búsqueda multiescala
    detecciones, mapas_resultado = buscar_coincidencias_multiescala(
        imagen_procesada, template_procesado, config
    )

    # Visualizar mapas de matching
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen, config)

    # Aplicar NMS
    detecciones_filtradas = aplicar_nms(detecciones, config)

    # Visualizar resultado final
    visualizar_resultado_final(imagen_original, detecciones_filtradas, nombre_imagen, config)

    # Visualizar comparación de escalas
    visualizar_comparacion_escalas(
        imagen_original, template_original, mapas_resultado, nombre_imagen, config
    )

    return detecciones_filtradas


def procesar_imagen_multi(ruta_imagen: str, template_data: Any, config: Dict[str, Any]) -> List[Dict]:
    """
    Procesa una imagen para múltiples detecciones.
    
    Args:
        ruta_imagen: Ruta a la imagen
        template_data: Template procesado
        config: Diccionario de configuración
    
    Returns:
        Lista de detecciones filtradas
    """
    nombre_imagen = os.path.basename(ruta_imagen)
    template_procesado = template_data

    print(f"\n{'='*60}")
    print(f"PROCESANDO: {nombre_imagen}")
    print(f"{'='*60}")

    # Cargar template original para comparación de escalas
    ruta_template = os.path.join(config['PATH_TEMPLATE'], 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    # Preprocesar imagen
    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen, config)
    print(f"Imagen cargada: {imagen_original.shape[1]}x{imagen_original.shape[0]} px")

    # Visualizar preprocesamiento
    visualizar_preprocesamiento(template_procesado, imagen_procesada, nombre_imagen, config)

    # Búsqueda en múltiples escalas
    detecciones, mapas_resultado = buscar_coincidencias_multiescala_multi(
        imagen_procesada, template_procesado, config
    )

    # Visualizar mapas de matching
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen, config)

    print(f"Detecciones antes del filtrado: {len(detecciones)}")

    # Aplicar NMS optimizado para múltiples detecciones
    detecciones_filtradas = aplicar_nms_multi_deteccion(detecciones, config)

    print(f"Detecciones finales: {len(detecciones_filtradas)}")

    # Generar todas las visualizaciones
    visualizar_comparacion_escalas(imagen_original, template_original, mapas_resultado, nombre_imagen, config)
    visualizar_todas_las_detecciones(imagen_original, detecciones, detecciones_filtradas, nombre_imagen, config)
    visualizar_detecciones_finales_numeradas(imagen_original, detecciones_filtradas, nombre_imagen, config)

    return detecciones_filtradas


def crear_directorio_resultados(config: Dict[str, Any]):
    """
    Crea el directorio de resultados si no existe.
    
    Args:
        config: Diccionario de configuración
    """
    carpeta_resultados = config.get('CARPETA_RESULTADOS', 'resultados')
    os.makedirs(carpeta_resultados, exist_ok=True)


def mostrar_resumen_configuracion(config: Dict[str, Any], titulo: str = "CONFIGURACIÓN ACTUAL"):
    """
    Muestra un resumen de la configuración actual.
    
    Args:
        config: Diccionario de configuración
        titulo: Título del resumen
    """
    print("\n" + "=" * 60)
    print(f"    {titulo}    ")
    print("=" * 60)
    print(f"   Método de matching: {config.get('METODO_MATCHING', 'N/A')}")
    
    if 'UMBRAL_SIMPLE_DETECCION' in config:
        print(f"   Umbral de detección: {config['UMBRAL_SIMPLE_DETECCION']}")
    elif 'UMBRAL_DETECCION' in config:
        print(f"   Umbral de detección: {config['UMBRAL_DETECCION']}")
    
    print(f"   Rango de escalas: {config.get('ESCALA_MIN', 'N/A')}x - {config.get('ESCALA_MAX', 'N/A')}x (paso: {config.get('PASO_ESCALA', 'N/A')})")
    
    if 'UMBRAL_CANNY' in config and isinstance(config['UMBRAL_CANNY'], tuple):
        print(f"   Umbral Canny: {config['UMBRAL_CANNY'][0]} - {config['UMBRAL_CANNY'][1]}")
    elif 'UMBRAL_CANNY_MIN' in config:
        print(f"   Umbral Canny: {config.get('UMBRAL_CANNY_MIN', 'N/A')} - {config.get('UMBRAL_CANNY_MAX', 'N/A')}")
    
    if 'CLUSTERING_EPS' in config:
        print(f"   Clustering eps: {config['CLUSTERING_EPS']}")
    
    if 'LIMITE_FINAL' in config:
        print(f"   Límite detecciones: {config['LIMITE_FINAL']}")
    elif 'LIMITE_DETECCIONES_FINALES' in config:
        print(f"   Límite detecciones: {config['LIMITE_DETECCIONES_FINALES']}")
    
    print()


def mostrar_resumen_resultados(resultados_totales: Dict[str, List[Dict]], config: Dict[str, Any]):
    """
    Muestra un resumen final de los resultados.
    
    Args:
        resultados_totales: Diccionario con resultados por imagen
        config: Diccionario de configuración
    """
    print("\n" + "=" * 60)
    print("                    RESUMEN FINAL                        ")
    print("=" * 60)

    total_detecciones = 0
    
    for nombre_img, detecciones in resultados_totales.items():
        num_detecciones = len(detecciones)
        total_detecciones += num_detecciones
        print(f"{nombre_img}: {num_detecciones} detecciones")

    print("-" * 60)
    print(f"TOTAL DETECCIONES: {total_detecciones}")
    print(f"Resultados guardados en: {config.get('CARPETA_RESULTADOS', 'resultados')}")
    print("Proceso completado exitosamente!")
    print("=" * 60)