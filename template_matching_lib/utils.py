import os
import glob
import cv2
from typing import List, Dict, Any

from .preprocessing import preprocesar_imagen
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
    """Procesa una sola imagen con el template (versión single detection)."""
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
    """Procesa una imagen para múltiples detecciones."""
    nombre_imagen = os.path.basename(ruta_imagen)
    template_procesado = template_data

    # Cargar template original para comparación de escalas
    ruta_template = os.path.join(config['PATH_TEMPLATE'], 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    # Preprocesar imagen
    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen, config)

    # Visualizar preprocesamiento
    visualizar_preprocesamiento(template_procesado, imagen_procesada, nombre_imagen, config)

    # Búsqueda en múltiples escalas
    detecciones, mapas_resultado = buscar_coincidencias_multiescala_multi(
        imagen_procesada, template_procesado, config
    )

    # Visualizar mapas de matching
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen, config)

    # Aplicar NMS optimizado para múltiples detecciones
    detecciones_filtradas = aplicar_nms_multi_deteccion(detecciones, config)

    # Generar todas las visualizaciones
    visualizar_comparacion_escalas(imagen_original, template_original, mapas_resultado, nombre_imagen, config)
    visualizar_todas_las_detecciones(imagen_original, detecciones, detecciones_filtradas, nombre_imagen, config)
    visualizar_detecciones_finales_numeradas(imagen_original, detecciones_filtradas, nombre_imagen, config)

    return detecciones_filtradas


def crear_directorio_resultados(config: Dict[str, Any]):
    """Crea el directorio de resultados si no existe."""
    carpeta_resultados = config.get('CARPETA_RESULTADOS', 'resultados')
    os.makedirs(carpeta_resultados, exist_ok=True)