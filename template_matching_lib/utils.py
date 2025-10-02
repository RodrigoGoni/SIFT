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


def obtener_imagenes(patron_imagenes: str = None, config: Dict[str, Any] = None, excluir_multi: bool = True) -> List[str]:
    """
    Obtiene la lista de imágenes del directorio.
    """
    if config is None:
        config = {'PATH_IMAGENES': 'TP3/images/'}
    
    directorio_imagenes = config.get('PATH_IMAGENES', 'TP3/images/')
    
    if not os.path.exists(directorio_imagenes):
        return []
    
    imagenes = []
    
    if patron_imagenes:
        # Buscar por patrón específico
        patrones = [patron_imagenes]
        for patron in patrones:
            imagenes.extend(glob.glob(os.path.join(directorio_imagenes, patron + '.png')))
    else:
        # Si no hay patrón específico, usar patrones por defecto o todas las imágenes
        patrones_por_defecto = ['*logo*', '*retro*', '*LOGO*']
        
        # Intentar con patrones por defecto primero
        for patron in patrones_por_defecto:
            imagenes.extend(glob.glob(os.path.join(directorio_imagenes, patron + '.png')))
        
        # Si no se encontraron imágenes con patrones, obtener todas las .png
        if not imagenes:
            imagenes = [
                os.path.join(directorio_imagenes, f) 
                for f in os.listdir(directorio_imagenes) 
                if f.lower().endswith('.png')
            ]
    
    # Excluir imágenes multi si se especifica
    if excluir_multi:
        imagenes = [img for img in imagenes if not os.path.basename(img).lower().startswith('coca_multi')]
    
    return imagenes


def procesar_imagen(ruta_imagen: str, template_data: Any, config: Dict[str, Any]) -> List[Dict]:
    """Procesa una sola imagen con el template."""
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