import cv2
import numpy as np
from typing import Tuple, Dict, Any


def cargar_template(ruta_template: str, config: Dict[str, Any]) -> np.ndarray:
    """
    Carga y preprocesa el template usando método Canny.
    """
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)
    if template_original is None:
        raise ValueError(f"No se pudo cargar el template desde {ruta_template}")
    
    # Obtener parámetros de configuración
    if 'KERNEL_GAUSS' in config:
        # Configuración nueva (multi/general)
        kernel = config['KERNEL_GAUSS']
        sigma = config['SIGMA_GAUSS']
        umbral_canny = config['UMBRAL_CANNY']
        filtrar_ruido = config.get('FILTRAR_RUIDO', True)
    else:
        # Configuración legacy (single)
        kernel = config.get('KERNEL_GAUSSIANO', (5, 5))
        sigma = config.get('SIGMA_GAUSSIANO', 1.5)
        umbral_canny = (config.get('UMBRAL_CANNY_MIN', 100), 
                       config.get('UMBRAL_CANNY_MAX', 250))
        filtrar_ruido = config.get('FILTRAR_RUIDO', True)
    
    if filtrar_ruido:
        template_filtrado = cv2.GaussianBlur(template_original, kernel, sigma)
    else:
        template_filtrado = template_original
    
    template_procesado = cv2.Canny(template_filtrado, umbral_canny[0], umbral_canny[1])
    
    return template_procesado


def preprocesar_imagen(ruta_imagen: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa una imagen usando el método Canny.
    """
    imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen_original is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    # Obtener parámetros de configuración
    if 'KERNEL_GAUSS' in config:
        # Configuración nueva (multi/general)
        kernel = config['KERNEL_GAUSS']
        sigma = config['SIGMA_GAUSS']
        umbral_canny = config['UMBRAL_CANNY']
        filtrar_ruido = config.get('FILTRAR_RUIDO', True)
    else:
        # Configuración legacy (single)
        kernel = config.get('KERNEL_GAUSSIANO', (5, 5))
        sigma = config.get('SIGMA_GAUSSIANO', 1.5)
        umbral_canny = (config.get('UMBRAL_CANNY_MIN', 100), 
                       config.get('UMBRAL_CANNY_MAX', 250))
        filtrar_ruido = config.get('FILTRAR_RUIDO', True)
    
    if filtrar_ruido:
        imagen_filtrada = cv2.GaussianBlur(imagen_original, kernel, sigma)
    else:
        imagen_filtrada = imagen_original
    
    imagen_procesada = cv2.Canny(imagen_filtrada, umbral_canny[0], umbral_canny[1])
    
    return imagen_original, imagen_procesada


def redimensionar_template(template: np.ndarray, escala: float) -> np.ndarray:
    """
    Redimensiona el template a una escala específica.
    """
    nuevo_ancho = max(1, int(template.shape[1] * escala))
    nuevo_alto = max(1, int(template.shape[0] * escala))

    try:
        template_redimensionado = cv2.resize(template, (nuevo_ancho, nuevo_alto))
        return template_redimensionado
    except Exception:
        return None


def validar_dimensiones_template(template: np.ndarray, imagen: np.ndarray, escala: float) -> bool:
    """
    Valida si el template escalado cabe en la imagen.
    """
    nuevo_ancho = int(template.shape[1] * escala)
    nuevo_alto = int(template.shape[0] * escala)
    
    return (nuevo_ancho <= imagen.shape[1] and nuevo_alto <= imagen.shape[0])


def obtener_dimensiones_escaladas(template: np.ndarray, escala: float) -> Tuple[int, int]:
    """
    Obtiene las dimensiones del template escalado.
    """
    nuevo_ancho = max(1, int(template.shape[1] * escala))
    nuevo_alto = max(1, int(template.shape[0] * escala))
    
    return nuevo_ancho, nuevo_alto