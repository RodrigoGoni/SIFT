import cv2
from typing import Dict, Any


class Config:
    """Configuración base compartida."""
    
    # Rutas
    PATH_IMAGENES = 'TP3/images/'
    PATH_TEMPLATE = 'TP3/template/'
    
    # Método de matching
    METODO_MATCHING = cv2.TM_CCOEFF_NORMED
    
    # Preprocesamiento Canny
    UMBRAL_CANNY = (100, 250)
    FILTRAR_RUIDO = True
    KERNEL_GAUSS = (5, 5)
    SIGMA_GAUSS = 1.5
    
    # Visualización
    DPI_FIGURA = 100
    ALPHA_VISUALIZACION = 0.7
    PADDING_BBOX = 0.3


class ConfigSingle(Config):
    """Configuración para detección simple (una detección por imagen)."""
    
    # Escalas
    ESCALA_MIN = 0.4
    ESCALA_MAX = 3.0
    PASO_ESCALA = 0.05
    
    # Umbrales Canny específicos
    UMBRAL_CANNY_MIN = 100
    UMBRAL_CANNY_MAX = 250
    KERNEL_GAUSSIANO = (5, 5)  # Compatibilidad con código legacy
    SIGMA_GAUSSIANO = 1.5      # Compatibilidad con código legacy
    
    # Detección
    UMBRAL_SIMPLE_DETECCION = 0.04
    UMBRAL_FINAL_NMS = 0.04
    UMBRAL_IOU_NMS = 0.04
    MAXIMO_MEJORES_CANDIDATOS = 10
    LIMITE_DETECCIONES_FINALES = 10
    
    # Early stopping
    EARLY_STOPPING_ESCALAS = 3
    
    # Resultados
    CARPETA_RESULTADOS = 'resultados_canny'


class ConfigMulti(Config):
    """Configuración para detección múltiple (múltiples detecciones por imagen)."""
    
    # Escalas optimizadas para múltiples detecciones
    ESCALA_MIN = 0.24
    ESCALA_MAX = 0.25
    PASO_ESCALA = 0.01
    
    # Preprocesamiento específico
    UMBRAL_CANNY = (150, 250)
    KERNEL_GAUSS = (3, 3)
    SIGMA_GAUSS = 1.7
    
    # Detección múltiple
    UMBRAL_DETECCION = 0.05
    UMBRAL_IOU_NMS = 0.2
    MAX_CANDIDATOS = 50
    MAX_DETECCIONES_POR_ESCALA = 200
    MAX_DETECCIONES_POR_GRUPO = 8
    LIMITE_FINAL = 50
    
    # Clustering
    CLUSTERING_EPS = 15
    CLUSTERING_MIN = 1
    UMBRAL_CONFIANZA_NORMALIZADA = 0.6
    
    # Early stopping
    EARLY_STOPPING_ESCALAS = 3
    
    # Visualización específica
    MAX_DETECCIONES_VISUALIZAR = 100
    MAX_ETIQUETAS_VISUALIZAR = 20
    
    # Resultados
    CARPETA_RESULTADOS = 'resultados_canny_multi'


class ConfigGeneral(Config):
    """Configuración para pruebas generales en todas las imágenes (equivalente a CONFIG_TEST_GENERAL)."""
    
    # Escalas más amplias - mismo rango que CONFIG_TEST_GENERAL original
    ESCALA_MIN = 0.25
    ESCALA_MAX = 2.8
    PASO_ESCALA = 0.01
    
    # Preprocesamiento - igual que CONFIG_TEST_GENERAL original
    UMBRAL_CANNY = (100, 250)
    KERNEL_GAUSS = (5, 5)
    SIGMA_GAUSS = 1.7
    
    # Detección - valores de CONFIG_TEST_GENERAL original
    UMBRAL_DETECCION = 0.03
    UMBRAL_IOU_NMS = 0.08
    MAX_CANDIDATOS = 20
    MAX_DETECCIONES_POR_ESCALA = 200
    MAX_DETECCIONES_POR_GRUPO = 8
    LIMITE_FINAL = 10
    
    # Clustering - valores de CONFIG_TEST_GENERAL original
    CLUSTERING_EPS = 20
    CLUSTERING_MIN = 1
    UMBRAL_CONFIANZA_NORMALIZADA = 0.6
    
    # Early stopping más permisivo - valor de CONFIG_TEST_GENERAL original
    EARLY_STOPPING_ESCALAS = 20
    
    # Visualización
    MAX_DETECCIONES_VISUALIZAR = 100
    MAX_ETIQUETAS_VISUALIZAR = 20
    
    # Resultados
    CARPETA_RESULTADOS = 'resultados_test_general'


def crear_config_dict(config_class) -> Dict[str, Any]:
    """
    Convierte una clase de configuración a un diccionario.
    
    Args:
        config_class: Clase de configuración (ConfigSingle, ConfigMulti, etc.)
    
    Returns:
        Dict con todos los atributos de configuración
    """
    config_dict = {}
    for attr in dir(config_class):
        if not attr.startswith('_') and not callable(getattr(config_class, attr)):
            config_dict[attr] = getattr(config_class, attr)
    return config_dict


def obtener_config_para_modo(modo: str) -> Dict[str, Any]:
    """
    Obtiene la configuración apropiada según el modo de operación.
    
    Args:
        modo: 'single', 'multi', o 'general'
    
    Returns:
        Diccionario de configuración
    """
    if modo == 'single':
        return crear_config_dict(ConfigSingle)
    elif modo == 'multi':
        return crear_config_dict(ConfigMulti)
    elif modo == 'general':
        return crear_config_dict(ConfigGeneral)
    else:
        raise ValueError(f"Modo no válido: {modo}. Use 'single', 'multi' o 'general'")