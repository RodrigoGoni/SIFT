import cv2
from typing import Dict, Any

# Configuraciones base
PATHS = {'PATH_IMAGENES': 'TP3/images/', 'PATH_TEMPLATE': 'TP3/template/'}
MATCHING = {'METODO_MATCHING': cv2.TM_CCOEFF_NORMED}
CANNY = {'UMBRAL_CANNY': (100, 250), 'FILTRAR_RUIDO': True, 'KERNEL_GAUSS': (5, 5), 'SIGMA_GAUSS': 1.5}
VIZ = {'DPI_FIGURA': 100, 'ALPHA_VISUALIZACION': 0.7, 'PADDING_BBOX': 0.3}

# Configuraciones específicas
CONFIGS = {
    'single': {
        **PATHS, **MATCHING, **CANNY, **VIZ,
        'ESCALA_MIN': 0.4, 'ESCALA_MAX': 3.0, 'PASO_ESCALA': 0.05,
        'UMBRAL_SIMPLE_DETECCION': 0.04, 'UMBRAL_IOU_NMS': 0.04,
        'UMBRAL_CONFIANZA_NORMALIZADA': 0.6, 'LIMITE_FINAL': 50,
        'LIMITE_DETECCIONES_FINALES': 10, 'EARLY_STOPPING_ESCALAS': 3,
        'CARPETA_RESULTADOS': 'resultados_canny'
    },
    'multi': {
        **PATHS, **MATCHING, **VIZ,
        'UMBRAL_CANNY': (150, 250), 'KERNEL_GAUSS': (3, 3), 'SIGMA_GAUSS': 1.7,
        'ESCALA_MIN': 0.24, 'ESCALA_MAX': 0.25, 'PASO_ESCALA': 0.01,
        'UMBRAL_DETECCION': 0.05, 'UMBRAL_IOU_NMS': 0.2, 'LIMITE_FINAL': 50,
        'CLUSTERING_EPS': 15, 'UMBRAL_CONFIANZA_NORMALIZADA': 0.6,
        'MAX_DETECCIONES_POR_ESCALA': 200, 'EARLY_STOPPING_ESCALAS': 3,
        'CARPETA_RESULTADOS': 'resultados_canny_multi'
    },
    'general': {
        **PATHS, **MATCHING, **VIZ,
        'UMBRAL_CANNY': (100, 250), 'KERNEL_GAUSS': (5, 5), 'SIGMA_GAUSS': 1.7,
        'ESCALA_MIN': 0.25, 'ESCALA_MAX': 2.8, 'PASO_ESCALA': 0.01,
        'UMBRAL_DETECCION': 0.03, 'UMBRAL_IOU_NMS': 0.08, 'LIMITE_FINAL': 10,
        'CLUSTERING_EPS': 20, 'UMBRAL_CONFIANZA_NORMALIZADA': 0.6,
        'EARLY_STOPPING_ESCALAS': 20, 'CARPETA_RESULTADOS': 'resultados_test_general'
    }
}

def crear_config_dict(config_name: str) -> Dict[str, Any]:
    """Crea config dict desde string."""
    return CONFIGS.get(config_name, CONFIGS['single'])

def obtener_config_para_modo(modo: str) -> Dict[str, Any]:
    """Obtiene configuración por modo."""
    return CONFIGS.get(modo, CONFIGS['single'])