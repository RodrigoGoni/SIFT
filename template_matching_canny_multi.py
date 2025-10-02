import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
import warnings

warnings.filterwarnings('ignore')  # Más conciso

# Configuración optimizada para múltiples detecciones
CONFIG = {
    'PATH_IMAGENES': 'TP3/images/',
    'PATH_TEMPLATE': 'TP3/template/',
    'METODO_MATCHING': cv2.TM_CCOEFF_NORMED,
    'ESCALA_MIN': 0.24,
    'ESCALA_MAX': 0.25,
    'PASO_ESCALA': 0.01,
    'UMBRAL_CANNY': (150, 250),
    'KERNEL_GAUSS': (3, 3),
    'SIGMA_GAUSS': 1.7,
    'UMBRAL_DETECCION': 0.05,  # Para valores sin normalizar (early stopping)
    'UMBRAL_IOU_NMS': 0.2,
    'MAX_CANDIDATOS': 50,
    'MAX_DETECCIONES_POR_ESCALA': 200,  # Máximo de detecciones por escala antes de NMS
    'MAX_DETECCIONES_POR_GRUPO': 8,  # Máximo de detecciones por cluster/grupo
    'LIMITE_FINAL': 50,
    'DPI_FIGURA': 100,
    'CARPETA_RESULTADOS': 'resultados_canny_multi',
    'CLUSTERING_EPS': 15,
    'CLUSTERING_MIN': 1,
    'UMBRAL_CONFIANZA_NORMALIZADA': 0.6,  # Para valores normalizados (0-1)
    'EARLY_STOPPING_ESCALAS': 3,  # Nuevo parámetro para early stopping
    'ALPHA_VISUALIZACION': 0.7,  # Transparencia para visualizaciones
    'PADDING_BBOX': 0.3,  # Padding para cajas de texto en visualizaciones
    'MAX_DETECCIONES_VISUALIZAR': 100,  # Máximo de detecciones a mostrar en visualización
    'MAX_ETIQUETAS_VISUALIZAR': 20  # Máximo de etiquetas de texto a mostrar
}

# Nueva configuración para pruebas generales en todas las imágenes
CONFIG_TEST_GENERAL = {
    'PATH_IMAGENES': 'TP3/images/',
    'PATH_TEMPLATE': 'TP3/template/',
    'METODO_MATCHING': cv2.TM_CCOEFF_NORMED,
    'ESCALA_MIN': 0.25,
    'ESCALA_MAX': 2.8,
    'PASO_ESCALA': 0.01,
    'UMBRAL_CANNY': (100, 250),
    'KERNEL_GAUSS': (5, 5),
    'SIGMA_GAUSS': 1.7,
    'UMBRAL_DETECCION': 0.03,  # Para valores sin normalizar (early stopping)
    'UMBRAL_IOU_NMS': 0.08,
    'MAX_CANDIDATOS': 20,
    'MAX_DETECCIONES_POR_ESCALA': 200,  # Máximo de detecciones por escala antes de NMS
    'MAX_DETECCIONES_POR_GRUPO': 8,  # Máximo de detecciones por cluster/grupo
    'LIMITE_FINAL': 10,
    'DPI_FIGURA': 100,
    'CARPETA_RESULTADOS': 'resultados_test_general',
    'CLUSTERING_EPS': 20,
    'CLUSTERING_MIN': 1,
    'UMBRAL_CONFIANZA_NORMALIZADA': 0.6,  # Para valores normalizados (0-1)
    'EARLY_STOPPING_ESCALAS': 10,  # Nuevo parámetro para early stopping
    'ALPHA_VISUALIZACION': 0.7,  # Transparencia para visualizaciones
    'PADDING_BBOX': 0.3,  # Padding para cajas de texto en visualizaciones
    'MAX_DETECCIONES_VISUALIZAR': 100,  # Máximo de detecciones a mostrar en visualización
    'MAX_ETIQUETAS_VISUALIZAR': 20  # Máximo de etiquetas de texto a mostrar
}


def cargar_template(ruta_template: str) -> np.ndarray:
    """Carga y preprocesa el template usando método Canny."""
    template = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"No se pudo cargar el template desde {ruta_template}")
    
    template_blur = cv2.GaussianBlur(template, CONFIG['KERNEL_GAUSS'], CONFIG['SIGMA_GAUSS'])
    return cv2.Canny(template_blur, *CONFIG['UMBRAL_CANNY'])


def preprocesar_imagen(ruta_imagen: str) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesa una imagen usando el método Canny."""
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    imagen_blur = cv2.GaussianBlur(imagen, CONFIG['KERNEL_GAUSS'], CONFIG['SIGMA_GAUSS'])
    return imagen, cv2.Canny(imagen_blur, *CONFIG['UMBRAL_CANNY'])


def redimensionar_template(template: np.ndarray, escala: float) -> np.ndarray:
    """Redimensiona el template."""
    nuevo_ancho = max(1, int(template.shape[1] * escala))
    nuevo_alto = max(1, int(template.shape[0] * escala))
    try:
        return cv2.resize(template, (nuevo_ancho, nuevo_alto))
    except Exception:
        return None


def procesar_escala_individual_multi(args):
    """Procesa una escala individual para template matching optimizado para múltiples detecciones."""
    escala, imagen_procesada, template_procesado, metodo_matching, umbral_simple = args
    
    detecciones_escala = []
    mapas_escala = []
    
    template_escalado = redimensionar_template(template_procesado, escala)

    if template_escalado is None:
        mapa_error = np.ones((1, 1), dtype=np.float32) * 999.0
        mapas_escala.append((mapa_error, escala, "error_redimension"))
        return detecciones_escala, mapas_escala

    try:
        resultado = cv2.matchTemplate(imagen_procesada, template_escalado, metodo_matching)
        
        # No normalizar para permitir comparación entre escalas en early stopping
        mapas_escala.append((resultado, escala, "directo"))

        # Usar umbral configurado directamente
        ubicaciones = np.where(resultado >= umbral_simple)
        
        # Comprensión de lista más concisa para crear detecciones
        detecciones_escala = [
            {
                'x': int(x), 'y': int(y),
                'ancho': template_escalado.shape[1], 'alto': template_escalado.shape[0],
                'confianza': float(resultado[y, x]), 'escala': escala,
                'centro_x': int(x + template_escalado.shape[1] / 2),
                'centro_y': int(y + template_escalado.shape[0] / 2)
            }
            for y, x in zip(ubicaciones[0], ubicaciones[1])
            if not (np.isnan(resultado[y, x]) or np.isinf(resultado[y, x]))
        ]

    except Exception:
        mapa_error = np.ones((1, 1), dtype=np.float32) * 999.0
        mapas_escala.append((mapa_error, escala, "error_matching"))
    
    return detecciones_escala, mapas_escala


def aplicar_nms_por_escala(detecciones_escala: List[Dict], max_detecciones: int = None) -> List[Dict]:
    """Aplica NMS dentro de una escala específica para reducir redundancia."""
    if not detecciones_escala:
        return []
    
    # Usar valor por defecto de configuración si no se especifica
    if max_detecciones is None:
        max_detecciones = CONFIG.get('MAX_DETECCIONES_POR_ESCALA', 100)
    
    detecciones_ordenadas = sorted(detecciones_escala, key=lambda x: x['confianza'], reverse=True)
    detecciones_filtradas = []
    
    for deteccion in detecciones_ordenadas:
        if len(detecciones_filtradas) >= max_detecciones:
            break
            
        # Expresión generadora más eficiente para verificar solapamiento
        if not any(calcular_iou(deteccion, det) > CONFIG['UMBRAL_IOU_NMS'] 
                  for det in detecciones_filtradas):
            detecciones_filtradas.append(deteccion)
    
    return detecciones_filtradas


def calcular_iou(det1: Dict, det2: Dict) -> float:
    """Calcula Intersection over Union entre dos detecciones."""
    # Coordenadas más concisas usando desempaquetado
    x1_min, y1_min, x1_max, y1_max = det1['x'], det1['y'], det1['x'] + det1['ancho'], det1['y'] + det1['alto']
    x2_min, y2_min, x2_max, y2_max = det2['x'], det2['y'], det2['x'] + det2['ancho'], det2['y'] + det2['alto']
    
    # Cálculo en una línea usando max y min
    area_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    area_union = det1['ancho'] * det1['alto'] + det2['ancho'] * det2['alto'] - area_inter
    
    return area_inter / area_union if area_union > 0 else 0.0


def buscar_coincidencias_multiescala_multi(imagen_procesada: np.ndarray,
                                          template_procesado: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
    """Realiza búsqueda de template en múltiples escalas con NMS por escala y early stopping optimizado para múltiples detecciones."""
    detecciones = []
    mapas_resultado = []

    # Generar escalas de mayor a menor para early stopping (como en template_matching_canny.py)
    escalas = np.arange(CONFIG['ESCALA_MAX'], CONFIG['ESCALA_MIN'] - CONFIG['PASO_ESCALA'], -CONFIG['PASO_ESCALA'])
    
    # Variables para early stopping
    mejor_confianza_global = -1.0
    escala_sin_mejora = 0
    escalas_procesadas = 0
    
    for escala in tqdm(escalas, desc="Procesando escalas", unit="escala"):
        # Verificar si el template escalado es más grande que la imagen
        nuevo_ancho, nuevo_alto = int(template_procesado.shape[1] * escala), int(template_procesado.shape[0] * escala)
        
        if nuevo_ancho > imagen_procesada.shape[1] or nuevo_alto > imagen_procesada.shape[0]:
            # Agregar mapa sintético pero no contar para early stopping
            mapas_resultado.append((np.array([[1.0]], dtype=np.float32), escala, "error_tamaño"))
            print(f"Escala {escala:.2f}x: Template demasiado grande ({nuevo_ancho}x{nuevo_alto}), saltando")
            continue
        
        detecciones_escala, mapas_escala = procesar_escala_individual_multi(
            (escala, imagen_procesada, template_procesado, CONFIG['METODO_MATCHING'], CONFIG['UMBRAL_DETECCION'])
        )
        
        # Aplicar NMS dentro de esta escala específica
        detecciones_escala_filtradas = aplicar_nms_por_escala(detecciones_escala, max_detecciones=CONFIG['MAX_DETECCIONES_POR_ESCALA'])
        
        detecciones.extend(detecciones_escala_filtradas)
        mapas_resultado.extend(mapas_escala)
        escalas_procesadas += 1
        
        # Obtener la mejor confianza de esta escala (del mapa de correlación)
        mejor_confianza_actual = -1.0
        if mapas_escala and len(mapas_escala) > 0:
            # Obtener el mapa de correlación (primer elemento del primer mapa)
            mapa_correlacion = mapas_escala[0][0]
            if mapa_correlacion.size > 1:  # No es un mapa de error
                mejor_confianza_actual = float(mapa_correlacion.max())
        
        print(f"Escala {escala:.2f}x: Confianza max = {mejor_confianza_actual:.4f}, {len(detecciones_escala)} → {len(detecciones_escala_filtradas)} detecciones (NMS)")
        
        # Verificar early stopping (solo después de procesar al menos 1 escala)
        if escalas_procesadas > 1:
            print(f"  Comparando: actual={mejor_confianza_actual:.4f} vs mejor_global={mejor_confianza_global:.4f}")
            
            # Actualizar el mejor global si es necesario
            if mejor_confianza_actual > mejor_confianza_global:
                mejor_confianza_global = mejor_confianza_actual
                escala_sin_mejora = 0  # Reset contador si hay nueva mejor
                print(f"  ¡Nuevo máximo global! Reset contador")
            else:
                escala_sin_mejora += 1
                print(f"  Sin mejora global: {escala_sin_mejora}/{CONFIG['EARLY_STOPPING_ESCALAS']}")
                if escala_sin_mejora >= CONFIG['EARLY_STOPPING_ESCALAS']:
                    print(f"Early stopping: Sin mejora global en {escala_sin_mejora} escalas consecutivas")
                    print(f"Mejor confianza global: {mejor_confianza_global:.4f}")
                    print(f"Última escala procesada: {escala:.2f}x")
                    break
        else:
            # Primera escala procesada
            mejor_confianza_global = mejor_confianza_actual
    

    # Ordenar mapas por escala (de menor a mayor para visualización)
    mapas_resultado.sort(key=lambda x: x[1])
    
    print(f"Escalas procesadas: {escalas_procesadas} de {len(escalas)} totales")
    print(f"Total de detecciones después del NMS por escala: {len(detecciones)}")
    
    return detecciones, mapas_resultado


def agrupar_detecciones_por_clustering(detecciones: List[Dict]) -> List[List[Dict]]:
    """Agrupa detecciones cercanas usando clustering DBSCAN."""
    if len(detecciones) < 2:
        return [detecciones] if detecciones else []
    
    # Extraer coordenadas usando comprensión de lista
    coordenadas = np.array([[det['centro_x'], det['centro_y']] for det in detecciones])
    
    # Aplicar DBSCAN y agrupar en un solo paso
    clustering = DBSCAN(eps=CONFIG['CLUSTERING_EPS'], min_samples=CONFIG['CLUSTERING_MIN'])
    etiquetas = clustering.fit_predict(coordenadas)
    
    # Agrupar usando comprensión de diccionario
    clusters = {}
    for i, etiqueta in enumerate(etiquetas):
        clusters.setdefault(etiqueta, []).append(detecciones[i])
    
    grupos = list(clusters.values())
    print(f"Clustering: {len(detecciones)} detecciones agrupadas en {len(grupos)} clusters")
    
    return grupos


def normalizar_detecciones_globalmente(detecciones: List[Dict]) -> List[Dict]:
    """
    Normaliza las confianzas de todas las detecciones de 0 a 1 basado en 
    los valores mínimo y máximo globales de todas las escalas.
    """
    if not detecciones:
        return []
    
    # Extraer todas las confianzas
    confianzas = [det['confianza'] for det in detecciones]
    
    if not confianzas:
        return []
    
    # Calcular min y max globales
    confianza_min = min(confianzas)
    confianza_max = max(confianzas)
    
    print(f"Normalización global: min={confianza_min:.4f}, max={confianza_max:.4f}")
    
    # Evitar división por cero
    if confianza_max == confianza_min:
        # Si todas las confianzas son iguales, asignar 0.5 a todas
        for det in detecciones:
            det['confianza_original'] = det['confianza']
            det['confianza'] = 0.5
        print("Todas las confianzas son iguales, asignando 0.5 a todas")
        return detecciones
    
    # Normalizar cada detección
    detecciones_normalizadas = []
    for det in detecciones:
        det_normalizada = det.copy()
        det_normalizada['confianza_original'] = det['confianza']
        det_normalizada['confianza'] = (det['confianza'] - confianza_min) / (confianza_max - confianza_min)
        detecciones_normalizadas.append(det_normalizada)
    
    print(f"Detecciones normalizadas: {len(detecciones_normalizadas)}")
    
    return detecciones_normalizadas


def aplicar_nms_multi_deteccion(detecciones: List[Dict]) -> List[Dict]:
    """Aplica NMS optimizado para múltiples detecciones entre escalas con normalización global."""
    if not detecciones:
        return []
    
    print(f"Detecciones antes del filtrado: {len(detecciones)}")
    
    # PASO 1: Normalizar todas las detecciones globalmente
    detecciones_normalizadas = normalizar_detecciones_globalmente(detecciones)
    
    # PASO 2: Filtrar por umbral de confianza normalizada
    umbral_normalizado = CONFIG.get('UMBRAL_CONFIANZA_NORMALIZADA', CONFIG['UMBRAL_CONFIANZA_NORMALIZADA'])
    detecciones_candidatas = [d for d in detecciones_normalizadas if d['confianza'] >= umbral_normalizado]
    
    if not detecciones_candidatas:
        # Si no hay suficientes con el umbral, tomar las mejores
        detecciones_candidatas = sorted(detecciones_normalizadas, key=lambda x: x['confianza'], reverse=True)[:CONFIG['MAX_CANDIDATOS']]
        print(f"No hay detecciones sobre el umbral {umbral_normalizado}, tomando las {len(detecciones_candidatas)} mejores")
    
    print(f"NMS entre escalas: {len(detecciones_candidatas)} candidatos después del filtrado por confianza normalizada (umbral: {umbral_normalizado})")
    
    # PASO 3: Aplicar NMS entre escalas diferentes
    detecciones_candidatas_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)
    detecciones_inter_escala = []
    
    for deteccion in detecciones_candidatas_ordenadas:
        if not any(calcular_iou(deteccion, det) > CONFIG['UMBRAL_IOU_NMS'] for det in detecciones_inter_escala):
            detecciones_inter_escala.append(deteccion)
    
    print(f"NMS entre escalas: {len(detecciones_inter_escala)} candidatos después de filtrar solapamientos")
    
    # PASO 4: Agrupar detecciones restantes por clustering espacial
    grupos_detecciones = agrupar_detecciones_por_clustering(detecciones_inter_escala)
    
    detecciones_finales = []
    
    # PASO 5: Aplicar NMS refinado dentro de cada grupo
    for i, grupo in enumerate(grupos_detecciones):
        print(f"Procesando grupo {i+1}: {len(grupo)} detecciones")
        
        grupo_ordenado = sorted(grupo, key=lambda x: x['confianza'], reverse=True)
        detecciones_grupo = []
        
        while grupo_ordenado and len(detecciones_grupo) < CONFIG['MAX_DETECCIONES_POR_GRUPO']:
            mejor = grupo_ordenado.pop(0)
            detecciones_grupo.append(mejor)
            
            # Filtrar detecciones muy cercanas usando comprensión de lista
            grupo_ordenado = [det for det in grupo_ordenado if calcular_iou(mejor, det) < CONFIG['UMBRAL_IOU_NMS']]
        
        detecciones_finales.extend(detecciones_grupo)
    
    # PASO 6: Ordenar y limitar resultado final
    detecciones_finales = sorted(detecciones_finales, key=lambda x: x['confianza'], reverse=True)[:CONFIG['LIMITE_FINAL']]
    
    print(f"NMS final: {len(detecciones_finales)} detecciones seleccionadas")
    
    return detecciones_finales


def visualizar_preprocesamiento(template_procesado: np.ndarray,
                                imagen_procesada: np.ndarray,
                                nombre_imagen: str):
    """Visualiza las entradas al algoritmo de matching."""
    os.makedirs(CONFIG['CARPETA_RESULTADOS'], exist_ok=True)
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=CONFIG['DPI_FIGURA'])
    fig.suptitle(f'ENTRADAS AL ALGORITMO - {nombre_imagen}', fontsize=16, weight='bold')

    # Usar zip para iteración más concisa
    for ax, (img, titulo, color) in zip(axes, [
        (imagen_procesada, 'IMAGEN PROCESADA', 'green'),
        (template_procesado, 'TEMPLATE PROCESADO', 'blue')
    ]):
        ax.imshow(img, cmap='gray')
        ax.set_title(titulo, fontsize=12, weight='bold', color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{CONFIG["CARPETA_RESULTADOS"]}/{nombre_base}_01_entradas_algoritmo.png',
                bbox_inches='tight', dpi=CONFIG['DPI_FIGURA'])
    plt.close()


def visualizar_mapas_coincidencias(mapas_resultado: List[Tuple], nombre_imagen: str):
    """Visualiza mapas de matching en un único plot con soporte para grids más grandes."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    if not mapas_resultado:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=CONFIG['DPI_FIGURA'])
        ax.text(0.5, 0.5, 'NO SE GENERARON MAPAS',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
        ax.set_title(f'MAPAS DE MATCHING - {nombre_imagen}', fontsize=16, weight='bold')
        ax.axis('off')
        plt.savefig(f'{CONFIG["CARPETA_RESULTADOS"]}/{nombre_base}_02_mapas_matching.png',
                    bbox_inches='tight', dpi=CONFIG['DPI_FIGURA'])
        plt.close()
        return

    num_mapas = len(mapas_resultado)
    
    # Determinar layout adaptativo para grids más grandes
    if num_mapas <= 4:
        filas, cols = 1, num_mapas
    elif num_mapas <= 8:
        filas, cols = 2, 4
    elif num_mapas <= 12:
        filas, cols = 3, 4
    elif num_mapas <= 16:
        filas, cols = 4, 4
    elif num_mapas <= 20:
        filas, cols = 4, 5
    elif num_mapas <= 24:
        filas, cols = 4, 6
    elif num_mapas <= 30:
        filas, cols = 5, 6
    elif num_mapas <= 36:
        filas, cols = 6, 6
    elif num_mapas <= 42:
        filas, cols = 6, 7
    elif num_mapas <= 48:
        filas, cols = 6, 8
    else:
        filas, cols = 8, 8  # Máximo 64 mapas
        num_mapas = min(num_mapas, 64)

    # Ajustar tamaño de figura dinámicamente
    tamano_subplot = 3  # Tamaño base de cada subplot
    fig, axes = plt.subplots(filas, cols, figsize=(cols * tamano_subplot, filas * tamano_subplot), dpi=CONFIG['DPI_FIGURA'])
    
    # Normalizar axes para iteración
    axes = [axes] if num_mapas == 1 else axes.flatten() if filas > 1 else axes

    fig.suptitle(f'MAPAS DE MATCHING - {nombre_imagen} ({num_mapas} escalas)', fontsize=16, weight='bold')

    for i in range(min(len(mapas_resultado), num_mapas)):
        mapa, escala = mapas_resultado[i][:2]
        ax = axes[i]
        
        if mapa.shape == (1, 1) and mapa[0, 0] > 900:
            ax.text(0.5, 0.5, f'ERROR\nEscala {escala:.1f}x',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=8, weight='bold', color='red')
            ax.set_title(f'Escala {escala:.1f}x - ERROR', fontsize=8, color='red')
        else:
            im = ax.imshow(mapa, cmap='hot', interpolation='nearest')
            ax.set_title(f'Escala {escala:.1f}x\nMax: {mapa.max():.3f}', fontsize=8)
            # Ajustar colorbar para grids grandes
            if cols <= 6:  # Solo mostrar colorbar en grids pequeños para no saturar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.axis('off')

    # Ocultar axes sobrantes
    for i in range(min(len(mapas_resultado), num_mapas), filas * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{CONFIG["CARPETA_RESULTADOS"]}/{nombre_base}_02_mapas_matching.png',
                bbox_inches='tight', dpi=CONFIG['DPI_FIGURA'])
    plt.close()


def visualizar_comparacion_escalas(imagen_original: np.ndarray,
                                 template_original: np.ndarray,
                                 mapas_resultado: List[Tuple],
                                 nombre_imagen: str):
    """Visualiza el template escalado superpuesto en la imagen."""
    if not mapas_resultado:
        return
        
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # Permitir mostrar más escalas adaptivamente
    num_escalas = min(24, len(mapas_resultado))  # Aumentar de 12 a 24
    mapas_ordenados = sorted(mapas_resultado, key=lambda x: x[1])
    
    # Determinar layout adaptativo para más escalas
    if num_escalas <= 3:
        filas, cols = 1, 3
    elif num_escalas <= 6:
        filas, cols = 2, 3
    elif num_escalas <= 9:
        filas, cols = 3, 3
    elif num_escalas <= 12:
        filas, cols = 3, 4
    elif num_escalas <= 16:
        filas, cols = 4, 4
    elif num_escalas <= 20:
        filas, cols = 4, 5
    else:
        filas, cols = 4, 6  # Máximo 24 escalas
    
    # Ajustar tamaño de figura dinámicamente
    tamano_subplot = 4  # Tamaño base por subplot
    fig, axes = plt.subplots(filas, cols, figsize=(cols * tamano_subplot, filas * tamano_subplot), dpi=CONFIG['DPI_FIGURA'])
    axes = [axes] if num_escalas == 1 else axes.flatten() if filas > 1 else axes
    
    fig.suptitle(f'COMPARACIÓN DE ESCALAS - {nombre_imagen} ({num_escalas} escalas)', fontsize=16, weight='bold')
    
    for i in range(num_escalas):
        mapa, escala = mapas_ordenados[i][:2]
        ax = axes[i]
        ax.imshow(imagen_original, cmap='gray', alpha=CONFIG['ALPHA_VISUALIZACION'])
        
        nuevo_ancho, nuevo_alto = int(template_original.shape[1] * escala), int(template_original.shape[0] * escala)
        
        if nuevo_ancho > 0 and nuevo_alto > 0:
            template_escalado = cv2.resize(template_original, (nuevo_ancho, nuevo_alto))
            
            center_x = imagen_original.shape[1] // 2 - nuevo_ancho // 2
            center_y = imagen_original.shape[0] // 2 - nuevo_alto // 2
            
            template_contorno = np.zeros_like(imagen_original)
            if (center_x >= 0 and center_y >= 0 and 
                center_x + nuevo_ancho <= imagen_original.shape[1] and
                center_y + nuevo_alto <= imagen_original.shape[0]):
                template_contorno[center_y:center_y+nuevo_alto, center_x:center_x+nuevo_ancho] = template_escalado
            
            ax.imshow(template_contorno, cmap='Reds', alpha=0.5)
            
            rect = plt.Rectangle((center_x, center_y), nuevo_ancho, nuevo_alto,
                               linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        # Ajustar tamaño de fuente según el número de escalas
        fontsize = 10 if num_escalas <= 12 else 8
        ax.set_title(f'Escala {escala:.1f}x\nTemplate: {nuevo_ancho}x{nuevo_alto}px', fontsize=fontsize)
        ax.axis('off')
    
    # Ocultar axes sobrantes
    for i in range(num_escalas, filas * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["CARPETA_RESULTADOS"]}/{nombre_base}_04_comparacion_escalas.png',
                bbox_inches='tight', dpi=CONFIG['DPI_FIGURA'])
    plt.close()


def visualizar_todas_las_detecciones(imagen_original: np.ndarray,
                                    detecciones_antes_nms: List[Dict],
                                    detecciones_despues_nms: List[Dict],
                                    nombre_imagen: str):
    """Visualiza todas las detecciones antes y después del NMS."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=CONFIG['DPI_FIGURA'])
    
    # Detecciones antes del NMS
    ax1.imshow(imagen_original, cmap='gray')
    ax1.set_title(f'ANTES NMS - {len(detecciones_antes_nms)} detecciones', fontsize=14, weight='bold')
    
    for i, det in enumerate(detecciones_antes_nms[:CONFIG['MAX_DETECCIONES_VISUALIZAR']]):  # Limitar visualización
        x, y, w, h = det['x'], det['y'], det['ancho'], det['alto']
        color = plt.cm.viridis(det['confianza'])[:3]
        
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                               edgecolor=color, facecolor='none', alpha=CONFIG['ALPHA_VISUALIZACION'])
        ax1.add_patch(rect)
        
        if i < CONFIG['MAX_ETIQUETAS_VISUALIZAR']:  # Solo mostrar texto para las primeras N
            ax1.text(x, y-5, f'{det["confianza"]:.2f}', 
                    fontsize=8, color=color, weight='bold')
    
    ax1.axis('off')
    
    # Detecciones después del NMS
    ax2.imshow(imagen_original, cmap='gray')
    ax2.set_title(f'DESPUÉS NMS - {len(detecciones_despues_nms)} detecciones finales', 
                 fontsize=14, weight='bold')
    
    colores = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
    
    for i, det in enumerate(detecciones_despues_nms):
        x, y, w, h = det['x'], det['y'], det['ancho'], det['alto']
        color = colores[i % len(colores)]
        
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                               edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        ax2.text(x, y-10, f'#{i+1}: {det["confianza"]:.3f}', 
                bbox=dict(boxstyle="round,pad={}".format(CONFIG['PADDING_BBOX']), facecolor=color, alpha=0.8),
                fontsize=10, color='white', weight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["CARPETA_RESULTADOS"]}/{nombre_base}_05_comparacion_detecciones.png',
                bbox_inches='tight', dpi=CONFIG['DPI_FIGURA'])
    plt.close()


def visualizar_detecciones_finales_numeradas(imagen_original: np.ndarray,
                                           detecciones_despues_nms: List[Dict],
                                           nombre_imagen: str):
    """Visualiza las detecciones finales con numeración clara."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    plt.figure(figsize=(16, 12), dpi=CONFIG['DPI_FIGURA'])
    plt.imshow(imagen_original, cmap='gray')
    
    colores = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
              'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'maroon', 'teal']
    
    if detecciones_despues_nms:
        for i, det in enumerate(detecciones_despues_nms):
            x, y, w, h = det['x'], det['y'], det['ancho'], det['alto']
            confianza, escala = det['confianza'], det['escala']
            color = colores[i % len(colores)]
            
            # Rectángulo de detección
            rect = plt.Rectangle((x, y), w, h, linewidth=4, 
                               edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Etiqueta con información
            etiqueta = f'#{i+1}\nConf: {confianza:.3f}\nEsc: {escala:.2f}x'
            plt.text(x, y-15, etiqueta, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9),
                    fontsize=11, color='white', weight='bold', ha='left')
            
            # Punto central
            centro_x, centro_y = det['centro_x'], det['centro_y']
            plt.plot(centro_x, centro_y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        titulo = f'DETECCIONES MÚLTIPLES - {nombre_imagen}\n{len(detecciones_despues_nms)} logos detectados'
        
    else:
        titulo = f'SIN DETECCIONES - {nombre_imagen}'
    
    plt.title(titulo, fontsize=16, weight='bold', pad=20)
    plt.axis('off')
    
    plt.savefig(f'{CONFIG["CARPETA_RESULTADOS"]}/{nombre_base}_06_detecciones_finales_numeradas.png',
                bbox_inches='tight', dpi=150)
    plt.close()


def procesar_imagen_multi(ruta_imagen: str, template_data: np.ndarray):
    """Procesa una imagen para múltiples detecciones."""
    nombre_imagen = os.path.basename(ruta_imagen)
    template_procesado = template_data

    print(f"\n{'='*60}")
    print(f"PROCESANDO: {nombre_imagen}")
    print(f"{'='*60}")

    ruta_template = os.path.join(CONFIG['PATH_TEMPLATE'], 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)
    print(f"Imagen cargada: {imagen_original.shape[1]}x{imagen_original.shape[0]} px")

    # Visualizar preprocesamiento
    visualizar_preprocesamiento(template_procesado, imagen_procesada, nombre_imagen)

    # Búsqueda en múltiples escalas
    detecciones, mapas_resultado = buscar_coincidencias_multiescala_multi(imagen_procesada, template_procesado)

    # Visualizar mapas de matching
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)

    print(f"Detecciones antes del filtrado: {len(detecciones)}")

    # Aplicar NMS optimizado para múltiples detecciones
    detecciones_filtradas = aplicar_nms_multi_deteccion(detecciones)

    print(f"Detecciones finales: {len(detecciones_filtradas)}")

    # Generar todas las visualizaciones
    visualizar_comparacion_escalas(imagen_original, template_original, mapas_resultado, nombre_imagen)
    visualizar_todas_las_detecciones(imagen_original, detecciones, detecciones_filtradas, nombre_imagen)
    visualizar_detecciones_finales_numeradas(imagen_original, detecciones_filtradas, nombre_imagen)


    return detecciones_filtradas


def test_algoritmo_todas_imagenes(config=None):
    """
    Función general que testa el algoritmo para todas las imágenes en el folder de images.
    Usa una configuración específica para pruebas generales.
    
    Args:
        config (dict, optional): Configuración a usar. Si es None, usa CONFIG_TEST_GENERAL.
    """
    # Usar configuración específica para tests o la proporcionada
    current_config = config if config is not None else CONFIG_TEST_GENERAL
    
    print("\n" + "=" * 80)
    print("        TEMPLATE MATCHING CANNY - TEST GENERAL EN TODAS LAS IMÁGENES        ")
    print("=" * 80)
    
    print("\nCONFIGURACIÓN PARA TEST GENERAL:")
    print(f"   Método de matching: {current_config['METODO_MATCHING']}")
    print(f"   Umbral de detección: {current_config['UMBRAL_DETECCION']}")
    print(f"   Umbral de confianza normalizada: {current_config['UMBRAL_CONFIANZA_NORMALIZADA']}")
    print(f"   Rango de escalas: {current_config['ESCALA_MIN']}x - {current_config['ESCALA_MAX']}x (paso: {current_config['PASO_ESCALA']})")
    print(f"   Umbral Canny: {current_config['UMBRAL_CANNY'][0]} - {current_config['UMBRAL_CANNY'][1]}")
    print(f"   Clustering eps: {current_config['CLUSTERING_EPS']}")
    print(f"   Límite detecciones: {current_config['LIMITE_FINAL']}")
    print()

    # Crear directorio de resultados
    os.makedirs(current_config['CARPETA_RESULTADOS'], exist_ok=True)

    # Cargar template con la configuración específica
    ruta_template = os.path.join(current_config['PATH_TEMPLATE'], 'pattern.png')
    
    # Modificar temporalmente la configuración global para cargar el template
    config_backup = CONFIG.copy()
    CONFIG.update(current_config)
    
    try:
        template_data = cargar_template(ruta_template)
    finally:
        # Restaurar configuración original
        CONFIG.clear()
        CONFIG.update(config_backup)

    # Obtener todas las imágenes del directorio
    directorio_imagenes = current_config['PATH_IMAGENES']
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    imagenes = [f for f in os.listdir(directorio_imagenes) 
                if f.lower().endswith(extensiones_validas)]
    
    if not imagenes:
        print(f"ERROR: No se encontraron imágenes válidas en {directorio_imagenes}")
        return

    print(f"Se encontraron {len(imagenes)} imágenes para procesar:")
    for img in imagenes:
        print(f"  - {img}")
    print()

    # Resumen de resultados
    resultados_generales = []

    # Procesar cada imagen
    for i, nombre_imagen in enumerate(imagenes, 1):
        print("=" * 60)
        print(f"PROCESANDO ({i}/{len(imagenes)}): {nombre_imagen}")
        print("=" * 60)
        
        ruta_imagen = os.path.join(directorio_imagenes, nombre_imagen)
        
        try:
            # Modificar temporalmente la configuración global
            CONFIG.clear()
            CONFIG.update(current_config)
            
            # Procesar la imagen
            detecciones = procesar_imagen_multi(ruta_imagen, template_data)
            
            # Guardar resultados
            resultado = {
                'imagen': nombre_imagen,
                'detecciones': len(detecciones),
                'mejor_confianza': max([d['confianza'] for d in detecciones]) if detecciones else 0.0,
                'escalas_detectadas': list(set([d['escala'] for d in detecciones])) if detecciones else [],
                'status': 'Exitoso'
            }
            resultados_generales.append(resultado)
            
            print(f"✓ Completado: {len(detecciones)} detecciones encontradas")
            if detecciones:
                print(f"  Mejor confianza: {resultado['mejor_confianza']:.3f}")
                print(f"  Escalas detectadas: {len(resultado['escalas_detectadas'])}")
            
        except Exception as e:
            print(f"✗ Error procesando {nombre_imagen}: {str(e)}")
            resultado = {
                'imagen': nombre_imagen,
                'detecciones': 0,
                'mejor_confianza': 0.0,
                'escalas_detectadas': [],
                'status': f'Error: {str(e)}'
            }
            resultados_generales.append(resultado)
        
        finally:
            # Restaurar configuración original
            CONFIG.clear()
            CONFIG.update(config_backup)

    # Mostrar resumen final
    print("\n" + "=" * 80)
    print("                          RESUMEN GENERAL FINAL                              ")
    print("=" * 80)
    
    exitosos = sum(1 for r in resultados_generales if r['status'] == 'Exitoso')
    total_detecciones = sum(r['detecciones'] for r in resultados_generales)
    
    print(f"Imágenes procesadas: {len(imagenes)}")
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
    
    print(f"\nResultados guardados en: {current_config['CARPETA_RESULTADOS']}")
    print("Test general completado!")
    print("=" * 80)
    
    return resultados_generales


def main():
    """Función principal para procesar coca_multi.png con configuración específica."""
    print("=" * 80)
    print("        TEMPLATE MATCHING CANNY - DETECCIÓN MÚLTIPLE DE LOGOS        ")
    print("=" * 80)
    print("\nCONFIGURACIÓN OPTIMIZADA PARA MÚLTIPLES DETECCIONES:")
    print(f"   Método de matching: {CONFIG['METODO_MATCHING']}")
    print(f"   Umbral de detección: {CONFIG['UMBRAL_DETECCION']}")
    print(f"   Umbral de confianza normalizada: {CONFIG['UMBRAL_CONFIANZA_NORMALIZADA']}")
    print(f"   Rango de escalas: {CONFIG['ESCALA_MIN']}x - {CONFIG['ESCALA_MAX']}x (paso: {CONFIG['PASO_ESCALA']})")
    print(f"   Umbral Canny: {CONFIG['UMBRAL_CANNY'][0]} - {CONFIG['UMBRAL_CANNY'][1]}")
    print(f"   Clustering eps: {CONFIG['CLUSTERING_EPS']}")
    print(f"   Límite detecciones: {CONFIG['LIMITE_FINAL']}")
    print()

    os.makedirs(CONFIG['CARPETA_RESULTADOS'], exist_ok=True)

    # Cargar template
    ruta_template = os.path.join(CONFIG['PATH_TEMPLATE'], 'pattern.png')
    template_data = cargar_template(ruta_template)

    # Procesar específicamente la imagen coca_multi.png
    ruta_imagen_multi = os.path.join(CONFIG['PATH_IMAGENES'], 'coca_multi.png')
    
    if not os.path.exists(ruta_imagen_multi):
        print(f"ERROR: No se encontró la imagen {ruta_imagen_multi}")
        return

    detecciones = procesar_imagen_multi(ruta_imagen_multi, template_data)

    print("\n" + "=" * 80)
    print("                          RESUMEN FINAL                              ")
    print("=" * 80)
    print(f"Imagen procesada: coca_multi.png")
    print(f"Detecciones encontradas: {len(detecciones)}")
    
    if detecciones:
        print("\nDETALLE DE DETECCIONES (Normalizada | Original):")
        for i, det in enumerate(detecciones):
            confianza_original = det.get('confianza_original', 'N/A')
            if confianza_original != 'N/A':
                print(f"  #{i+1}: Confianza {det['confianza']:.3f} | {confianza_original:.4f}, "
                      f"Escala {det['escala']:.2f}x, "
                      f"Posición ({det['x']}, {det['y']})")
            else:
                print(f"  #{i+1}: Confianza {det['confianza']:.3f}, "
                      f"Escala {det['escala']:.2f}x, "
                      f"Posición ({det['x']}, {det['y']})")
    
    print(f"\nResultados guardados en: {CONFIG['CARPETA_RESULTADOS']}")
    print("Proceso de detección múltiple completado exitosamente!")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # Si se pasa "test" como argumento, ejecutar test general
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        test_algoritmo_todas_imagenes()
    else:
        main()