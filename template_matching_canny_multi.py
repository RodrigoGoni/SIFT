import cv2
import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
import datetime

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configuración de parámetros optimizada para múltiples detecciones
PATH_IMAGENES = 'TP3/images/'
PATH_TEMPLATE = 'TP3/template/'
METODO_MATCHING = cv2.TM_CCOEFF_NORMED
ESCALA_MIN = 0.24
ESCALA_MAX = 0.25
PASO_ESCALA = 0.01
UMBRAL_CANNY_MIN = 150
UMBRAL_CANNY_MAX = 250
FILTRAR_RUIDO = True
KERNEL_GAUSSIANO = (3, 3)
SIGMA_GAUSSIANO = 1.7
UMBRAL_SIMPLE_DETECCION = 0.118
UMBRAL_IOU_NMS = 0.2
MAXIMO_MEJORES_CANDIDATOS = 50
LIMITE_DETECCIONES_FINALES = 50
DPI_FIGURA = 100
CARPETA_RESULTADOS = 'resultados_canny_multi'
EARLY_STOPPING_ESCALAS = 5

# Parámetros específicos para múltiples detecciones
CLUSTERING_EPS = 15  # Distancia máxima para agrupar detecciones (más estricto)
CLUSTERING_MIN_SAMPLES = 1  # Mínimo de muestras por cluster
UMBRAL_CONFIANZA_MINIMA = 0.118# Filtro más alto para reducir candidatos


def cargar_template(ruta_template: str) -> np.ndarray:
    """Carga y preprocesa el template usando método Canny."""
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)
    if template_original is None:
        raise ValueError(f"No se pudo cargar el template desde {ruta_template}")
    
    if FILTRAR_RUIDO:
        template_filtrado = cv2.GaussianBlur(template_original, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
    else:
        template_filtrado = template_original
    
    template_procesado = cv2.Canny(template_filtrado, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX)
    
    return template_procesado


def preprocesar_imagen(ruta_imagen: str) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesa una imagen usando el método Canny."""
    imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen_original is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    if FILTRAR_RUIDO:
        imagen_filtrada = cv2.GaussianBlur(imagen_original, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
    else:
        imagen_filtrada = imagen_original
    
    imagen_procesada = cv2.Canny(imagen_filtrada, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX)
    
    return imagen_original, imagen_procesada


def redimensionar_template(template: np.ndarray, escala: float) -> np.ndarray:
    """Redimensiona el template."""
    nuevo_ancho = max(1, int(template.shape[1] * escala))
    nuevo_alto = max(1, int(template.shape[0] * escala))

    try:
        template_redimensionado = cv2.resize(template, (nuevo_ancho, nuevo_alto))
        return template_redimensionado
    except Exception:
        return None


def procesar_escala_individual_multi(args):
    """Procesa una escala individual para template matching optimizado para múltiples detecciones."""
    (escala, imagen_procesada, template_procesado, 
     metodo_matching, umbral_simple) = args
    
    detecciones_escala = []
    mapas_escala = []
    
    template_escalado = redimensionar_template(template_procesado, escala)

    if template_escalado is None:
        mapa_error = np.ones((1, 1), dtype=np.float32) * 999.0
        mapas_escala.append((mapa_error, escala, "error_redimension"))
        return detecciones_escala, mapas_escala

    try:
        resultado = cv2.matchTemplate(imagen_procesada, template_escalado, metodo_matching)
        mapas_escala.append((resultado, escala, "directo"))

        # Usar umbral más bajo para capturar más candidatos
        umbral_reducido = max(0.05, umbral_simple * 0.7)
        ubicaciones = np.where(resultado >= umbral_reducido)
        
        for y, x in zip(ubicaciones[0], ubicaciones[1]):
            confianza = float(resultado[y, x])
            if np.isnan(confianza) or np.isinf(confianza):
                continue

            detecciones_escala.append({
                'x': int(x),
                'y': int(y),
                'ancho': template_escalado.shape[1],
                'alto': template_escalado.shape[0],
                'confianza': confianza,
                'escala': escala,
                'centro_x': int(x + template_escalado.shape[1] / 2),
                'centro_y': int(y + template_escalado.shape[0] / 2)
            })

    except Exception:
        mapa_error = np.ones((1, 1), dtype=np.float32) * 999.0
        mapas_escala.append((mapa_error, escala, "error_matching"))
    
    return detecciones_escala, mapas_escala


def aplicar_nms_por_escala(detecciones_escala: List[Dict], max_detecciones: int = 100) -> List[Dict]:
    """Aplica NMS dentro de una escala específica para reducir redundancia."""
    if not detecciones_escala:
        return []
    
    # Ordenar por confianza
    detecciones_ordenadas = sorted(detecciones_escala, key=lambda x: x['confianza'], reverse=True)
    
    detecciones_filtradas = []
    
    for deteccion in detecciones_ordenadas:
        if len(detecciones_filtradas) >= max_detecciones:
            break
            
        # Verificar solapamiento con detecciones ya seleccionadas
        mantener = True
        for det_existente in detecciones_filtradas:
            iou = calcular_iou(deteccion, det_existente)
            if iou > UMBRAL_IOU_NMS:  # Si hay mucho solapamiento, descartar
                mantener = False
                break
        
        if mantener:
            detecciones_filtradas.append(deteccion)
    
    return detecciones_filtradas


def calcular_iou(det1: Dict, det2: Dict) -> float:
    """Calcula Intersection over Union entre dos detecciones."""
    x1_min, y1_min = det1['x'], det1['y']
    x1_max, y1_max = det1['x'] + det1['ancho'], det1['y'] + det1['alto']
    
    x2_min, y2_min = det2['x'], det2['y']
    x2_max, y2_max = det2['x'] + det2['ancho'], det2['y'] + det2['alto']
    
    # Calcular intersección
    x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    area_inter = x_inter * y_inter
    
    # Calcular áreas
    area1 = det1['ancho'] * det1['alto']
    area2 = det2['ancho'] * det2['alto']
    area_union = area1 + area2 - area_inter
    
    return area_inter / area_union if area_union > 0 else 0.0


def buscar_coincidencias_multiescala_multi(imagen_procesada: np.ndarray,
                                          template_procesado: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
    """Realiza búsqueda de template en múltiples escalas con NMS por escala optimizado para múltiples detecciones."""
    detecciones = []
    mapas_resultado = []

    escalas = np.arange(ESCALA_MIN, ESCALA_MAX + PASO_ESCALA, PASO_ESCALA)
    
    for escala in tqdm(escalas, desc="Procesando escalas", unit="escala"):
        # Verificar si el template escalado es más grande que la imagen
        nuevo_ancho = int(template_procesado.shape[1] * escala)
        nuevo_alto = int(template_procesado.shape[0] * escala)
        
        if (nuevo_ancho > imagen_procesada.shape[1] or nuevo_alto > imagen_procesada.shape[0]):
            mapa_sintetico = np.array([[1.0]], dtype=np.float32)
            mapas_resultado.append((mapa_sintetico, escala, "error_tamaño"))
            continue
        
        detecciones_escala, mapas_escala = procesar_escala_individual_multi(
            (escala, imagen_procesada, template_procesado, 
             METODO_MATCHING, UMBRAL_SIMPLE_DETECCION)
        )
        
        # Aplicar NMS dentro de esta escala específica
        detecciones_escala_filtradas = aplicar_nms_por_escala(detecciones_escala, max_detecciones=200)
        
        detecciones.extend(detecciones_escala_filtradas)
        mapas_resultado.extend(mapas_escala)
        
        # Información de progreso
        if mapas_escala and len(mapas_escala) > 0:
            mapa_correlacion = mapas_escala[0][0]
            if mapa_correlacion.size > 1:
                mejor_confianza = float(mapa_correlacion.max())
                num_detecciones_original = len(detecciones_escala)
                num_detecciones_filtradas = len(detecciones_escala_filtradas)
                print(f"Escala {escala:.2f}x: {num_detecciones_original} → {num_detecciones_filtradas} detecciones (NMS), confianza max = {mejor_confianza:.4f}")

    mapas_resultado.sort(key=lambda x: x[1])
    print(f"Total de detecciones después del NMS por escala: {len(detecciones)}")
    
    return detecciones, mapas_resultado


def agrupar_detecciones_por_clustering(detecciones: List[Dict]) -> List[List[Dict]]:
    """Agrupa detecciones cercanas usando clustering DBSCAN."""
    if len(detecciones) < 2:
        return [detecciones] if detecciones else []
    
    # Extraer coordenadas de centros para clustering
    coordenadas = np.array([[det['centro_x'], det['centro_y']] for det in detecciones])
    
    # Aplicar DBSCAN
    clustering = DBSCAN(eps=CLUSTERING_EPS, min_samples=CLUSTERING_MIN_SAMPLES)
    etiquetas = clustering.fit_predict(coordenadas)
    
    # Agrupar detecciones por clusters
    clusters = {}
    for i, etiqueta in enumerate(etiquetas):
        if etiqueta not in clusters:
            clusters[etiqueta] = []
        clusters[etiqueta].append(detecciones[i])
    
    # Convertir a lista de grupos
    grupos = list(clusters.values())
    
    print(f"Clustering: {len(detecciones)} detecciones agrupadas en {len(grupos)} clusters")
    
    return grupos


def aplicar_nms_multi_deteccion(detecciones: List[Dict]) -> List[Dict]:
    """Aplica NMS optimizado para múltiples detecciones entre escalas."""
    if not detecciones:
        return []
    
    # Filtrar por umbral de confianza mínima
    detecciones_candidatas = [d for d in detecciones if d['confianza'] >= UMBRAL_CONFIANZA_MINIMA]
    
    if not detecciones_candidatas:
        # Si no hay suficientes con el umbral, tomar las mejores
        detecciones_candidatas = sorted(detecciones, key=lambda x: x['confianza'], reverse=True)[:MAXIMO_MEJORES_CANDIDATOS]
    
    print(f"NMS entre escalas: {len(detecciones_candidatas)} candidatos después del filtrado por confianza")
    
    # Aplicar NMS entre escalas diferentes
    detecciones_candidatas_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)
    detecciones_inter_escala = []
    
    for deteccion in detecciones_candidatas_ordenadas:
        mantener = True
        for det_existente in detecciones_inter_escala:
            iou = calcular_iou(deteccion, det_existente)
            # Umbral más estricto para detecciones entre escalas
            if iou > UMBRAL_IOU_NMS :
                mantener = False
                break
        
        if mantener:
            detecciones_inter_escala.append(deteccion)
    
    print(f"NMS entre escalas: {len(detecciones_inter_escala)} candidatos después de filtrar solapamientos")
    
    # Agrupar detecciones restantes por clustering espacial
    grupos_detecciones = agrupar_detecciones_por_clustering(detecciones_inter_escala)
    
    detecciones_finales = []
    
    # Aplicar NMS refinado dentro de cada grupo
    for i, grupo in enumerate(grupos_detecciones):
        print(f"Procesando grupo {i+1}: {len(grupo)} detecciones")
        
        # Ordenar por confianza dentro del grupo
        grupo_ordenado = sorted(grupo, key=lambda x: x['confianza'], reverse=True)
        detecciones_grupo = []
        
        while grupo_ordenado and len(detecciones_grupo) < 8:  # Máximo 8 detecciones por grupo (reducido)
            mejor = grupo_ordenado.pop(0)
            detecciones_grupo.append(mejor)
            
            # Filtrar detecciones muy cercanas a la seleccionada
            grupo_filtrado = []
            for det in grupo_ordenado:
                iou = calcular_iou(mejor, det)
                
                if iou < UMBRAL_IOU_NMS:
                    grupo_filtrado.append(det)
            
            grupo_ordenado = grupo_filtrado
        
        detecciones_finales.extend(detecciones_grupo)
    
    # Ordenar resultado final por confianza
    detecciones_finales = sorted(detecciones_finales, key=lambda x: x['confianza'], reverse=True)
    
    # Limitar número total de detecciones
    detecciones_finales = detecciones_finales[:LIMITE_DETECCIONES_FINALES]
    
    print(f"NMS final: {len(detecciones_finales)} detecciones seleccionadas")
    
    return detecciones_finales


def visualizar_preprocesamiento(template_procesado: np.ndarray,
                                imagen_procesada: np.ndarray,
                                nombre_imagen: str):
    """Visualiza las entradas al algoritmo de matching."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=DPI_FIGURA)
    fig.suptitle(f'ENTRADAS AL ALGORITMO - {nombre_imagen}', fontsize=16, weight='bold')

    axes[0].imshow(imagen_procesada, cmap='gray')
    axes[0].set_title('IMAGEN PROCESADA', fontsize=12, weight='bold', color='green')
    axes[0].axis('off')

    axes[1].imshow(template_procesado, cmap='gray')
    axes[1].set_title('TEMPLATE PROCESADO', fontsize=12, weight='bold', color='blue')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_01_entradas_algoritmo.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_mapas_coincidencias(mapas_resultado: List[Tuple], nombre_imagen: str):
    """Visualiza mapas de matching en un único plot."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    if not mapas_resultado:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=DPI_FIGURA)
        ax.text(0.5, 0.5, 'NO SE GENERARON MAPAS',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
        ax.set_title(f'MAPAS DE MATCHING - {nombre_imagen}', fontsize=16, weight='bold')
        ax.axis('off')
        plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_02_mapas_matching.png',
                    bbox_inches='tight', dpi=DPI_FIGURA)
        plt.close()
        return

    num_mapas = len(mapas_resultado)
    
    if num_mapas <= 4:
        filas, cols = 1, num_mapas
    elif num_mapas <= 8:
        filas, cols = 2, 4
    elif num_mapas <= 12:
        filas, cols = 3, 4
    else:
        filas, cols = 4, 4
        num_mapas = 16

    fig, axes = plt.subplots(filas, cols, figsize=(cols * 4, filas * 3), dpi=DPI_FIGURA)
    
    if num_mapas == 1:
        axes = [axes]
    elif filas == 1:
        axes = axes
    else:
        axes = axes.flatten()

    fig.suptitle(f'MAPAS DE MATCHING - {nombre_imagen}', fontsize=16, weight='bold')

    for i in range(min(len(mapas_resultado), num_mapas)):
        mapa, escala = mapas_resultado[i][:2]
        ax = axes[i]
        
        if mapa.shape == (1, 1) and mapa[0, 0] > 900:
            ax.text(0.5, 0.5, f'ERROR\nEscala {escala:.1f}x',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, weight='bold', color='red')
            ax.set_title(f'Escala {escala:.1f}x - ERROR', fontsize=10, color='red')
        else:
            im = ax.imshow(mapa, cmap='hot', interpolation='nearest')
            ax.set_title(f'Escala {escala:.1f}x\nMax: {mapa.max():.3f}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.axis('off')

    for i in range(min(len(mapas_resultado), num_mapas), filas * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_02_mapas_matching.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_resultado_final(imagen_original: np.ndarray,
                               detecciones_despues_nms: List[Dict],
                               nombre_imagen: str):
    """Visualiza la mejor detección del NMS."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(imagen_original, cmap='gray')
    
    if detecciones_despues_nms:
        mejor_det = detecciones_despues_nms[0]
        x, y = mejor_det['x'], mejor_det['y']
        w, h = mejor_det['ancho'], mejor_det['alto']
        confianza = mejor_det['confianza']
        
        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        
        plt.text(x, y-10, f'Mejor: {confianza:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                fontsize=12, color='black', weight='bold')
        
        titulo = f'MEJOR DETECCIÓN - {nombre_imagen}\nConfianza: {confianza:.3f} | Escala: {mejor_det["escala"]:.1f}x'
    else:
        titulo = f'SIN DETECCIONES - {nombre_imagen}'
    
    plt.title(titulo, fontsize=14, weight='bold')
    plt.axis('off')
    
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_03_mejor_deteccion.png',
                bbox_inches='tight', dpi=150)
    plt.close()


def visualizar_comparacion_escalas(imagen_original: np.ndarray,
                                 template_original: np.ndarray,
                                 mapas_resultado: List[Tuple],
                                 nombre_imagen: str):
    """Visualiza el template escalado superpuesto en la imagen."""
    if not mapas_resultado:
        return
        
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    num_escalas = min(12, len(mapas_resultado))
    mapas_ordenados = sorted(mapas_resultado, key=lambda x: x[1])
    
    if num_escalas <= 3:
        filas, cols = 1, 3
    elif num_escalas <= 6:
        filas, cols = 2, 3
    elif num_escalas <= 9:
        filas, cols = 3, 3
    else:
        filas, cols = 3, 4
    
    fig, axes = plt.subplots(filas, cols, figsize=(cols * 6, filas * 4), dpi=DPI_FIGURA)
    if num_escalas == 1:
        axes = [axes]
    elif filas == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'COMPARACIÓN DE ESCALAS - {nombre_imagen}', fontsize=16, weight='bold')
    
    for i in range(num_escalas):
        mapa, escala = mapas_ordenados[i][:2]
        ax = axes[i]
        ax.imshow(imagen_original, cmap='gray', alpha=0.7)
        
        nuevo_ancho = int(template_original.shape[1] * escala)
        nuevo_alto = int(template_original.shape[0] * escala)
        
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
        
        ax.set_title(f'Escala {escala:.1f}x\nTemplate: {nuevo_ancho}x{nuevo_alto}px', fontsize=10)
        ax.axis('off')
    
    for i in range(num_escalas, filas * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_04_comparacion_escalas.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_todas_las_detecciones(imagen_original: np.ndarray,
                                    detecciones_antes_nms: List[Dict],
                                    detecciones_despues_nms: List[Dict],
                                    nombre_imagen: str):
    """Visualiza todas las detecciones antes y después del NMS."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=DPI_FIGURA)
    
    # Detecciones antes del NMS
    ax1.imshow(imagen_original, cmap='gray')
    ax1.set_title(f'ANTES NMS - {len(detecciones_antes_nms)} detecciones', fontsize=14, weight='bold')
    
    for i, det in enumerate(detecciones_antes_nms[:100]):  # Limitar visualización
        x, y, w, h = det['x'], det['y'], det['ancho'], det['alto']
        color = plt.cm.viridis(det['confianza'])[:3]
        
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                               edgecolor=color, facecolor='none', alpha=0.7)
        ax1.add_patch(rect)
        
        if i < 20:  # Solo mostrar texto para las primeras 20
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
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='white', weight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_05_comparacion_detecciones.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_detecciones_finales_numeradas(imagen_original: np.ndarray,
                                           detecciones_despues_nms: List[Dict],
                                           nombre_imagen: str):
    """Visualiza las detecciones finales con numeración clara."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    plt.figure(figsize=(16, 12), dpi=DPI_FIGURA)
    plt.imshow(imagen_original, cmap='gray')
    
    colores = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
              'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'maroon', 'teal']
    
    if detecciones_despues_nms:
        for i, det in enumerate(detecciones_despues_nms):
            x, y = det['x'], det['y']
            w, h = det['ancho'], det['alto']
            confianza = det['confianza']
            escala = det['escala']
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
    
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_06_detecciones_finales_numeradas.png',
                bbox_inches='tight', dpi=150)
    plt.close()


def generar_reporte_detecciones(detecciones: List[Dict], nombre_imagen: str):
    """Genera un reporte textual de las detecciones."""
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    reporte_path = f'{CARPETA_RESULTADOS}/{nombre_base}_reporte.txt'
    
    with open(reporte_path, 'w', encoding='utf-8') as f:
        f.write(f"REPORTE DE DETECCIONES MÚLTIPLES\n")
        f.write(f"================================\n")
        f.write(f"Imagen: {nombre_imagen}\n")
        f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de detecciones: {len(detecciones)}\n\n")
        
        if detecciones:
            f.write("DETECCIONES ENCONTRADAS:\n")
            f.write("-" * 50 + "\n")
            
            for i, det in enumerate(detecciones):
                f.write(f"Detección #{i+1}:\n")
                f.write(f"  Posición: ({det['x']}, {det['y']})\n")
                f.write(f"  Tamaño: {det['ancho']} x {det['alto']} px\n")
                f.write(f"  Centro: ({det['centro_x']}, {det['centro_y']})\n")
                f.write(f"  Confianza: {det['confianza']:.4f}\n")
                f.write(f"  Escala: {det['escala']:.2f}x\n")
                f.write(f"  Área: {det['ancho'] * det['alto']} px²\n")
                f.write("\n")
            
            # Estadísticas
            confianzas = [d['confianza'] for d in detecciones]
            escalas = [d['escala'] for d in detecciones]
            
            f.write("ESTADÍSTICAS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Confianza promedio: {np.mean(confianzas):.4f}\n")
            f.write(f"Confianza máxima: {np.max(confianzas):.4f}\n")
            f.write(f"Confianza mínima: {np.min(confianzas):.4f}\n")
            f.write(f"Escala promedio: {np.mean(escalas):.2f}x\n")
            f.write(f"Escala máxima: {np.max(escalas):.2f}x\n")
            f.write(f"Escala mínima: {np.min(escalas):.2f}x\n")
        else:
            f.write("No se encontraron detecciones.\n")


def procesar_imagen_multi(ruta_imagen: str, template_data: np.ndarray):
    """Procesa una imagen para múltiples detecciones."""
    nombre_imagen = os.path.basename(ruta_imagen)
    template_procesado = template_data

    print(f"\n{'='*60}")
    print(f"PROCESANDO: {nombre_imagen}")
    print(f"{'='*60}")

    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)
    print(f"Imagen cargada: {imagen_original.shape[1]}x{imagen_original.shape[0]} px")

    # Visualizar preprocesamiento (del archivo original)
    visualizar_preprocesamiento(
        template_procesado, imagen_procesada, nombre_imagen
    )

    # Búsqueda en múltiples escalas
    detecciones, mapas_resultado = buscar_coincidencias_multiescala_multi(
        imagen_procesada, template_procesado
    )

    # Visualizar mapas de matching (del archivo original)
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)

    print(f"Detecciones antes del filtrado: {len(detecciones)}")

    # Aplicar NMS optimizado para múltiples detecciones
    detecciones_filtradas = aplicar_nms_multi_deteccion(detecciones)

    print(f"Detecciones finales: {len(detecciones_filtradas)}")

    # Visualizar mejor detección (del archivo original)
    visualizar_resultado_final(
        imagen_original, detecciones_filtradas, nombre_imagen
    )

    # Visualizar comparación de escalas (del archivo original)
    visualizar_comparacion_escalas(
        imagen_original, template_original, mapas_resultado, nombre_imagen
    )

    # Generar visualizaciones adicionales del multi
    visualizar_todas_las_detecciones(
        imagen_original, detecciones, detecciones_filtradas, nombre_imagen
    )

    visualizar_detecciones_finales_numeradas(
        imagen_original, detecciones_filtradas, nombre_imagen
    )

    # Generar reporte
    generar_reporte_detecciones(detecciones_filtradas, nombre_imagen)

    return detecciones_filtradas


def main():
    """Función principal para detección múltiple."""
    print("=" * 80)
    print("        TEMPLATE MATCHING CANNY - DETECCIÓN MÚLTIPLE DE LOGOS        ")
    print("=" * 80)
    print("\nCONFIGURACIÓN OPTIMIZADA PARA MÚLTIPLES DETECCIONES:")
    print(f"   Método de matching: {METODO_MATCHING}")
    print(f"   Umbral de detección: {UMBRAL_SIMPLE_DETECCION}")
    print(f"   Umbral de confianza mínima: {UMBRAL_CONFIANZA_MINIMA}")
    print(f"   Rango de escalas: {ESCALA_MIN}x - {ESCALA_MAX}x (paso: {PASO_ESCALA})")
    print(f"   Umbral Canny: {UMBRAL_CANNY_MIN} - {UMBRAL_CANNY_MAX}")
    print(f"   Clustering eps: {CLUSTERING_EPS}")
    print(f"   Límite detecciones: {LIMITE_DETECCIONES_FINALES}")
    print()

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    # Cargar template
    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    template_data = cargar_template(ruta_template)

    # Procesar específicamente la imagen coca_multi.png
    ruta_imagen_multi = os.path.join(PATH_IMAGENES, 'coca_multi.png')
    
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
        print("\nDETALLE DE DETECCIONES:")
        for i, det in enumerate(detecciones):
            print(f"  #{i+1}: Confianza {det['confianza']:.3f}, "
                  f"Escala {det['escala']:.2f}x, "
                  f"Posición ({det['x']}, {det['y']})")
    
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
    print("Proceso de detección múltiple completado exitosamente!")
    print("=" * 80)


if __name__ == "__main__":
    main()