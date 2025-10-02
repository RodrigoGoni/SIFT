"""
Módulo de Non-Maximum Suppression (NMS) y filtrado
==================================================

Contiene funciones para filtrar y agrupar detecciones usando NMS, clustering y normalización.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import DBSCAN


def calcular_iou(det1: Dict, det2: Dict) -> float:
    """
    Calcula Intersection over Union entre dos detecciones.
    
    Args:
        det1: Primera detección
        det2: Segunda detección
    
    Returns:
        Valor IoU entre 0 y 1
    """
    # Coordenadas de las cajas
    x1_min, y1_min = det1['x'], det1['y']
    x1_max, y1_max = det1['x'] + det1['ancho'], det1['y'] + det1['alto']
    
    x2_min, y2_min = det2['x'], det2['y']
    x2_max, y2_max = det2['x'] + det2['ancho'], det2['y'] + det2['alto']
    
    # Calcular intersección
    x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    area_inter = x_inter * y_inter
    
    # Calcular unión
    area1 = det1['ancho'] * det1['alto']
    area2 = det2['ancho'] * det2['alto']
    area_union = area1 + area2 - area_inter
    
    return area_inter / area_union if area_union > 0 else 0.0


def aplicar_nms(detecciones: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """
    Aplica Non-Maximum Suppression simplificado (versión single detection).
    
    Args:
        detecciones: Lista de detecciones
        config: Diccionario de configuración
    
    Returns:
        Lista de detecciones filtradas
    """
    if not detecciones:
        return []
    
    # Filtrar por umbral y ordenar por confianza
    umbral_final = config.get('UMBRAL_FINAL_NMS', config.get('UMBRAL_DETECCION', 0.04))
    max_candidatos = config.get('MAXIMO_MEJORES_CANDIDATOS', 10)
    limite_final = config.get('LIMITE_DETECCIONES_FINALES', 10)
    umbral_iou = config.get('UMBRAL_IOU_NMS', 0.04)
    
    detecciones_candidatas = [d for d in detecciones if d['confianza'] >= umbral_final]
    if not detecciones_candidatas:
        detecciones_candidatas = sorted(detecciones, key=lambda x: x['confianza'], reverse=True)[:max_candidatos]
    
    detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)
    detecciones_finales = []
    
    while detecciones_ordenadas and len(detecciones_finales) < limite_final:
        mejor = detecciones_ordenadas.pop(0)
        detecciones_finales.append(mejor)
        
        detecciones_filtradas = []
        for det in detecciones_ordenadas:
            iou = calcular_iou(mejor, det)
            if iou < umbral_iou:
                detecciones_filtradas.append(det)
        
        detecciones_ordenadas = detecciones_filtradas

    return detecciones_finales


def aplicar_nms_por_escala(detecciones_escala: List[Dict], 
                          max_detecciones: int = 100,
                          umbral_iou: float = 0.2) -> List[Dict]:
    """
    Aplica NMS dentro de una escala específica para reducir redundancia.
    
    Args:
        detecciones_escala: Detecciones de una escala específica
        max_detecciones: Máximo número de detecciones a retornar
        umbral_iou: Umbral IoU para NMS
    
    Returns:
        Lista de detecciones filtradas
    """
    if not detecciones_escala:
        return []
    
    detecciones_ordenadas = sorted(detecciones_escala, key=lambda x: x['confianza'], reverse=True)
    detecciones_filtradas = []
    
    for deteccion in detecciones_ordenadas:
        if len(detecciones_filtradas) >= max_detecciones:
            break
            
        # Verificar solapamiento con detecciones ya seleccionadas
        if not any(calcular_iou(deteccion, det) > umbral_iou for det in detecciones_filtradas):
            detecciones_filtradas.append(deteccion)
    
    return detecciones_filtradas


def normalizar_detecciones_globalmente(detecciones: List[Dict]) -> List[Dict]:
    """
    Normaliza las confianzas de todas las detecciones de 0 a 1.
    
    Args:
        detecciones: Lista de detecciones
    
    Returns:
        Lista de detecciones con confianzas normalizadas
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


def agrupar_detecciones_por_clustering(detecciones: List[Dict], 
                                      eps: float = 15, 
                                      min_samples: int = 1) -> List[List[Dict]]:
    """
    Agrupa detecciones cercanas usando clustering DBSCAN.
    
    Args:
        detecciones: Lista de detecciones
        eps: Radio máximo para considerar puntos vecinos
        min_samples: Número mínimo de muestras en un cluster
    
    Returns:
        Lista de grupos de detecciones
    """
    if len(detecciones) < 2:
        return [detecciones] if detecciones else []
    
    # Extraer coordenadas del centro
    coordenadas = np.array([[det['centro_x'], det['centro_y']] for det in detecciones])
    
    # Aplicar DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    etiquetas = clustering.fit_predict(coordenadas)
    
    # Agrupar por etiquetas
    clusters = {}
    for i, etiqueta in enumerate(etiquetas):
        clusters.setdefault(etiqueta, []).append(detecciones[i])
    
    grupos = list(clusters.values())
    print(f"Clustering: {len(detecciones)} detecciones agrupadas en {len(grupos)} clusters")
    
    return grupos


def aplicar_nms_multi_deteccion(detecciones: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """
    Aplica NMS optimizado para múltiples detecciones con normalización global.
    
    Args:
        detecciones: Lista de detecciones
        config: Diccionario de configuración
    
    Returns:
        Lista de detecciones filtradas
    """
    if not detecciones:
        return []
    
    print(f"Detecciones antes del filtrado: {len(detecciones)}")
    
    # PASO 1: Normalizar todas las detecciones globalmente
    detecciones_normalizadas = normalizar_detecciones_globalmente(detecciones)
    
    # PASO 2: Filtrar por umbral de confianza normalizada
    umbral_normalizado = config.get('UMBRAL_CONFIANZA_NORMALIZADA', 0.6)
    detecciones_candidatas = [d for d in detecciones_normalizadas if d['confianza'] >= umbral_normalizado]
    
    if not detecciones_candidatas:
        # Si no hay suficientes con el umbral, tomar las mejores
        max_candidatos = config.get('MAX_CANDIDATOS', 50)
        detecciones_candidatas = sorted(detecciones_normalizadas, key=lambda x: x['confianza'], reverse=True)[:max_candidatos]
        print(f"No hay detecciones sobre el umbral {umbral_normalizado}, tomando las {len(detecciones_candidatas)} mejores")
    
    print(f"NMS entre escalas: {len(detecciones_candidatas)} candidatos después del filtrado por confianza normalizada (umbral: {umbral_normalizado})")
    
    # PASO 3: Aplicar NMS entre escalas diferentes
    detecciones_candidatas_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)
    detecciones_inter_escala = []
    
    umbral_iou = config.get('UMBRAL_IOU_NMS', 0.2)
    for deteccion in detecciones_candidatas_ordenadas:
        if not any(calcular_iou(deteccion, det) > umbral_iou for det in detecciones_inter_escala):
            detecciones_inter_escala.append(deteccion)
    
    print(f"NMS entre escalas: {len(detecciones_inter_escala)} candidatos después de filtrar solapamientos")
    
    # PASO 4: Agrupar detecciones restantes por clustering espacial
    clustering_eps = config.get('CLUSTERING_EPS', 15)
    clustering_min = config.get('CLUSTERING_MIN', 1)
    grupos_detecciones = agrupar_detecciones_por_clustering(
        detecciones_inter_escala, 
        eps=clustering_eps, 
        min_samples=clustering_min
    )
    
    detecciones_finales = []
    
    # PASO 5: Aplicar NMS refinado dentro de cada grupo
    max_por_grupo = config.get('MAX_DETECCIONES_POR_GRUPO', 8)
    for i, grupo in enumerate(grupos_detecciones):
        print(f"Procesando grupo {i+1}: {len(grupo)} detecciones")
        
        grupo_ordenado = sorted(grupo, key=lambda x: x['confianza'], reverse=True)
        detecciones_grupo = []
        
        while grupo_ordenado and len(detecciones_grupo) < max_por_grupo:
            mejor = grupo_ordenado.pop(0)
            detecciones_grupo.append(mejor)
            
            # Filtrar detecciones muy cercanas dentro del grupo
            grupo_ordenado = [det for det in grupo_ordenado 
                            if calcular_iou(mejor, det) <= umbral_iou]
        
        detecciones_finales.extend(detecciones_grupo)
    
    # PASO 6: Ordenar y limitar resultado final
    limite_final = config.get('LIMITE_FINAL', 50)
    detecciones_finales = sorted(detecciones_finales, key=lambda x: x['confianza'], reverse=True)[:limite_final]
    
    print(f"NMS final: {len(detecciones_finales)} detecciones seleccionadas")
    
    return detecciones_finales