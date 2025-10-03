import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import DBSCAN


def calcular_iou(det1: Dict, det2: Dict) -> float:
    """
    Calcula Intersection over Union entre dos detecciones.
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


def aplicar_nms(detecciones, config):
    """
    Aplica Non-Maximum Suppression a las detecciones usando normalización global.
    """
    if not detecciones:
        return []
    umbral_confianza = config['UMBRAL_CONFIANZA_NORMALIZADA']
    umbral_iou = config['UMBRAL_IOU_NMS']
    max_detecciones = config['LIMITE_FINAL']
    # Normalizar las detecciones globalmente
    detecciones_normalizadas = normalizar_detecciones_globalmente(detecciones)
    
    # Filtrar por umbral de confianza usando valores normalizados
    detecciones_filtradas = [
        det for det in detecciones_normalizadas
        if det['confianza_normalizada'] >= umbral_confianza
    ]
    
    if not detecciones_filtradas:
        return []
    
    # Ordenar por confianza normalizada descendente
    detecciones_filtradas = sorted(
        detecciones_filtradas,
        key=lambda x: x['confianza_normalizada'],
        reverse=True
    )
    
    # Aplicar NMS
    detecciones_seleccionadas = []
    indice = 0
    
    while len(detecciones_seleccionadas) < max_detecciones and indice < len(detecciones_filtradas):
        deteccion_actual = detecciones_filtradas[indice]
        
        # Calcular IoU máximo con detecciones ya seleccionadas
        max_iou = 0
        for det_seleccionada in detecciones_seleccionadas:
            iou = calcular_iou(deteccion_actual, det_seleccionada)
            max_iou = max(max_iou, iou)
        
        if max_iou < umbral_iou:
            detecciones_seleccionadas.append(deteccion_actual)
        
        indice += 1
    
    return detecciones_seleccionadas



def normalizar_detecciones_globalmente(detecciones: List[Dict]) -> List[Dict]:
    """
    Normaliza las confianzas usando el máximo como referencia.
    """
    if not detecciones:
        return []
    
    # Extraer todas las confianzas
    confianzas = [det['confianza'] for det in detecciones]
    
    if not confianzas:
        return []
    
    # Usar el máximo como referencia (normalización 0-1 basada en máximo)
    confianza_max = max(confianzas)
    
    # Normalizar cada detección dividiendo por el máximo
    detecciones_normalizadas = []
    for det in detecciones:
        det_normalizada = det.copy()
        
        # Normalización simple: dividir por el máximo
        confianza_normalizada = det['confianza'] / confianza_max
        det_normalizada['confianza_normalizada'] = confianza_normalizada
        
        detecciones_normalizadas.append(det_normalizada)
    
    return detecciones_normalizadas