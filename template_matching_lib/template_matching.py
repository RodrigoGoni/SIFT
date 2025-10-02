"""
Módulo de Template Matching
============================

Contiene las funciones principales para realizar template matching multiescala.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from .preprocessing import redimensionar_template, validar_dimensiones_template


def procesar_escala_individual(args) -> Tuple[List[Dict], List[Tuple]]:
    """
    Procesa una escala individual para template matching (versión simple).
    
    Args:
        args: Tupla con (escala, imagen_procesada, template_procesado, metodo_matching, umbral_simple)
    
    Returns:
        Tupla (detecciones_escala, mapas_escala)
    """
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
        mapas_escala.append((resultado, escala, "directo"))

        ubicaciones = np.where(resultado >= umbral_simple)
        
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
                'escala': escala
            })

    except Exception:
        mapa_error = np.ones((1, 1), dtype=np.float32) * 999.0
        mapas_escala.append((mapa_error, escala, "error_matching"))
    
    return detecciones_escala, mapas_escala


def procesar_escala_individual_multi(args) -> Tuple[List[Dict], List[Tuple]]:
    """
    Procesa una escala individual para template matching (versión múltiple detecciones).
    
    Args:
        args: Tupla con (escala, imagen_procesada, template_procesado, metodo_matching, umbral_simple)
    
    Returns:
        Tupla (detecciones_escala, mapas_escala)
    """
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


def buscar_coincidencias_multiescala(imagen_procesada: np.ndarray,
                                    template_procesado: np.ndarray,
                                    config: Dict[str, Any]) -> Tuple[List[Dict], List[Tuple]]:
    """
    Realiza búsqueda de template en múltiples escalas con early stopping (versión simple).
    
    Args:
        imagen_procesada: Imagen preprocesada
        template_procesado: Template preprocesado
        config: Diccionario de configuración
    
    Returns:
        Tupla (detecciones, mapas_resultado)
    """
    detecciones = []
    mapas_resultado = []

    # Generar escalas de mayor a menor para early stopping
    escalas = np.arange(config['ESCALA_MAX'], 
                       config['ESCALA_MIN'] - config['PASO_ESCALA'], 
                       -config['PASO_ESCALA'])
    
    mejor_confianza_anterior = -1.0
    escala_sin_mejora = 0
    escalas_procesadas = 0
    
    for escala in tqdm(escalas, desc="Procesando escalas", unit="escala"):
        # Verificar si el template escalado es más grande que la imagen
        if not validar_dimensiones_template(template_procesado, imagen_procesada, escala):
            # Agregar mapa sintético pero no contar para early stopping
            mapa_sintetico = np.array([[1.0]], dtype=np.float32)
            mapas_resultado.append((mapa_sintetico, escala, "error_tamaño"))
            continue
        
        detecciones_escala, mapas_escala = procesar_escala_individual(
            (escala, imagen_procesada, template_procesado, 
             config['METODO_MATCHING'], 
             config.get('UMBRAL_SIMPLE_DETECCION', config.get('UMBRAL_DETECCION', 0.04)))
        )
        
        detecciones.extend(detecciones_escala)
        mapas_resultado.extend(mapas_escala)
        escalas_procesadas += 1
        
        # Obtener la mejor confianza de esta escala
        mejor_confianza_actual = -1.0
        if mapas_escala and len(mapas_escala) > 0:
            mapa_correlacion = mapas_escala[0][0]
            if mapa_correlacion.size > 1:  # No es un mapa de error
                mejor_confianza_actual = float(mapa_correlacion.max())
        
        print(f"Escala {escala:.1f}x: Confianza max = {mejor_confianza_actual:.4f}")
        
        # Verificar early stopping
        if escalas_procesadas > 1:
            if mejor_confianza_actual <= mejor_confianza_anterior:
                escala_sin_mejora += 1
                print(f"  Sin mejora: {escala_sin_mejora}/{config['EARLY_STOPPING_ESCALAS']}")
                if escala_sin_mejora >= config['EARLY_STOPPING_ESCALAS']:
                    print(f"Early stopping: Sin mejora en {escala_sin_mejora} escalas consecutivas")
                    print(f"Última escala procesada: {escala:.1f}x")
                    break
            else:
                escala_sin_mejora = 0
                print(f"  ¡Mejora detectada! Reset contador")
        
        mejor_confianza_anterior = mejor_confianza_actual

    # Ordenar mapas por escala
    mapas_resultado.sort(key=lambda x: x[1])
    
    print(f"Escalas procesadas: {escalas_procesadas} de {len(escalas)} totales")
    
    return detecciones, mapas_resultado


def buscar_coincidencias_multiescala_multi(imagen_procesada: np.ndarray,
                                          template_procesado: np.ndarray,
                                          config: Dict[str, Any]) -> Tuple[List[Dict], List[Tuple]]:
    """
    Realiza búsqueda de template en múltiples escalas optimizado para múltiples detecciones.
    
    Args:
        imagen_procesada: Imagen preprocesada
        template_procesado: Template preprocesado
        config: Diccionario de configuración
    
    Returns:
        Tupla (detecciones, mapas_resultado)
    """
    from .nms import aplicar_nms_por_escala
    
    detecciones = []
    mapas_resultado = []

    # Generar escalas de mayor a menor para early stopping
    escalas = np.arange(config['ESCALA_MAX'], 
                       config['ESCALA_MIN'] - config['PASO_ESCALA'], 
                       -config['PASO_ESCALA'])
    
    # Variables para early stopping - implementación igual al archivo original
    mejor_confianza_global = -1.0
    escala_sin_mejora = 0
    escalas_procesadas = 0
    
    for escala in tqdm(escalas, desc="Procesando escalas", unit="escala"):
        # Verificar si el template escalado es más grande que la imagen
        if not validar_dimensiones_template(template_procesado, imagen_procesada, escala):
            mapas_resultado.append((np.array([[1.0]], dtype=np.float32), escala, "error_tamaño"))
            nuevo_ancho, nuevo_alto = int(template_procesado.shape[1] * escala), int(template_procesado.shape[0] * escala)
            print(f"Escala {escala:.2f}x: Template demasiado grande ({nuevo_ancho}x{nuevo_alto}), saltando")
            continue
        
        detecciones_escala, mapas_escala = procesar_escala_individual_multi(
            (escala, imagen_procesada, template_procesado, 
             config['METODO_MATCHING'], config['UMBRAL_DETECCION'])
        )
        
        # Aplicar NMS dentro de esta escala específica
        detecciones_escala_filtradas = aplicar_nms_por_escala(
            detecciones_escala, 
            config.get('MAX_DETECCIONES_POR_ESCALA', 100),
            config['UMBRAL_IOU_NMS']
        )
        
        detecciones.extend(detecciones_escala_filtradas)
        mapas_resultado.extend(mapas_escala)
        escalas_procesadas += 1
        
        # Obtener la mejor confianza de esta escala
        mejor_confianza_actual = -1.0
        if mapas_escala and len(mapas_escala) > 0:
            mapa_correlacion = mapas_escala[0][0]
            if mapa_correlacion.size > 1:
                mejor_confianza_actual = float(mapa_correlacion.max())
        
        print(f"Escala {escala:.2f}x: Confianza max = {mejor_confianza_actual:.4f}, "
              f"{len(detecciones_escala)} → {len(detecciones_escala_filtradas)} detecciones (NMS)")
        
        # Verificar early stopping - lógica igual al archivo original
        if escalas_procesadas > 1:
            print(f"  Comparando: actual={mejor_confianza_actual:.4f} vs mejor_global={mejor_confianza_global:.4f}")
            
            # Actualizar el mejor global si es necesario
            if mejor_confianza_actual > mejor_confianza_global:
                mejor_confianza_global = mejor_confianza_actual
                escala_sin_mejora = 0  # Reset contador si hay nueva mejor
                print(f"  ¡Nuevo máximo global! Reset contador")
            else:
                escala_sin_mejora += 1
                print(f"  Sin mejora global: {escala_sin_mejora}/{config['EARLY_STOPPING_ESCALAS']}")
                if escala_sin_mejora >= config['EARLY_STOPPING_ESCALAS']:
                    print(f"Early stopping: Sin mejora global en {escala_sin_mejora} escalas consecutivas")
                    print(f"Mejor confianza global: {mejor_confianza_global:.4f}")
                    print(f"Última escala procesada: {escala:.2f}x")
                    break
        else:
            # Primera escala procesada
            mejor_confianza_global = mejor_confianza_actual

    # Ordenar mapas por escala
    mapas_resultado.sort(key=lambda x: x[1])
    
    print(f"Escalas procesadas: {escalas_procesadas} de {len(escalas)} totales")
    print(f"Total de detecciones después del NMS por escala: {len(detecciones)}")
    
    return detecciones, mapas_resultado