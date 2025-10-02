"""
Módulo de visualización para Template Matching
==============================================

Contiene todas las funciones de visualización y generación de gráficos.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any


def visualizar_preprocesamiento(template_procesado: np.ndarray,
                                imagen_procesada: np.ndarray,
                                nombre_imagen: str,
                                config: Dict[str, Any]):
    """
    Visualiza las entradas al algoritmo de matching.
    
    Args:
        template_procesado: Template procesado
        imagen_procesada: Imagen procesada
        nombre_imagen: Nombre de la imagen
        config: Diccionario de configuración
    """
    os.makedirs(config['CARPETA_RESULTADOS'], exist_ok=True)
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=config.get('DPI_FIGURA', 100))
    fig.suptitle(f'ENTRADAS AL ALGORITMO - {nombre_imagen}', fontsize=16, weight='bold')

    # Imagen procesada
    axes[0].imshow(imagen_procesada, cmap='gray')
    axes[0].set_title('IMAGEN PROCESADA', fontsize=12, weight='bold', color='green')
    axes[0].axis('off')

    # Template procesado
    axes[1].imshow(template_procesado, cmap='gray')
    axes[1].set_title('TEMPLATE PROCESADO', fontsize=12, weight='bold', color='blue')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_01_entradas_algoritmo.png',
                bbox_inches='tight', dpi=config.get('DPI_FIGURA', 100))
    plt.close()


def visualizar_mapas_coincidencias(mapas_resultado: List[Tuple], 
                                  nombre_imagen: str,
                                  config: Dict[str, Any]):
    """
    Visualiza mapas de matching en un único plot.
    
    Args:
        mapas_resultado: Lista de mapas de correlación
        nombre_imagen: Nombre de la imagen
        config: Diccionario de configuración
    """
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    if not mapas_resultado:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=config.get('DPI_FIGURA', 100))
        ax.text(0.5, 0.5, 'NO SE GENERARON MAPAS',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
        ax.set_title(f'MAPAS DE MATCHING - {nombre_imagen}', fontsize=16, weight='bold')
        ax.axis('off')
        plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_02_mapas_matching.png',
                    bbox_inches='tight', dpi=config.get('DPI_FIGURA', 100))
        plt.close()
        return

    num_mapas = len(mapas_resultado)
    
    # Determinar layout adaptativo
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
        filas, cols = 8, 8
        num_mapas = min(num_mapas, 64)

    # Ajustar tamaño de figura dinámicamente
    tamano_subplot = 3
    fig, axes = plt.subplots(filas, cols, figsize=(cols * tamano_subplot, filas * tamano_subplot), 
                            dpi=config.get('DPI_FIGURA', 100))
    
    # Normalizar axes para iteración
    if num_mapas == 1:
        axes = [axes]
    elif filas == 1:
        axes = axes
    else:
        axes = axes.flatten()

    fig.suptitle(f'MAPAS DE MATCHING - {nombre_imagen} ({num_mapas} escalas)', fontsize=16, weight='bold')

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

    # Ocultar axes sobrantes
    for i in range(min(len(mapas_resultado), num_mapas), filas * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_02_mapas_matching.png',
                bbox_inches='tight', dpi=config.get('DPI_FIGURA', 100))
    plt.close()


def visualizar_resultado_final(imagen_original: np.ndarray,
                              detecciones_despues_nms: List[Dict],
                              nombre_imagen: str,
                              config: Dict[str, Any]):
    """
    Visualiza la mejor detección del NMS (versión single detection).
    
    Args:
        imagen_original: Imagen original
        detecciones_despues_nms: Detecciones después del NMS
        nombre_imagen: Nombre de la imagen
        config: Diccionario de configuración
    """
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
    
    plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_03_mejor_deteccion.png',
                bbox_inches='tight', dpi=150)
    plt.close()


def visualizar_comparacion_escalas(imagen_original: np.ndarray,
                                 template_original: np.ndarray,
                                 mapas_resultado: List[Tuple],
                                 nombre_imagen: str,
                                 config: Dict[str, Any]):
    """
    Visualiza el template escalado superpuesto en la imagen.
    
    Args:
        imagen_original: Imagen original
        template_original: Template original
        mapas_resultado: Lista de mapas de correlación
        nombre_imagen: Nombre de la imagen
        config: Diccionario de configuración
    """
    if not mapas_resultado:
        return
        
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # Permitir mostrar más escalas
    num_escalas = min(24, len(mapas_resultado))
    mapas_ordenados = sorted(mapas_resultado, key=lambda x: x[1])
    
    # Determinar layout adaptativo
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
        filas, cols = 4, 6
    
    # Ajustar tamaño de figura dinámicamente
    tamano_subplot = 4
    fig, axes = plt.subplots(filas, cols, figsize=(cols * tamano_subplot, filas * tamano_subplot), 
                            dpi=config.get('DPI_FIGURA', 100))
    
    if num_escalas == 1:
        axes = [axes]
    elif filas == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'COMPARACIÓN DE ESCALAS - {nombre_imagen} ({num_escalas} escalas)', fontsize=16, weight='bold')
    
    alpha_vis = config.get('ALPHA_VISUALIZACION', 0.7)
    
    for i in range(num_escalas):
        mapa, escala = mapas_ordenados[i][:2]
        ax = axes[i]
        ax.imshow(imagen_original, cmap='gray', alpha=alpha_vis)
        
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
        
        # Ajustar tamaño de fuente según el número de escalas
        fontsize = 10 if num_escalas <= 12 else 8
        ax.set_title(f'Escala {escala:.1f}x\nTemplate: {nuevo_ancho}x{nuevo_alto}px', fontsize=fontsize)
        ax.axis('off')
    
    # Ocultar axes sobrantes
    for i in range(num_escalas, filas * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_04_comparacion_escalas.png',
                bbox_inches='tight', dpi=config.get('DPI_FIGURA', 100))
    plt.close()


def visualizar_todas_las_detecciones(imagen_original: np.ndarray,
                                    detecciones_antes_nms: List[Dict],
                                    detecciones_despues_nms: List[Dict],
                                    nombre_imagen: str,
                                    config: Dict[str, Any]):
    """
    Visualiza todas las detecciones antes y después del NMS.
    
    Args:
        imagen_original: Imagen original
        detecciones_antes_nms: Detecciones antes del NMS
        detecciones_despues_nms: Detecciones después del NMS
        nombre_imagen: Nombre de la imagen
        config: Diccionario de configuración
    """
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=config.get('DPI_FIGURA', 100))
    
    max_detecciones_vis = config.get('MAX_DETECCIONES_VISUALIZAR', 100)
    max_etiquetas_vis = config.get('MAX_ETIQUETAS_VISUALIZAR', 20)
    alpha_vis = config.get('ALPHA_VISUALIZACION', 0.7)
    padding_bbox = config.get('PADDING_BBOX', 0.3)
    
    # Detecciones antes del NMS
    ax1.imshow(imagen_original, cmap='gray')
    ax1.set_title(f'ANTES NMS - {len(detecciones_antes_nms)} detecciones', fontsize=14, weight='bold')
    
    for i, det in enumerate(detecciones_antes_nms[:max_detecciones_vis]):
        x, y, w, h = det['x'], det['y'], det['ancho'], det['alto']
        color = plt.cm.viridis(det['confianza'])[:3]
        
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                               edgecolor=color, facecolor='none', alpha=alpha_vis)
        ax1.add_patch(rect)
        
        if i < max_etiquetas_vis:
            ax1.text(x, y-5, f'{det["confianza"]:.2f}', 
                    fontsize=8, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
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
                bbox=dict(boxstyle=f"round,pad={padding_bbox}", facecolor=color, alpha=0.8),
                fontsize=10, color='white', weight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_05_comparacion_detecciones.png',
                bbox_inches='tight', dpi=config.get('DPI_FIGURA', 100))
    plt.close()


def visualizar_detecciones_finales_numeradas(imagen_original: np.ndarray,
                                           detecciones_despues_nms: List[Dict],
                                           nombre_imagen: str,
                                           config: Dict[str, Any]):
    """
    Visualiza las detecciones finales con numeración clara.
    
    Args:
        imagen_original: Imagen original
        detecciones_despues_nms: Detecciones después del NMS
        nombre_imagen: Nombre de la imagen
        config: Diccionario de configuración
    """
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    plt.figure(figsize=(16, 12), dpi=config.get('DPI_FIGURA', 100))
    plt.imshow(imagen_original, cmap='gray')
    
    colores = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
              'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'maroon', 'teal']
    
    if detecciones_despues_nms:
        for i, det in enumerate(detecciones_despues_nms):
            x, y, w, h = det['x'], det['y'], det['ancho'], det['alto']
            color = colores[i % len(colores)]
            
            # Dibujar rectángulo
            rect = patches.Rectangle((x, y), w, h, linewidth=4, 
                                   edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Etiqueta con número y confianza
            plt.text(x, y-15, f'#{i+1}', 
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, alpha=0.9),
                    fontsize=14, color='white', weight='bold', ha='center')
            
            plt.text(x+w+5, y+h//2, f'{det["confianza"]:.3f}', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    fontsize=12, color='white', weight='bold', va='center')
        
        titulo = f'DETECCIONES MÚLTIPLES - {nombre_imagen}\n{len(detecciones_despues_nms)} logos detectados'
        
    else:
        titulo = f'SIN DETECCIONES - {nombre_imagen}'
    
    plt.title(titulo, fontsize=16, weight='bold', pad=20)
    plt.axis('off')
    
    plt.savefig(f'{config["CARPETA_RESULTADOS"]}/{nombre_base}_06_detecciones_finales_numeradas.png',
                bbox_inches='tight', dpi=150)
    plt.close()