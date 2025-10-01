import cv2
import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configuración de parámetros
PATH_IMAGENES = 'TP3/images/'
PATH_TEMPLATE = 'TP3/template/'
METODO_MATCHING = cv2.TM_CCOEFF_NORMED
ESCALA_MIN = 0.3
ESCALA_MAX = 3.5
PASO_ESCALA = 0.1
UMBRAL_CANNY_MIN = 100
UMBRAL_CANNY_MAX = 250
FILTRAR_RUIDO = True
KERNEL_GAUSSIANO = (5, 5)
SIGMA_GAUSSIANO = 1.5
UMBRAL_SIMPLE_DETECCION = 0.04
UMBRAL_FINAL_NMS = 0.04
UMBRAL_IOU_NMS = 0.04
MAXIMO_MEJORES_CANDIDATOS = 10
LIMITE_DETECCIONES_FINALES = 10
DPI_FIGURA = 100
CARPETA_RESULTADOS = 'resultados_canny'


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


def procesar_escala_individual(args):
    """Procesa una escala individual para template matching."""
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


def buscar_coincidencias_multiescala(imagen_procesada: np.ndarray,
                                     template_procesado: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
    """Realiza búsqueda de template en múltiples escalas con early stopping."""
    detecciones = []
    mapas_resultado = []

    # Generar escalas de mayor a menor para early stopping
    escalas = np.arange(ESCALA_MAX, ESCALA_MIN - PASO_ESCALA, -PASO_ESCALA)
    
    mejor_confianza_anterior = -1.0
    escala_sin_mejora = 0
    escalas_procesadas = 0
    
    for escala in tqdm(escalas, desc="Procesando escalas", unit="escala"):
        # Verificar si el template escalado es más grande que la imagen
        nuevo_ancho = int(template_procesado.shape[1] * escala)
        nuevo_alto = int(template_procesado.shape[0] * escala)
        
        if (nuevo_ancho > imagen_procesada.shape[1] or nuevo_alto > imagen_procesada.shape[0]):
            # Agregar mapa sintético pero no contar para early stopping
            mapa_sintetico = np.array([[1.0]], dtype=np.float32)
            mapas_resultado.append((mapa_sintetico, escala, "error_tamaño"))
            continue
        
        detecciones_escala, mapas_escala = procesar_escala_individual(
            (escala, imagen_procesada, template_procesado, 
             METODO_MATCHING, UMBRAL_SIMPLE_DETECCION)
        )
        
        detecciones.extend(detecciones_escala)
        mapas_resultado.extend(mapas_escala)
        escalas_procesadas += 1
        
        # Obtener la mejor confianza de esta escala (del mapa de correlación, no de las detecciones filtradas)
        mejor_confianza_actual = -1.0
        if mapas_escala and len(mapas_escala) > 0:
            # Obtener el mapa de correlación (primer elemento del primer mapa)
            mapa_correlacion = mapas_escala[0][0]
            if mapa_correlacion.size > 1:  # No es un mapa de error
                mejor_confianza_actual = float(mapa_correlacion.max())
        
        print(f"Escala {escala:.1f}x: Confianza max = {mejor_confianza_actual:.4f}")
        
        # Verificar early stopping (solo después de procesar al menos 1 escala)
        if escalas_procesadas > 1:
            if mejor_confianza_actual <= mejor_confianza_anterior:
                escala_sin_mejora += 1
                print(f"  Sin mejora: {escala_sin_mejora}/2")
                if escala_sin_mejora >= 2:
                    print(f"Early stopping: Sin mejora en {escala_sin_mejora} escalas consecutivas")
                    print(f"Última escala procesada: {escala:.1f}x")
                    break
            else:
                escala_sin_mejora = 0  # Reset contador si hay mejora
                print(f"  ¡Mejora detectada! Reset contador")
        
        mejor_confianza_anterior = mejor_confianza_actual

    # Ordenar mapas por escala (de menor a mayor para visualización)
    mapas_resultado.sort(key=lambda x: x[1])
    
    print(f"Escalas procesadas: {escalas_procesadas} de {len(escalas)} totales")
    
    return detecciones, mapas_resultado


def aplicar_nms(detecciones: List[Dict]) -> List[Dict]:
    """Aplica Non-Maximum Suppression simplificado."""
    if not detecciones:
        return []
    
    # Filtrar por umbral y ordenar por confianza
    detecciones_candidatas = [d for d in detecciones if d['confianza'] >= UMBRAL_FINAL_NMS]
    if not detecciones_candidatas:
        detecciones_candidatas = sorted(detecciones, key=lambda x: x['confianza'], reverse=True)[:MAXIMO_MEJORES_CANDIDATOS]
    
    detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)
    detecciones_finales = []
    
    while detecciones_ordenadas and len(detecciones_finales) < LIMITE_DETECCIONES_FINALES:
        mejor = detecciones_ordenadas.pop(0)
        detecciones_finales.append(mejor)
        
        detecciones_filtradas = []
        for det in detecciones_ordenadas:
            x1_min, y1_min = mejor['x'], mejor['y']
            x1_max, y1_max = mejor['x'] + mejor['ancho'], mejor['y'] + mejor['alto']
            
            x2_min, y2_min = det['x'], det['y']
            x2_max, y2_max = det['x'] + det['ancho'], det['y'] + det['alto']
            
            x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            area_inter = x_inter * y_inter
            
            area1 = mejor['ancho'] * mejor['alto']
            area2 = det['ancho'] * det['alto']
            area_union = area1 + area2 - area_inter
            
            iou = area_inter / area_union if area_union > 0 else 0.0
            
            if iou < UMBRAL_IOU_NMS:
                detecciones_filtradas.append(det)
        
        detecciones_ordenadas = detecciones_filtradas

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

def obtener_imagenes_objetivo() -> List[str]:
    """Obtiene la lista de imágenes que contienen 'logo' o 'retro' en el nombre."""
    patrones = ['*logo*', '*retro*', '*LOGO*']
    imagenes = []

    for patron in patrones:
        imagenes.extend(glob.glob(os.path.join(PATH_IMAGENES, patron + '.png')))
        imagenes.extend(glob.glob(os.path.join(PATH_IMAGENES, patron + '.jpg')))
        imagenes.extend(glob.glob(os.path.join(PATH_IMAGENES, patron + '.jpeg')))

    return imagenes


def procesar_imagen(ruta_imagen: str, template_data: np.ndarray):
    """Procesa una sola imagen con el template."""
    nombre_imagen = os.path.basename(ruta_imagen)
    template_procesado = template_data

    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)

    visualizar_preprocesamiento(
        template_procesado, imagen_procesada, nombre_imagen
    )

    detecciones, mapas_resultado = buscar_coincidencias_multiescala(
        imagen_procesada, template_procesado
    )

    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)

    detecciones_filtradas = aplicar_nms(detecciones)

    visualizar_resultado_final(
        imagen_original, detecciones_filtradas, nombre_imagen
    )

    visualizar_comparacion_escalas(
        imagen_original, template_original, mapas_resultado, nombre_imagen
    )

    return detecciones_filtradas


def main():
    """Función principal del script."""
    print("=" * 60)
    print("     TEMPLATE MATCHING - DETECTOR DE BORDES CANNY     ")
    print("=" * 60)
    print("\nCONFIGURACION ACTUAL:")
    print(f"   Metodo de matching: {METODO_MATCHING}")
    print(f"   Umbral de detección: {UMBRAL_SIMPLE_DETECCION}")
    print(f"   Rango de escalas: {ESCALA_MIN}x - {ESCALA_MAX}x (paso: {PASO_ESCALA})")
    print(f"   Umbral Canny: {UMBRAL_CANNY_MIN} - {UMBRAL_CANNY_MAX}")
    print()

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    template_data = cargar_template(ruta_template)

    imagenes = obtener_imagenes_objetivo()
    print(f"Imagenes encontradas: {len(imagenes)}")

    resultados_totales = {}
    
    for ruta_imagen in tqdm(imagenes, desc="Procesando imágenes", unit="img"):
        nombre_imagen = os.path.basename(ruta_imagen)
        detecciones = procesar_imagen(ruta_imagen, template_data)
        resultados_totales[nombre_imagen] = detecciones

    print("\n" + "=" * 60)
    print("                    RESUMEN FINAL                        ")
    print("=" * 60)

    total_detecciones = 0
    
    for nombre_img, detecciones in resultados_totales.items():
        num_detecciones = len(detecciones)
        total_detecciones += num_detecciones
        print(f"{nombre_img}: {num_detecciones} detecciones")

    print("-" * 60)
    print(f"TOTAL DETECCIONES: {total_detecciones}")
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")
    print("Proceso completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()