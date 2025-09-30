import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
from typing import List, Tuple, Dict
from tqdm import tqdm
import time

# Suprimir warnings de NumPy para una salida más limpia
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===================================================================
# SECCIÓN DE CONFIGURACIÓN DE PARÁMETROS
# ===================================================================

# Rutas de archivos
PATH_IMAGENES = 'TP3/images/'
PATH_TEMPLATE = 'TP3/template/'

# Parámetros de Matching
METODO_MATCHING = cv2.TM_SQDIFF_NORMED

# Parámetros de Pirámide de Escala
ESCALA_MIN = 0.3  # Comenzar desde escalas más pequeñas
ESCALA_MAX = 3
PASO_ESCALA = 0.1

# Parámetros de Canny
UMBRAL_CANNY_MIN = 50
UMBRAL_CANNY_MAX = 200

# Parámetros de Filtro de Ruido (antes de Canny)
FILTRAR_RUIDO = True
KERNEL_GAUSSIANO = (5, 5)
SIGMA_GAUSSIANO = 1.4

# Parámetros de Detección y NMS
UMBRAL_MATCHING = 15.0  # Para SQDIFF_NORMED: ajustado para valores reales encontrados
UMBRAL_IOU_NMS = 0.3

# Parámetros de visualización
TAMAÑO_FIGURA = (15, 10)
DPI_FIGURA = 100
CARPETA_RESULTADOS = 'resultados_canny'

# Parámetros de optimización
MOSTRAR_PROGRESO = True
EARLY_STOPPING = False
LIMITE_DETECCIONES_TOTALES = 10000

# Parámetros de filtrado por confianza
UMBRAL_CONFIANZA_NMS = 0.02  # Más estricto para SQDIFF_NORMED
UMBRAL_PERCENTIL_MALAS = 0.2


def cargar_template(ruta_template: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga y preprocesa el template usando método Canny.
    
    Returns:
        template_procesado, template_procesado_inv, mascara
    """
    print("METODO DE PREPROCESAMIENTO: 3")
    print("- Usando escala de grises + filtro de ruido + Canny")
    
    # Cargar template
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)
    if template_original is None:
        raise ValueError(f"No se pudo cargar el template desde {ruta_template}")
    
    # Aplicar filtro de ruido si está activado
    if FILTRAR_RUIDO:
        template_filtrado = cv2.GaussianBlur(
            template_original, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
    else:
        template_filtrado = template_original
    
    # Aplicar detector de bordes Canny
    template_procesado = cv2.Canny(
        template_filtrado, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)
    
    # Crear versión invertida
    template_procesado_inv = 255.0 - template_procesado
    
    # Crear máscara donde hay bordes detectados
    mascara = (template_procesado > 0).astype(np.uint8) * 255
    
    return template_procesado, template_procesado_inv, mascara


def preprocesar_imagen(ruta_imagen: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa una imagen usando el método Canny.
    
    Returns:
        imagen_original, imagen_procesada
    """
    # Cargar imagen
    imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen_original is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    # Aplicar filtro de ruido si está activado
    if FILTRAR_RUIDO:
        imagen_filtrada = cv2.GaussianBlur(
            imagen_original, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
    else:
        imagen_filtrada = imagen_original
    
    # Aplicar detector de bordes Canny
    imagen_procesada = cv2.Canny(
        imagen_filtrada, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)
    
    return imagen_original.astype(np.float32), imagen_procesada


def redimensionar_template_y_mascara(template: np.ndarray, mascara: np.ndarray, escala: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona tanto el template como su máscara correspondiente.
    Manejo mejorado para escalas pequeñas.
    """
    nuevo_ancho = max(1, int(template.shape[1] * escala))
    nuevo_alto = max(1, int(template.shape[0] * escala))

    # Verificar que las dimensiones mínimas sean razonables para Canny
    if nuevo_ancho < 5 or nuevo_alto < 5:
        print(f"Template muy pequeño para escala {escala}: {nuevo_ancho}x{nuevo_alto}")
        return None, None

    try:
        template_redimensionado = cv2.resize(template, (nuevo_ancho, nuevo_alto))
        mascara_redimensionada = cv2.resize(mascara, (nuevo_ancho, nuevo_alto))
        
        # Verificar que el redimensionamiento no haya causado problemas
        if template_redimensionado.size == 0 or mascara_redimensionada.size == 0:
            print(f"Error en redimensionamiento para escala {escala}")
            return None, None
            
        # Para escalas muy pequeñas, verificar que aún hay información útil
        if escala < 0.5:
            # Verificar que hay suficientes bordes detectados
            num_bordes = np.sum(template_redimensionado > 0)
            if num_bordes < 10:  # Mínimo 10 píxeles de borde
                print(f"Insuficientes bordes para escala {escala}: {num_bordes} píxeles")
                return None, None
        
        return template_redimensionado, mascara_redimensionada
        
    except Exception as e:
        print(f"Error redimensionando template para escala {escala}: {e}")
        return None, None


def procesar_escala_individual(args):
    """
    Procesa una escala individual para template matching.
    OPTIMIZACIÓN: Busca tanto máximos como mínimos en una sola pasada 
    para detectar logos normales e invertidos sin duplicar el procesamiento.
    """
    (escala, imagen_procesada, template_procesado, template_procesado_inv, 
     mascara, metodo_matching, umbral_matching) = args
    
    detecciones_escala = []
    mapas_escala = []
    
    # Redimensionar template y máscara
    template_escalado, mascara_escalada = redimensionar_template_y_mascara(
        template_procesado, mascara, escala)

    if template_escalado is None:
        return detecciones_escala, mapas_escala

    # Verificar si el template escalado es más grande que la imagen
    if (template_escalado.shape[0] > imagen_procesada.shape[0] or
            template_escalado.shape[1] > imagen_procesada.shape[1]):
        # Crear un mapa sintético con valores que indiquen que no se pudo procesar
        if metodo_matching in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            mapa_sintetico = np.array([[1.0]], dtype=np.float32)
        else:
            mapa_sintetico = np.array([[0.0]], dtype=np.float32)
        
        mapas_escala.append((mapa_sintetico, escala, "directo"))
        return detecciones_escala, mapas_escala

    try:
        # Convertir a float32 para evitar problemas
        img_match = imagen_procesada.astype(np.float32)
        temp_match = template_escalado.astype(np.float32)

        # Realizar template matching una sola vez
        resultado = None
        try:
            mask_match = mascara_escalada.astype(np.uint8)
            resultado = cv2.matchTemplate(
                img_match, temp_match, metodo_matching, mask=mask_match)
        except Exception as e:
            # Si falla con máscara, usar sin máscara
            try:
                resultado = cv2.matchTemplate(
                    img_match, temp_match, metodo_matching)
            except Exception as e2:
                print(f"Error en template matching escala {escala}: {e2}")
                if metodo_matching in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    mapa_sintetico = np.array([[1.0]], dtype=np.float32)
                else:
                    mapa_sintetico = np.array([[0.0]], dtype=np.float32)
                mapas_escala.append((mapa_sintetico, escala, "directo"))
                return detecciones_escala, mapas_escala

        # Validar resultado con manejo más robusto
        if resultado is None:
            print(f"Resultado None para escala {escala}")
            return detecciones_escala, mapas_escala
            
        if resultado.size == 0:
            print(f"Resultado vacío para escala {escala}")
            return detecciones_escala, mapas_escala
            
        # Manejo mejorado de NaN/Inf - intentar limpiar en lugar de descartar
        num_nan = np.sum(np.isnan(resultado))
        num_inf = np.sum(np.isinf(resultado))
        total_pixels = resultado.size
        
        if num_nan > 0 or num_inf > 0:
            porcentaje_corrupto = (num_nan + num_inf) / total_pixels * 100
            print(f"Escala {escala}: {num_nan} NaN, {num_inf} Inf ({porcentaje_corrupto:.1f}% corrupto)")
            
            if porcentaje_corrupto > 50:
                # Si más del 50% está corrupto, descartar
                print(f"Descartando escala {escala} - demasiado corrupta")
                return detecciones_escala, mapas_escala
            else:
                # Intentar limpiar reemplazando NaN/Inf con valores válidos
                print(f"Limpiando NaN/Inf para escala {escala}")
                # Para SQDIFF_NORMED, reemplazar con valores altos (malos matches)
                if metodo_matching == cv2.TM_SQDIFF_NORMED:
                    resultado = np.nan_to_num(resultado, nan=10.0, posinf=10.0, neginf=10.0)
                else:
                    resultado = np.nan_to_num(resultado, nan=0.0, posinf=1.0, neginf=0.0)

        # SIEMPRE agregar el mapa, incluso si no hay detecciones
        mapas_escala.append((resultado, escala, "directo"))

        # OPTIMIZACIÓN: Buscar COINCIDENCIAS con threshold dinámico como OpenCV
        # Calcular threshold dinámico basado en estadísticas del mapa de correlación
        media_resultado = np.mean(resultado)
        std_resultado = np.std(resultado)
        
        # 1. Buscar BUENAS COINCIDENCIAS con threshold adaptativo más estricto
        if metodo_matching == cv2.TM_SQDIFF_NORMED:
            # Para SQDIFF_NORMED, los MEJORES matches son valores BAJOS (mínimos globales)
            min_valor = resultado.min()
            max_valor = resultado.max()
            rango = max_valor - min_valor
            # Threshold adaptativo MÁS ESTRICTO: mínimo + 5% del rango para capturar solo los mejores matches
            threshold_buenas = min_valor + (rango * 0.05)
            # También usar el umbral fijo si es más restrictivo
            threshold_buenas = min(threshold_buenas, umbral_matching)
            ubicaciones_buenas = np.where(resultado <= threshold_buenas)
            tipo_buenas = 'minimos_globales'
        elif metodo_matching == cv2.TM_SQDIFF:
            threshold_buenas = min(umbral_matching, media_resultado - 2*std_resultado)
            ubicaciones_buenas = np.where(resultado <= threshold_buenas)
            tipo_buenas = 'minimos_globales'
        else:
            # Para CCORR y CCOEFF, los buenos matches son valores ALTOS (máximos globales)
            threshold_buenas = max(umbral_matching, media_resultado + 2*std_resultado)
            ubicaciones_buenas = np.where(resultado >= threshold_buenas)
            tipo_buenas = 'maximos_globales'
        
        # Limitar número de buenas coincidencias para eficiencia
        if len(ubicaciones_buenas[0]) > 50:
            # Tomar solo las 50 mejores
            valores_buenas = resultado[ubicaciones_buenas]
            if metodo_matching == cv2.TM_SQDIFF_NORMED:
                indices_mejores = np.argsort(valores_buenas)[:50]  # Los 50 valores MÁS PEQUEÑOS
            elif metodo_matching == cv2.TM_SQDIFF:
                indices_mejores = np.argsort(valores_buenas)[:50]  # Los 50 valores MÁS PEQUEÑOS
            else:
                indices_mejores = np.argsort(valores_buenas)[-50:]  # Los 50 valores MÁS GRANDES
            
            ubicaciones_buenas = (ubicaciones_buenas[0][indices_mejores], 
                                 ubicaciones_buenas[1][indices_mejores])
            valores_buenas = valores_buenas[indices_mejores]
        else:
            valores_buenas = resultado[ubicaciones_buenas]
        
        # Agregar buenas coincidencias
        for i, (y, x) in enumerate(zip(ubicaciones_buenas[0], ubicaciones_buenas[1])):
            confianza = float(valores_buenas[i])
            if np.isnan(confianza) or np.isinf(confianza):
                continue

            detecciones_escala.append({
                'x': int(x),
                'y': int(y),
                'ancho': template_escalado.shape[1],
                'alto': template_escalado.shape[0],
                'confianza': confianza,
                'escala': escala,
                'version': 'directo',
                'tipo_correlacion': tipo_buenas
            })

        # 2. Buscar MALAS COINCIDENCIAS con threshold adaptativo estricto
        if metodo_matching == cv2.TM_SQDIFF_NORMED:
            # Para SQDIFF_NORMED, las MALAS coincidencias son valores ALTOS (cerca del máximo)
            # Threshold adaptativo MÁS ESTRICTO: máximo - 5% del rango para capturar solo las peores matches
            threshold_malas = max_valor - (rango * 0.05)
            # También usar un threshold mínimo de 0.95 para capturar las peores coincidencias
            threshold_malas = max(threshold_malas, 0.95)
            ubicaciones_malas = np.where(resultado >= threshold_malas)
            tipo_malas = 'maximos_globales'
        elif metodo_matching == cv2.TM_SQDIFF:
            threshold_malas = np.percentile(resultado, 98)  # Solo el 2% peor
            ubicaciones_malas = np.where(resultado >= threshold_malas)
            tipo_malas = 'maximos_globales'
        else:
            # Para CCORR y CCOEFF, las malas coincidencias son valores BAJOS
            threshold_malas = np.percentile(resultado, 2)  # Solo el 2% peor
            ubicaciones_malas = np.where(resultado <= threshold_malas)
            tipo_malas = 'minimos_globales'
        
        # Limitar malas coincidencias también para eficiencia (menos malas que buenas)
        if len(ubicaciones_malas[0]) > 10:  # Solo 10 malas vs 50 buenas
            valores_malas = resultado[ubicaciones_malas]
            if metodo_matching == cv2.TM_SQDIFF_NORMED:
                indices_peores = np.argsort(valores_malas)[-10:]  # Los 10 valores MÁS GRANDES
            elif metodo_matching == cv2.TM_SQDIFF:
                indices_peores = np.argsort(valores_malas)[-10:]  # Los 10 valores MÁS GRANDES
            else:
                indices_peores = np.argsort(valores_malas)[:10]   # Los 10 valores MÁS PEQUEÑOS
            
            ubicaciones_malas = (ubicaciones_malas[0][indices_peores],
                                ubicaciones_malas[1][indices_peores])
            valores_malas = valores_malas[indices_peores]
        else:
            valores_malas = resultado[ubicaciones_malas]
        
        # Agregar malas coincidencias
        for i, (y, x) in enumerate(zip(ubicaciones_malas[0], ubicaciones_malas[1])):
            confianza = float(valores_malas[i])
            if np.isnan(confianza) or np.isinf(confianza):
                continue

            detecciones_escala.append({
                'x': int(x),
                'y': int(y),
                'ancho': template_escalado.shape[1],
                'alto': template_escalado.shape[0],
                'confianza': confianza,
                'escala': escala,
                'version': 'invertido',
                'tipo_correlacion': tipo_malas
            })

    except Exception as e:
        print(f"Error en escala {escala}: {e}")
        mapa_error = np.ones((1, 1), dtype=np.float32) * 999.0
        mapas_escala.append((mapa_error, escala, "directo"))
    
    return detecciones_escala, mapas_escala


def buscar_coincidencias_multiescala(imagen_procesada: np.ndarray,
                                     template_procesado: np.ndarray,
                                     template_procesado_inv: np.ndarray,
                                     mascara: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
    """
    Realiza búsqueda de template en múltiples escalas.
    """
    detecciones = []
    mapas_resultado = []

    # Generar escalas
    escalas = np.arange(ESCALA_MIN, ESCALA_MAX + PASO_ESCALA, PASO_ESCALA)
    escalas_validas = []
    
    # Pre-filtrar escalas válidas
    for escala in escalas:
        nuevo_ancho = int(template_procesado.shape[1] * escala)
        nuevo_alto = int(template_procesado.shape[0] * escala)
        
        # CRITERIO ADAPTATIVO: Para Canny necesitamos templates grandes suficiente
        # para que los bordes se detecten, pero también queremos escalas pequeñas
        min_size = 15  # Mínimo más pequeño pero realista para Canny
        if nuevo_ancho >= min_size and nuevo_alto >= min_size:
            escalas_validas.append(escala)
    
    if not escalas_validas:
        print("ADVERTENCIA: No se encontraron escalas válidas, usando escala 1.0")
        escalas_validas = [1.0]
    
    if MOSTRAR_PROGRESO:
        print(f"Probando {len(escalas_validas)} escalas válidas (de {len(escalas)} generadas)...")
    
    # Procesamiento secuencial
    escalas_iter = tqdm(escalas_validas, desc="Procesando escalas", unit="escala") if MOSTRAR_PROGRESO else escalas_validas
    
    for escala in escalas_iter:
        detecciones_escala, mapas_escala = procesar_escala_individual(
            (escala, imagen_procesada, template_procesado, 
             template_procesado_inv, mascara, METODO_MATCHING, UMBRAL_MATCHING)
        )
        detecciones.extend(detecciones_escala)
        mapas_resultado.extend(mapas_escala)
        
        # Early stopping
        if EARLY_STOPPING and len(detecciones) > LIMITE_DETECCIONES_TOTALES:
            if MOSTRAR_PROGRESO:
                print(f"Early stopping: {len(detecciones)} detecciones encontradas")
            break

    # Ordenar mapas por escala
    mapas_resultado.sort(key=lambda x: x[1])
    
    if MOSTRAR_PROGRESO:
        print(f"Total detecciones encontradas: {len(detecciones)}")
        if mapas_resultado:
            print(f"Escalas procesadas: {mapas_resultado[0][1]:.1f}x a {mapas_resultado[-1][1]:.1f}x")
    
    return detecciones, mapas_resultado


def aplicar_nms(detecciones: List[Dict]) -> List[Dict]:
    """
    Aplica Non-Maximum Suppression CORREGIDO para SQDIFF_NORMED.
    Para SQDIFF_NORMED: 0.0 = coincidencia perfecta, valores más altos = peores coincidencias
    """
    if not detecciones:
        return []

    print(f"DEBUG NMS: Recibidas {len(detecciones)} detecciones")
    
    # Mostrar algunas confianzas para debug
    if detecciones:
        confianzas_sample = [d['confianza'] for d in detecciones[:5]]
        print(f"DEBUG NMS: Primeras 5 confianzas: {confianzas_sample}")

    # PASO 1: Filtrar por confianza directamente
    if METODO_MATCHING == cv2.TM_SQDIFF_NORMED:
        # Para SQDIFF_NORMED: MEJORES = valores MÁS BAJOS (cercanos a 0)
        # Usar umbral adaptativo basado en los valores reales
        confianzas = [d['confianza'] for d in detecciones]
        confianza_min = min(confianzas)
        confianza_media = np.mean(confianzas)
        
        # Umbral dinámico: permitir hasta la mitad del rango desde el mínimo
        umbral_dinamico = confianza_min + (confianza_media - confianza_min) * 0.5
        
        print(f"DEBUG NMS: Confianza mín: {confianza_min:.3f}, media: {confianza_media:.3f}")
        print(f"DEBUG NMS: Umbral dinámico: {umbral_dinamico:.3f}")
        
        detecciones_candidatas = [d for d in detecciones if d['confianza'] <= umbral_dinamico]
        print(f"DEBUG NMS: Candidatas con umbral dinámico: {len(detecciones_candidatas)}")
        
        # Si no hay suficientes, tomar al menos las 3 mejores
        if len(detecciones_candidatas) < 3:
            detecciones_candidatas = sorted(detecciones, key=lambda x: x['confianza'])[:5]
            print(f"DEBUG NMS: Tomando las 5 mejores por confianza: {len(detecciones_candidatas)}")
    else:
        # Para otros métodos: MEJORES = valores MÁS ALTOS
        detecciones_candidatas = [d for d in detecciones if d['confianza'] >= 0.7]
    
    if not detecciones_candidatas:
        print("DEBUG NMS: No hay candidatas válidas")
        return []

    print(f"DEBUG NMS: {len(detecciones_candidatas)} candidatas válidas")

    # PASO 2: Ordenar por confianza CORRECTAMENTE
    if METODO_MATCHING == cv2.TM_SQDIFF_NORMED:
        # Para SQDIFF_NORMED: ordenar ASCENDENTE (valores más bajos = mejores)
        detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'])
        print(f"DEBUG NMS: Mejor confianza después de ordenar: {detecciones_ordenadas[0]['confianza']:.6f}")
    else:
        # Para otros métodos: ordenar DESCENDENTE (valores más altos = mejores)
        detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)

    # PASO 3: Aplicar NMS
    detecciones_finales = []
    
    while detecciones_ordenadas and len(detecciones_finales) < 10:  # Limitar a 10 max
        mejor = detecciones_ordenadas.pop(0)
        detecciones_finales.append(mejor)
        
        # Filtrar detecciones solapantes
        detecciones_filtradas = []
        for det in detecciones_ordenadas:
            # Calcular IoU
            x1_min, y1_min = mejor['x'], mejor['y']
            x1_max, y1_max = mejor['x'] + mejor['ancho'], mejor['y'] + mejor['alto']
            
            x2_min, y2_min = det['x'], det['y']
            x2_max, y2_max = det['x'] + det['ancho'], det['y'] + det['alto']
            
            # Intersección
            x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            area_inter = x_inter * y_inter
            
            # Áreas
            area1 = mejor['ancho'] * mejor['alto']
            area2 = det['ancho'] * det['alto']
            area_union = area1 + area2 - area_inter
            
            iou = area_inter / area_union if area_union > 0 else 0.0
            
            if iou < UMBRAL_IOU_NMS:
                detecciones_filtradas.append(det)
        
        detecciones_ordenadas = detecciones_filtradas

    print(f"DEBUG NMS: Detecciones finales: {len(detecciones_finales)}")
    if detecciones_finales:
        print(f"DEBUG NMS: Mejor confianza final: {detecciones_finales[0]['confianza']:.6f}")

    return detecciones_finales


# ===================================================================
# FUNCIONES DE VISUALIZACIÓN (copiadas del archivo original)
# ===================================================================

def visualizar_preprocesamiento(template_procesado: np.ndarray,
                                template_procesado_inv: np.ndarray,
                                mascara: np.ndarray,
                                imagen_original: np.ndarray,
                                imagen_procesada: np.ndarray,
                                nombre_imagen: str):
    """
    Visualiza las entradas REALES al algoritmo de matching.
    """
    # Crear carpeta si no existe
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # CREAR RESUMEN SIMPLE - ENTRADAS REALES AL ALGORITMO DE MATCHING
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=DPI_FIGURA)

    fig.suptitle(f'ENTRADAS REALES AL ALGORITMO DE MATCHING - {nombre_imagen}',
                 fontsize=16, weight='bold')

    # IMAGEN PROCESADA (la que realmente entra al algoritmo de matching)
    axes[0].imshow(imagen_procesada, cmap='gray')
    axes[0].set_title('IMAGEN PROCESADA\n(Entrada real al matching)',
                     fontsize=12, weight='bold', color='green')
    axes[0].axis('off')

    # TEMPLATE PROCESADO (la que realmente entra al algoritmo de matching)
    axes[1].imshow(template_procesado, cmap='gray')
    axes[1].set_title('TEMPLATE PROCESADO\n(Entrada real al matching)',
                     fontsize=12, weight='bold', color='blue')
    axes[1].axis('off')

    plt.tight_layout()

    # GUARDAR ENTRADAS REALES AL ALGORITMO
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_01_entradas_algoritmo.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_mapas_coincidencias(mapas_resultado: List[Tuple], nombre_imagen: str):
    """
    Visualiza TODOS los mapas de matching en un único plot.
    """
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    if not mapas_resultado:
        # Si no hay mapas, crear un plot informativo
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=DPI_FIGURA)
        ax.text(0.5, 0.5, 'NO SE GENERARON MAPAS DE MATCHING\n\n'
                           'Posibles causas:\n'
                           '• Escalas demasiado pequeñas\n'
                           '• Errores en el procesamiento\n'
                           '• Template no compatible',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
        ax.set_title(f'MAPAS DE MATCHING - {nombre_imagen}', fontsize=16, weight='bold')
        ax.axis('off')
        plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_02_mapas_matching.png',
                    bbox_inches='tight', dpi=DPI_FIGURA)
        plt.close()
        return

    # Determinar número de subplots necesarios
    num_mapas = len(mapas_resultado)
    
    # Calcular distribución de subplots (máximo 4 columnas)
    if num_mapas <= 4:
        filas, cols = 1, num_mapas
    elif num_mapas <= 8:
        filas, cols = 2, 4
    elif num_mapas <= 12:
        filas, cols = 3, 4
    else:
        filas, cols = 4, 4
        num_mapas = 16  # Limitar a 16 mapas máximo

    # Crear plot unificado con TODOS los mapas de matching
    fig, axes = plt.subplots(filas, cols, figsize=(cols * 4, filas * 3), dpi=DPI_FIGURA)
    
    # Asegurar que axes sea siempre un array
    if num_mapas == 1:
        axes = [axes]
    elif filas == 1:
        axes = axes
    else:
        axes = axes.flatten()

    fig.suptitle(f'MAPAS DE MATCHING - TODAS LAS ESCALAS - {nombre_imagen}\n'
                 f'Mostrando {min(len(mapas_resultado), num_mapas)} escalas', 
                 fontsize=16, weight='bold')

    # Mostrar cada mapa de matching
    for i in range(min(len(mapas_resultado), num_mapas)):
        mapa_data = mapas_resultado[i]
        if len(mapa_data) == 3:
            mapa, escala, version = mapa_data
        else:
            mapa, escala = mapa_data
            version = "directo"
        
        ax = axes[i]
        
        # Verificar si es un mapa sintético de error
        if mapa.shape == (1, 1) and mapa[0, 0] > 900:
            ax.text(0.5, 0.5, f'ERROR\nEscala {escala:.1f}x\nTemplate muy grande',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, weight='bold', color='red')
            ax.set_title(f'Escala {escala:.1f}x - ERROR', fontsize=10, color='red')
        else:
            # Mostrar mapa normal
            im = ax.imshow(mapa, cmap='hot', interpolation='nearest')
            ax.set_title(f'Escala {escala:.1f}x\n'
                        f'Max: {mapa.max():.3f}, Min: {mapa.min():.3f}', 
                        fontsize=10)
            
            # Añadir colorbar pequeño
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.axis('off')

    # Ocultar subplots vacíos
    for i in range(min(len(mapas_resultado), num_mapas), filas * cols):
        axes[i].axis('off')

    plt.tight_layout()

    # Guardar mapas unificados
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_02_mapas_matching.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_resultado_final(imagen_original: np.ndarray,
                               detecciones_antes_nms: List[Dict],
                               detecciones_despues_nms: List[Dict],
                               nombre_imagen: str):
    """
    Visualiza solo la mejor detección del NMS de forma simplificada.
    """
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # MOSTRAR SOLO LA MEJOR DETECCIÓN DEL NMS
    plt.figure(figsize=(12, 8))
    plt.imshow(imagen_original, cmap='gray')
    
    if detecciones_despues_nms:
        mejor_det = detecciones_despues_nms[0]
        x, y = mejor_det['x'], mejor_det['y']
        w, h = mejor_det['ancho'], mejor_det['alto']
        confianza = mejor_det['confianza']
        
        # Dibujar rectángulo
        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Añadir texto con confianza
        plt.text(x, y-10, f'Mejor: {confianza:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                fontsize=12, color='black', weight='bold')
        
        titulo = f'MEJOR DETECCIÓN - {nombre_imagen}\nConfianza: {confianza:.3f} | Escala: {mejor_det["escala"]:.1f}x'
    else:
        titulo = f'SIN DETECCIONES - {nombre_imagen}'
    
    plt.title(titulo, fontsize=14, weight='bold')
    plt.axis('off')
    
    # Guardar resultado final
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_03_mejor_deteccion.png',
                bbox_inches='tight', dpi=150)
    plt.close()


def visualizar_comparacion_escalas(imagen_original: np.ndarray,
                                 template_original: np.ndarray,
                                 mapas_resultado: List[Tuple],
                                 nombre_imagen: str):
    """
    Nuevo plot: Imagen original sobre el logo escalado para ver la escala óptima.
    """
    if not mapas_resultado:
        return
        
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # Usar TODAS las escalas disponibles (máximo 12 para no saturar)
    num_escalas = min(12, len(mapas_resultado))
    
    # ORDENAR los mapas por escala ascendente para mostrar desde la menor
    mapas_ordenados = sorted(mapas_resultado, key=lambda x: x[1])  # x[1] es la escala
    
    # Calcular distribución de subplots
    if num_escalas <= 3:
        filas, cols = 1, 3
    elif num_escalas <= 6:
        filas, cols = 2, 3
    elif num_escalas <= 9:
        filas, cols = 3, 3
    else:
        filas, cols = 3, 4
    
    # Crear subplot con las escalas seleccionadas
    fig, axes = plt.subplots(filas, cols, figsize=(cols * 6, filas * 4), dpi=DPI_FIGURA)
    if num_escalas == 1:
        axes = [axes]
    elif filas == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'COMPARACIÓN DE ESCALAS - {nombre_imagen}\n'
                 f'Template superpuesto en diferentes escalas (desde {mapas_ordenados[0][1]:.1f}x hasta {mapas_ordenados[num_escalas-1][1]:.1f}x)', 
                 fontsize=16, weight='bold')
    
    # Usar las escalas ORDENADAS empezando desde la menor (0.1x)
    for i in range(num_escalas):
        mapa_data = mapas_ordenados[i]
        if len(mapa_data) == 3:
            mapa, escala, version = mapa_data
        else:
            mapa, escala = mapa_data
            version = "directo"
        
        ax = axes[i]
        
        # Mostrar imagen original
        ax.imshow(imagen_original, cmap='gray', alpha=0.7)
        
        # Redimensionar template a esta escala
        nuevo_ancho = int(template_original.shape[1] * escala)
        nuevo_alto = int(template_original.shape[0] * escala)
        
        if nuevo_ancho > 0 and nuevo_alto > 0:
            template_escalado = cv2.resize(template_original, (nuevo_ancho, nuevo_alto))
            
            # Superponer template escalado en el centro de la imagen
            center_x = imagen_original.shape[1] // 2 - nuevo_ancho // 2
            center_y = imagen_original.shape[0] // 2 - nuevo_alto // 2
            
            # Crear una versión visible del template (contorno)
            template_contorno = np.zeros_like(imagen_original)
            if (center_x >= 0 and center_y >= 0 and 
                center_x + nuevo_ancho <= imagen_original.shape[1] and
                center_y + nuevo_alto <= imagen_original.shape[0]):
                template_contorno[center_y:center_y+nuevo_alto, center_x:center_x+nuevo_ancho] = template_escalado
            
            # Superponer con transparencia
            ax.imshow(template_contorno, cmap='Reds', alpha=0.5)
            
            # Dibujar rectángulo del template
            rect = plt.Rectangle((center_x, center_y), nuevo_ancho, nuevo_alto,
                               linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        ax.set_title(f'Escala {escala:.1f}x\nTemplate: {nuevo_ancho}x{nuevo_alto}px',
                    fontsize=10)
        ax.axis('off')
    
    # Ocultar subplots vacíos
    for i in range(num_escalas, filas * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Guardar comparación de escalas
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_04_comparacion_escalas.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


# ===================================================================
# FUNCIÓN PRINCIPAL
# ===================================================================

def obtener_imagenes_objetivo() -> List[str]:
    """
    Obtiene la lista de imágenes que contienen 'logo' o 'retro' en el nombre.
    """
    patrones = ['*logo*', '*retro*', '*LOGO*']
    imagenes = []

    for patron in patrones:
        imagenes.extend(glob.glob(os.path.join(PATH_IMAGENES, patron + '.png')))
        imagenes.extend(glob.glob(os.path.join(PATH_IMAGENES, patron + '.jpg')))
        imagenes.extend(glob.glob(os.path.join(PATH_IMAGENES, patron + '.jpeg')))

    return imagenes


def procesar_imagen(ruta_imagen: str, template_data: Tuple):
    """
    Procesa una sola imagen con el template.
    """
    nombre_imagen = os.path.basename(ruta_imagen)
    
    # Solo mostrar headers detallados si no hay barra de progreso global
    if not MOSTRAR_PROGRESO:
        print(f"{'='*60}")
        print(f"PROCESANDO: {nombre_imagen}")
        print(f"{'='*60}")

    # Desempaquetar datos del template
    template_procesado, template_procesado_inv, mascara = template_data

    # Cargar template original para comparación de escalas
    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    # Preprocesar imagen
    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)

    if not MOSTRAR_PROGRESO:
        print("1. Preprocesamiento completado")
        print(f"   Imagen: {imagen_original.shape[1]}x{imagen_original.shape[0]}")
        print(f"   Template: {template_procesado.shape[1]}x{template_procesado.shape[0]}")

    # Visualizar preprocesamiento
    if not MOSTRAR_PROGRESO:
        print("2. Generando visualización de preprocesamiento...")
    visualizar_preprocesamiento(
        template_procesado, template_procesado_inv, mascara,
        imagen_original, imagen_procesada, nombre_imagen
    )

    # Búsqueda multi-escala
    if not MOSTRAR_PROGRESO:
        print("3. Iniciando búsqueda multi-escala...")
    detecciones, mapas_resultado = buscar_coincidencias_multiescala(
        imagen_procesada, template_procesado, template_procesado_inv, mascara
    )

    # Visualizar mapas de coincidencias (TODOS los mapas de matching en un plot)
    if not MOSTRAR_PROGRESO:
        print("4. Generando mapas de coincidencias...")
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)

    # Aplicar NMS
    if not MOSTRAR_PROGRESO:
        print("5. Aplicando Non-Maximum Suppression...")
    detecciones_filtradas = aplicar_nms(detecciones)

    # Visualizar resultado final (solo mejor detección)
    if not MOSTRAR_PROGRESO:
        print("6. Generando resultado final...")
    visualizar_resultado_final(
        imagen_original, detecciones, detecciones_filtradas, nombre_imagen
    )

    # Visualizar comparación de escalas (nuevo plot)
    if not MOSTRAR_PROGRESO:
        print("7. Generando comparación de escalas...")
    visualizar_comparacion_escalas(
        imagen_original, template_original, mapas_resultado, nombre_imagen
    )

    return detecciones_filtradas


def main():
    """
    Función principal del script.
    """
    print("=" + "="*58 + "=")
    print("=     TEMPLATE MATCHING - DETECTOR DE BORDES CANNY     =")
    print("=" + "="*58 + "=")
    print()
    print("CONFIGURACION ACTUAL:")
    print(f"   Metodo de matching: {METODO_MATCHING}")
    print(f"   Metodo de preprocesamiento: 3 (Canny)")
    print(f"   Umbral de confianza: {UMBRAL_MATCHING}")
    print(f"   Umbral filtro NMS: {UMBRAL_IOU_NMS}")
    print(f"   Rango de escalas: {ESCALA_MIN}x - {ESCALA_MAX}x (paso: {PASO_ESCALA})")
    print(f"   Umbral Canny: {UMBRAL_CANNY_MIN} - {UMBRAL_CANNY_MAX}")
    print()

    # Crear carpeta de resultados
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    print(f"Carpeta de resultados: {CARPETA_RESULTADOS}/")

    # Cargar y preprocesar template
    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    print(f"Cargando template desde: {ruta_template}")
    template_data = cargar_template(ruta_template)
    template_shape = template_data[0].shape
    print(f"Template cargado: {template_shape[1]}x{template_shape[0]} pixeles")
    print()

    # Obtener imágenes objetivo
    imagenes = obtener_imagenes_objetivo()
    print(f"Imagenes encontradas para procesar: {len(imagenes)}")
    for i, img in enumerate(imagenes, 1):
        print(f"   {i}. {os.path.basename(img)}")
    print()

    print("COMENZANDO PROCESAMIENTO...")
    print("   El template (pattern.png) se buscara DENTRO de cada imagen")
    print("   Se probaran diferentes escalas para encontrar coincidencias")
    print("   Los resultados se guardaran en:", CARPETA_RESULTADOS)
    print()

    # Mostrar información de optimización
    print(f"OPTIMIZACIONES ACTIVAS:")
    print(f"   - Progreso visual: {'Sí' if MOSTRAR_PROGRESO else 'No'}")
    print()

    # Procesar cada imagen con barra de progreso
    resultados_totales = {}
    
    imagenes_iter = tqdm(imagenes, desc="Procesando imágenes", unit="img") if MOSTRAR_PROGRESO else imagenes
    
    for ruta_imagen in imagenes_iter:
        nombre_imagen = os.path.basename(ruta_imagen)
        if not MOSTRAR_PROGRESO:
            print(f"\nProcesando: {nombre_imagen}")
        
        detecciones = procesar_imagen(ruta_imagen, template_data)
        resultados_totales[nombre_imagen] = detecciones

    # Resumen final
    print("\n" + "=" + "="*58 + "=")
    print("=                    RESUMEN FINAL                        =")
    print("=" + "="*58 + "=")

    total_detecciones = 0
    total_buenas = 0
    total_malas = 0
    
    for nombre_img, detecciones in resultados_totales.items():
        num_detecciones = len(detecciones)
        total_detecciones += num_detecciones
        
        # Contar buenas y malas detecciones
        for det in detecciones:
            if det['version'] == 'directo':
                total_buenas += 1
            else:
                total_malas += 1
        
        print(f"{nombre_img}: {num_detecciones} detecciones")

    print("-" * 60)
    print(f"TOTAL DETECCIONES: {total_detecciones} (BUENAS: {total_buenas}, MALAS: {total_malas})")
    print(f"Método usado: SQDIFF_NORMED con preprocesamiento Canny")
    print("INTERPRETACIÓN: BUENAS = coincidencias directas, MALAS = coincidencias invertidas")
    print("Template usado: pattern.png")
    print("Imagenes procesadas: TP3/images/")
    print("Resultados guardados en:", CARPETA_RESULTADOS)
    
    print("Proceso completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()