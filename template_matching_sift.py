import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from functools import partial
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
# Opciones disponibles: cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED
# CONFIGURACIÓN OPTIMIZADA: TM_CCOEFF_NORMED es más selectivo y discriminativo
METODO_MATCHING = cv2.TM_SQDIFF_NORMED

# Parámetros de Pirámide de Escala - CORREGIDOS SEGÚN ANÁLISIS
# Escalas corregidas para template grande (400x175) vs imágenes pequeñas (~500x200)
ESCALA_MIN = 0.1    # Reducido drásticamente para template muy grande
ESCALA_MAX = 3    # Reducido para evitar que template sea mayor que imagen
PASO_ESCALA = 0.1  # Pasos más finos para mejor precisión

# Parámetros de Canny
UMBRAL_CANNY_MIN = 50
UMBRAL_CANNY_MAX = 200

# Parámetros de Filtro de Ruido (antes de Canny)
# Filtro Gaussiano para reducir ruido antes de detectar bordes
FILTRAR_RUIDO = True  # Activar/desactivar filtro de ruido
KERNEL_GAUSSIANO = (5, 5)  # Tamaño del kernel Gaussiano (debe ser impar)
SIGMA_GAUSSIANO = 1.4  # Desviación estándar del filtro Gaussiano

# Parámetros de Detección y NMS - CORREGIDOS
# CONFIGURACIÓN CORREGIDA para template grande con TM_CCOEFF_NORMED
UMBRAL_MATCHING = 0.1  # Para SQDIFF_NORMED: valores menores = mejor coincidencia (0.0 = perfecta coincidencia)
UMBRAL_IOU_NMS = 0.3    # Mantenido para NMS agresivo

# METODOS DE PREPROCESAMIENTO DISPONIBLES
# 1: Solo escala de grises (RECOMENDADO - ÓPTIMO)
# 2: Escala de grises + binarización
# 3: Escala de grises + Canny
# 4: Escala de grises + binarización + Canny
# 5: Pirámides Laplacianas (reduce información, enfoca en bordes importantes)
# Método de preprocesamiento (1-5)
METODO_PREPROCESAMIENTO = 3# Método 1: Solo escala de grises (mejor para SQDIFF_NORMED)

# Parámetros para Pirámides Laplacianas (método 5)
NIVELES_PIRAMIDE = 5  # Número de niveles de la pirámide
USAR_PIRAMIDE_GAUSSIANA = True  # Usar Gaussiana antes de Laplaciana

# Parámetros de umbralización para binarización
UMBRAL_BINARIO = 127

# Parámetros de visualización
TAMAÑO_FIGURA = (15, 10)
DPI_FIGURA = 100
CARPETA_RESULTADOS = 'resultados_template_matching'

# Parámetros de optimización y paralelización
NUM_PROCESOS = max(1, cpu_count() - 1)  # Usar todos los cores menos 1
BATCH_SIZE_ESCALAS = 5  # Procesar escalas en lotes para reducir overhead
USAR_MULTIPROCESSING = True  # Activar/desactivar paralelización
MOSTRAR_PROGRESO = True  # Mostrar barras de progreso

# Parámetros de optimización avanzada
EARLY_STOPPING = False  # Parar cuando se encuentren suficientes detecciones buenas
LIMITE_DETECCIONES_TOTALES = 10000  # Parar si se encuentran muchas detecciones (solo para early stopping)

# Parámetros de filtrado por confianza
UMBRAL_CONFIANZA_NMS = 0.15  # Umbral mínimo de confianza para NMS
UMBRAL_PERCENTIL_MALAS = 0.2  # Percentil para detectar malas coincidencias (20% superior/inferior)



def crear_piramide_laplaciana(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica filtro Laplaciano para detectar bordes y reducir información irrelevante.
    
    Args:
        imagen: Imagen en escala de grises

    Returns:
        Imagen procesada con filtro Laplaciano (enfocada en bordes)
    """
    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3:
        img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        img = imagen.copy()
    
    # Convertir a float32 para cálculos precisos
    img = img.astype(np.float32)
    
    # Aplicar filtro Gaussiano para reducir ruido
    img_suavizada = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Aplicar filtro Laplaciano para detectar bordes
    # Usar kernel Laplaciano más agresivo
    kernel_laplaciano = np.array([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]], dtype=np.float32)
    
    img_laplaciana = cv2.filter2D(img_suavizada, -1, kernel_laplaciano)
    
    # Tomar valor absoluto para obtener magnitud de los bordes
    img_laplaciana = np.abs(img_laplaciana)
    
    # Normalizar entre 0 y 255
    if img_laplaciana.max() > 0:
        img_laplaciana = (img_laplaciana / img_laplaciana.max()) * 255.0
    
    # Aplicar threshold para enfatizar solo bordes fuertes
    _, img_threshold = cv2.threshold(img_laplaciana, 30, 255, cv2.THRESH_BINARY)
    
    return img_threshold.astype(np.uint8)


def cargar_template(ruta_template: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga y preprocesa el template según el método seleccionado.

    Returns:
        template_procesado, template_procesado_inv, mascara
    """
    # Cargar template original
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)
    if template_original is None:
        raise ValueError(
            f"No se pudo cargar el template desde {ruta_template}")

    print(f"METODO DE PREPROCESAMIENTO: {METODO_PREPROCESAMIENTO}")

    if METODO_PREPROCESAMIENTO == 1:
        print("- Usando solo escala de grises")
        template_procesado = template_original.astype(np.float32)
        template_procesado_inv = 255.0 - template_procesado
        # Máscara simple (toda la imagen)
        mascara = np.ones_like(template_original, dtype=np.uint8) * 255

    elif METODO_PREPROCESAMIENTO == 2:
        print("- Usando escala de grises + binarización")
        # Binarización
        _, template_bin = cv2.threshold(
            template_original, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY)
        template_procesado = template_bin.astype(np.float32)
        template_procesado_inv = cv2.bitwise_not(
            template_bin).astype(np.float32)
        # Máscara del template binarizado
        mascara = template_bin.astype(np.uint8)

    elif METODO_PREPROCESAMIENTO == 3:
        print("- Usando escala de grises + filtro de ruido + Canny")
        # Aplicar filtro de ruido antes de Canny
        if FILTRAR_RUIDO:
            template_filtrado = cv2.GaussianBlur(
                template_original, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
        else:
            template_filtrado = template_original
        # Aplicar Canny sobre imagen filtrada
        template_procesado = cv2.Canny(
            template_filtrado, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)
        template_procesado_inv = 255.0 - template_procesado
        # Máscara donde hay bordes
        mascara = (template_procesado > 0).astype(np.uint8) * 255

    elif METODO_PREPROCESAMIENTO == 4:
        print("- Usando escala de grises + binarización + filtro de ruido + Canny (RECOMENDADO)")
        # Binarización primero
        _, template_bin = cv2.threshold(
            template_original, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY)
        # Aplicar filtro de ruido antes de Canny
        if FILTRAR_RUIDO:
            template_bin_filtrado = cv2.GaussianBlur(
                template_bin, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
        else:
            template_bin_filtrado = template_bin
        # Luego Canny
        template_procesado = cv2.Canny(
            template_bin_filtrado, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)
        template_bin_inv = cv2.bitwise_not(template_bin)
        if FILTRAR_RUIDO:
            template_bin_inv_filtrado = cv2.GaussianBlur(
                template_bin_inv, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
        else:
            template_bin_inv_filtrado = template_bin_inv
        template_procesado_inv = cv2.Canny(
            template_bin_inv_filtrado, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)
        # Máscara del template binarizado
        mascara = template_bin.astype(np.uint8)

    elif METODO_PREPROCESAMIENTO == 5:
        print("- Usando Pirámides Laplacianas (reduce información, enfoca en bordes)")
        # Crear pirámide laplaciana para reducir información
        template_procesado = crear_piramide_laplaciana(template_original)
        template_procesado_inv = 255.0 - template_procesado
        # Máscara donde hay información significativa
        mascara = (np.abs(template_procesado) > 10).astype(np.uint8) * 255

    else:
        raise ValueError(
            f"Método de preprocesamiento no válido: {METODO_PREPROCESAMIENTO}")

    return template_procesado, template_procesado_inv, mascara


def preprocesar_imagen(ruta_imagen: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga y preprocesa una imagen de entrada según el método seleccionado.

    Returns:
        imagen_original, imagen_procesada
    """
    # Cargar imagen
    imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen_original is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_imagen}")

    if METODO_PREPROCESAMIENTO == 1:
        # Solo escala de grises
        imagen_procesada = imagen_original.astype(np.float32)

    elif METODO_PREPROCESAMIENTO == 2:
        # Escala de grises + binarización
        _, imagen_procesada = cv2.threshold(
            imagen_original, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY)
        imagen_procesada = imagen_procesada.astype(np.float32)

    elif METODO_PREPROCESAMIENTO == 3:
        # Escala de grises + filtro de ruido + Canny
        if FILTRAR_RUIDO:
            imagen_filtrada = cv2.GaussianBlur(
                imagen_original, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
        else:
            imagen_filtrada = imagen_original
        imagen_procesada = cv2.Canny(
            imagen_filtrada, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)

    elif METODO_PREPROCESAMIENTO == 4:
        # Escala de grises + binarización + filtro de ruido + Canny
        _, imagen_bin = cv2.threshold(
            imagen_original, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY)
        if FILTRAR_RUIDO:
            imagen_bin_filtrada = cv2.GaussianBlur(
                imagen_bin, KERNEL_GAUSSIANO, SIGMA_GAUSSIANO)
        else:
            imagen_bin_filtrada = imagen_bin
        imagen_procesada = cv2.Canny(
            imagen_bin_filtrada, UMBRAL_CANNY_MIN, UMBRAL_CANNY_MAX).astype(np.float32)

    elif METODO_PREPROCESAMIENTO == 5:
        # Pirámides Laplacianas
        imagen_procesada = crear_piramide_laplaciana(imagen_original)

    return imagen_original.astype(np.float32), imagen_procesada


def redimensionar_template_y_mascara(template: np.ndarray, mascara: np.ndarray, escala: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona tanto el template como su máscara correspondiente.
    """
    nuevo_ancho = int(template.shape[1] * escala)
    nuevo_alto = int(template.shape[0] * escala)

    if nuevo_ancho <= 0 or nuevo_alto <= 0:
        return None, None

    template_redimensionado = cv2.resize(template, (nuevo_ancho, nuevo_alto))
    mascara_redimensionada = cv2.resize(mascara, (nuevo_ancho, nuevo_alto))

    return template_redimensionado, mascara_redimensionada

# ===================================================================
# FUNCIONES DE TEMPLATE MATCHING Y DETECCIÓN
# ===================================================================

def procesar_escala_individual(args):
    """
    Procesa una escala individual para template matching.
    Función auxiliar para paralelización.
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

    # NOTA: OpenCV puede manejar templates más grandes que la imagen
    # El resultado será un mapa más pequeño pero válido
    # No rechazamos escalas por ser más grandes que la imagen

    # NOTA: Verificar si el template escalado es más grande que la imagen
    # OpenCV no puede hacer template matching si template > imagen
    if (template_escalado.shape[0] > imagen_procesada.shape[0] or
            template_escalado.shape[1] > imagen_procesada.shape[1]):
        # Crear un mapa sintético con valores que indiquen que no se pudo procesar
        # El mapa tendrá tamaño 1x1 con un valor que indique "no procesable"
        if metodo_matching in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            # Para SQDIFF, usar valor alto (mala coincidencia)
            mapa_sintetico = np.array([[1.0]], dtype=np.float32)
        else:
            # Para CCOEFF/CCORR, usar valor bajo (mala coincidencia) 
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
                # Si también falla sin máscara, reportar y crear mapa sintético
                print(f"Error en template matching escala {escala}: {e2}")
                if metodo_matching in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    mapa_sintetico = np.array([[1.0]], dtype=np.float32)
                else:
                    mapa_sintetico = np.array([[0.0]], dtype=np.float32)
                mapas_escala.append((mapa_sintetico, escala, "directo"))
                return detecciones_escala, mapas_escala

        # Validar resultado
        if resultado is None:
            print(f"Resultado None para escala {escala}")
            return detecciones_escala, mapas_escala
            
        if resultado.size == 0:
            print(f"Resultado vacío para escala {escala}")
            return detecciones_escala, mapas_escala
            
        if np.any(np.isnan(resultado)) or np.any(np.isinf(resultado)):
            print(f"Resultado con NaN/Inf para escala {escala}")
            return detecciones_escala, mapas_escala

        # SIEMPRE agregar el mapa, incluso si no hay detecciones
        mapas_escala.append((resultado, escala, "directo"))

        # OPTIMIZACIÓN: Buscar COINCIDENCIAS con threshold dinámico como OpenCV
        # En lugar de buscar todas las detecciones, usar threshold adaptativo
        
        # Calcular threshold dinámico basado en estadísticas del mapa de correlación
        media_resultado = np.mean(resultado)
        std_resultado = np.std(resultado)
        
        # 1. Buscar BUENAS COINCIDENCIAS con threshold adaptativo más estricto
        if metodo_matching == cv2.TM_SQDIFF_NORMED:
            # Para SQDIFF_NORMED, los MEJORES matches son valores BAJOS (mínimos globales)
            # ESTRATEGIA: Buscar los valores más cercanos al mínimo del mapa
            min_valor = resultado.min()
            max_valor = resultado.max()
            rango = max_valor - min_valor
            # Threshold adaptativo: mínimo + 10% del rango para capturar los mejores matches
            threshold_buenas = min_valor + (rango * 0.1)
            ubicaciones_buenas = np.where(resultado <= threshold_buenas)
            tipo_buenas = 'minimos_globales'
        elif metodo_matching == cv2.TM_SQDIFF:
            # Para SQDIFF sin normalizar, los buenos matches también son valores BAJOS
            threshold_buenas = max(umbral_matching, media_resultado - 2*std_resultado)
            ubicaciones_buenas = np.where(resultado <= threshold_buenas)
            tipo_buenas = 'minimos_globales'
        else:
            # Para CCORR y CCOEFF, los buenos matches son valores ALTOS (máximos globales)
            threshold_buenas = min(umbral_matching, media_resultado + 2*std_resultado)
            ubicaciones_buenas = np.where(resultado >= threshold_buenas)
            tipo_buenas = 'maximos_globales'
        
        # Limitar número de buenas coincidencias para eficiencia
        if len(ubicaciones_buenas[0]) > 50:
            # Tomar solo las 50 mejores
            valores_buenas = resultado[ubicaciones_buenas]
            if metodo_matching == cv2.TM_SQDIFF_NORMED:
                # Para SQDIFF_NORMED: menores valores = mejores matches
                indices_mejores = np.argsort(valores_buenas)[:50]  # Los 50 valores MÁS PEQUEÑOS
            elif metodo_matching == cv2.TM_SQDIFF:
                # Para SQDIFF: menores valores = mejores matches
                indices_mejores = np.argsort(valores_buenas)[:50]  # Los 50 valores MÁS PEQUEÑOS
            else:
                # Para CCORR y CCOEFF: mayores valores = mejores matches
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
            # Para SQDIFF_NORMED, las MALAS coincidencias son valores ALTOS
            # ESTRATEGIA: Buscar los valores más cercanos al máximo del mapa
            # Threshold adaptativo: máximo - 10% del rango para capturar las peores matches
            threshold_malas = max_valor - (rango * 0.1)
            ubicaciones_malas = np.where(resultado >= threshold_malas)
            tipo_malas = 'maximos_globales'
        elif metodo_matching == cv2.TM_SQDIFF:
            # Para SQDIFF sin normalizar, las malas coincidencias también son valores ALTOS
            threshold_malas = np.percentile(resultado, 95)
            ubicaciones_malas = np.where(resultado >= threshold_malas)
            tipo_malas = 'maximos_globales'
        else:
            # Para CCORR y CCOEFF, las malas coincidencias son valores BAJOS
            threshold_malas = np.percentile(resultado, 5)
            ubicaciones_malas = np.where(resultado <= threshold_malas)
            tipo_malas = 'minimos_globales'
        
        # Limitar malas coincidencias también para eficiencia
        if len(ubicaciones_malas[0]) > 30:
            # Tomar solo las 30 mejores malas coincidencias
            valores_malas = resultado[ubicaciones_malas]
            if metodo_matching == cv2.TM_SQDIFF_NORMED:
                indices_mejores = np.argsort(valores_malas)[-30:]  # Valores más altos (peores matches)
            elif metodo_matching == cv2.TM_SQDIFF:
                indices_mejores = np.argsort(valores_malas)[-30:]  # Valores más altos (peores matches)
            else:
                indices_mejores = np.argsort(valores_malas)[:30]   # Valores más bajos
            
            ubicaciones_malas = (ubicaciones_malas[0][indices_mejores],
                                ubicaciones_malas[1][indices_mejores])
            valores_malas = valores_malas[indices_mejores]
        else:
            valores_malas = resultado[ubicaciones_malas]
        
        # Agregar malas coincidencias
        for i, (y, x) in enumerate(zip(ubicaciones_malas[0], ubicaciones_malas[1])):
            confianza_original = float(valores_malas[i])
            if np.isnan(confianza_original) or np.isinf(confianza_original):
                continue

            # Transformar confianza para comparación uniforme
            if metodo_matching == cv2.TM_SQDIFF_NORMED:
                # Para SQDIFF_NORMED: valor alto (cerca de 1) = mala coincidencia
                # Transformar: 1 - valor = confianza (valores altos se vuelven bajos)
                confianza_transformada = 1.0 - confianza_original
            elif metodo_matching == cv2.TM_SQDIFF:
                # Para SQDIFF: valores altos = malas coincidencias
                confianza_transformada = 255 - confianza_original
            else:
                # Para CCORR y CCOEFF: usar valor absoluto de valores negativos/bajos
                confianza_transformada = abs(confianza_original)

            detecciones_escala.append({
                'x': int(x),
                'y': int(y),
                'ancho': template_escalado.shape[1],
                'alto': template_escalado.shape[0],
                'confianza': confianza_transformada,
                'escala': escala,
                'version': 'invertido',
                'tipo_correlacion': tipo_malas
            })

    except Exception as e:
        # Reportar error pero continuar procesamiento
        print(f"Error procesando escala {escala}: {e}")
    
    return detecciones_escala, mapas_escala


def buscar_coincidencias_multiescala(imagen_procesada: np.ndarray,
                                     template_procesado: np.ndarray,
                                     template_procesado_inv: np.ndarray,
                                     mascara: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
    """
    Realiza búsqueda de template en múltiples escalas con optimizaciones.
    Usa paralelización cuando está disponible.

    Returns:
        Lista de detecciones con información de posición, escala y confianza
    """
    detecciones = []
    mapas_resultado = []

    # Generar escalas
    escalas = np.arange(ESCALA_MIN, ESCALA_MAX + PASO_ESCALA, PASO_ESCALA)
    escalas_validas = []
    
    # Pre-filtrar escalas válidas - CRITERIO CORREGIDO
    # Solo eliminar escalas que sean demasiado pequeñas (menores a 5 píxeles)
    for escala in escalas:
        nuevo_ancho = int(template_procesado.shape[1] * escala)
        nuevo_alto = int(template_procesado.shape[0] * escala)
        
        # Criterio más permisivo: solo rechazar si es demasiado pequeño
        # Template matching puede manejar templates más grandes que la imagen
        if nuevo_ancho >= 5 and nuevo_alto >= 5:
            escalas_validas.append(escala)
    
    # Si no hay escalas válidas, usar al menos la escala 1.0
    if not escalas_validas:
        escalas_validas = [1.0]
        print("ADVERTENCIA: No se encontraron escalas válidas, usando escala 1.0")
    
    if MOSTRAR_PROGRESO:
        print(f"Probando {len(escalas_validas)} escalas válidas (de {len(escalas)} generadas)...")
    
    # Decidir si usar paralelización
    if USAR_MULTIPROCESSING and len(escalas_validas) > 4:
        # Procesamiento paralelo
        args_list = [(escala, imagen_procesada, template_procesado, 
                     template_procesado_inv, mascara, METODO_MATCHING, UMBRAL_MATCHING)
                    for escala in escalas_validas]
        
        with Pool(processes=NUM_PROCESOS) as pool:
            if MOSTRAR_PROGRESO:
                # Usar tqdm para mostrar progreso
                resultados = list(tqdm(
                    pool.imap(procesar_escala_individual, args_list),
                    total=len(args_list),
                    desc="Procesando escalas",
                    unit="escala"
                ))
            else:
                resultados = pool.map(procesar_escala_individual, args_list)
        
        # Combinar resultados con early stopping
        for detecciones_escala, mapas_escala in resultados:
            detecciones.extend(detecciones_escala)
            mapas_resultado.extend(mapas_escala)
            
            # Early stopping si encontramos muchas detecciones
            if EARLY_STOPPING and len(detecciones) > LIMITE_DETECCIONES_TOTALES:
                if MOSTRAR_PROGRESO:
                    print(f"Early stopping: {len(detecciones)} detecciones encontradas")
                break
    
    else:
        # Procesamiento secuencial con early stopping
        escalas_iter = tqdm(escalas_validas, desc="Procesando escalas", unit="escala") if MOSTRAR_PROGRESO else escalas_validas
        
        for escala in escalas_iter:
            detecciones_escala, mapas_escala = procesar_escala_individual(
                (escala, imagen_procesada, template_procesado, 
                 template_procesado_inv, mascara, METODO_MATCHING, UMBRAL_MATCHING)
            )
            detecciones.extend(detecciones_escala)
            mapas_resultado.extend(mapas_escala)
            
            # Early stopping en modo secuencial
            if EARLY_STOPPING and len(detecciones) > LIMITE_DETECCIONES_TOTALES:
                if MOSTRAR_PROGRESO:
                    print(f"Early stopping: {len(detecciones)} detecciones encontradas")
                break

    # ORDENAR mapas por escala para asegurar orden correcto desde 0.1x
    mapas_resultado.sort(key=lambda x: x[1])  # x[1] es la escala
    
    if MOSTRAR_PROGRESO:
        print(f"Total detecciones encontradas: {len(detecciones)}")
        if mapas_resultado:
            print(f"Escalas procesadas: {mapas_resultado[0][1]:.1f}x a {mapas_resultado[-1][1]:.1f}x")
    
    return detecciones, mapas_resultado


def calcular_iou_modificado(caja1: Dict, caja2: Dict) -> float:
    """
    Calcula IoU modificado que penaliza diferencias de área.
    """
    # Coordenadas de las cajas
    x1_min, y1_min = caja1['x'], caja1['y']
    x1_max, y1_max = x1_min + caja1['ancho'], y1_min + caja1['alto']

    x2_min, y2_min = caja2['x'], caja2['y']
    x2_max, y2_max = x2_min + caja2['ancho'], y2_min + caja2['alto']

    # Calcular intersección
    x_interseccion = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_interseccion = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    area_interseccion = x_interseccion * y_interseccion

    # Calcular áreas
    area1 = caja1['ancho'] * caja1['alto']
    area2 = caja2['ancho'] * caja2['alto']
    area_union = area1 + area2 - area_interseccion

    if area_union == 0:
        return 0

    # IoU básico
    iou = area_interseccion / area_union

    # Penalización por diferencia de área
    ratio_area = min(area1, area2) / max(area1, area2)
    iou_modificado = iou * ratio_area

    return iou_modificado


def aplicar_nms(detecciones: List[Dict]) -> List[Dict]:
    """
    Aplica Non-Maximum Suppression OPTIMIZADO con pre-filtrado por threshold.
    NUEVA ESTRATEGIA: Filtrar primero por threshold como OpenCV, luego NMS eficiente.
    """
    if not detecciones:
        return []

    # PASO 1: PRE-FILTRADO AGRESIVO POR THRESHOLD (como OpenCV)
    # Separar por tipo de correlación
    buenas_detecciones = [d for d in detecciones if d['version'] == 'directo']
    malas_detecciones = [d for d in detecciones if d['version'] == 'invertido']
    
    detecciones_candidatas = []
    
    # Filtrar buenas coincidencias con threshold dinámico
    if buenas_detecciones:
        confianzas_buenas = [d['confianza'] for d in buenas_detecciones]
        # Threshold más estricto: percentil 85 para reducir candidatos
        threshold_buenas = np.percentile(confianzas_buenas, 85)
        candidatas_buenas = [d for d in buenas_detecciones if d['confianza'] >= threshold_buenas]
        detecciones_candidatas.extend(candidatas_buenas)
    
    # Filtrar malas coincidencias con threshold dinámico
    if malas_detecciones:
        confianzas_malas = [d['confianza'] for d in malas_detecciones]
        # Threshold aún más estricto: percentil 90 para reducir candidatos
        threshold_malas = np.percentile(confianzas_malas, 90)
        candidatas_malas = [d for d in malas_detecciones if d['confianza'] >= threshold_malas]
        detecciones_candidatas.extend(candidatas_malas)
    
    # Si después del filtrado quedan demasiadas, tomar solo las mejores
    if len(detecciones_candidatas) > 100:
        if METODO_MATCHING == cv2.TM_SQDIFF_NORMED:
            # Para SQDIFF_NORMED: ordenar por confianza transformada (mayor = mejor)
            detecciones_candidatas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)[:100]
        elif METODO_MATCHING == cv2.TM_SQDIFF:
            # Para SQDIFF: menor valor = mejor (ordenar ascendente)
            detecciones_candidatas = sorted(detecciones_candidatas, key=lambda x: x['confianza'])[:100]
        else:
            # Para CCORR y CCOEFF: mayor valor = mejor (ordenar descendente)
            detecciones_candidatas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)[:100]
    
    if not detecciones_candidatas:
        return []

    # PASO 2: NMS OPTIMIZADO solo en candidatos filtrados
    if METODO_MATCHING == cv2.TM_SQDIFF_NORMED:
        # Para SQDIFF_NORMED: usar confianza transformada (mayor = mejor)
        detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)
    elif METODO_MATCHING == cv2.TM_SQDIFF:
        # Para SQDIFF: menor valor = mejor
        detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'])
    else:
        # Para CCORR y CCOEFF: mayor valor = mejor
        detecciones_ordenadas = sorted(detecciones_candidatas, key=lambda x: x['confianza'], reverse=True)

    detecciones_finales = []
    
    while detecciones_ordenadas and len(detecciones_finales) < 20:
        # Tomar la mejor detección
        mejor_deteccion = detecciones_ordenadas.pop(0)
        detecciones_finales.append(mejor_deteccion)

        # NMS vectorizado para mayor eficiencia
        detecciones_ordenadas = [
            det for det in detecciones_ordenadas
            if calcular_iou_modificado(mejor_deteccion, det) <= UMBRAL_IOU_NMS
        ]

    return detecciones_finales

# ===================================================================
# FUNCIONES DE VISUALIZACIÓN
# ===================================================================


def visualizar_preprocesamiento(template_procesado: np.ndarray,
                                template_procesado_inv: np.ndarray,
                                mascara: np.ndarray,
                                imagen_original: np.ndarray,
                                imagen_procesada: np.ndarray,
                                nombre_imagen: str):
    """
    Visualiza las entradas REALES al algoritmo de matching.
    Muestra la imagen procesada que realmente se usa en el matching, no la original.
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
    Muestra los mapas de correlación de todas las escalas en subplots.
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
        num_mapas = 16  # Limitamos a 16 para no saturar

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
        mapa, escala, version = mapas_resultado[i]

        # Mostrar mapa como imagen de calor
        im = axes[i].imshow(mapa, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Escala: {escala:.1f}x\n({version})', fontsize=10, weight='bold')
        axes[i].axis('off')

        # Encontrar y marcar extremos
        max_match = np.unravel_index(np.argmax(mapa), mapa.shape)
        max_value = np.max(mapa)
        min_match = np.unravel_index(np.argmin(mapa), mapa.shape)
        min_value = np.min(mapa)

        # Marcar máximo (azul)
        axes[i].plot(max_match[1], max_match[0], 'bo', markersize=6,
                     markeredgecolor='white', markeredgewidth=1)
        
        # Marcar mínimo (rojo)
        axes[i].plot(min_match[1], min_match[0], 'ro', markersize=6,
                     markeredgecolor='white', markeredgewidth=1)

        # Agregar colorbar pequeña
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Match', rotation=270, labelpad=15, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        # Mostrar valores extremos
        axes[i].text(0.02, 0.98, f'MAX: {max_value:.3f}\nMIN: {min_value:.3f}', 
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                    va='top', fontsize=8, weight='bold')

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
    SIMPLIFICADO: Solo muestra la mejor detección final, sin todas las detecciones.
    """
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # MOSTRAR SOLO LA MEJOR DETECCIÓN DEL NMS
    plt.figure(figsize=(12, 8))
    plt.imshow(imagen_original, cmap='gray')
    
    if detecciones_despues_nms:
        # Encontrar la mejor detección (mayor confianza)
        mejor_deteccion = max(detecciones_despues_nms, key=lambda x: x['confianza'])
        
        # Dibujar la mejor detección
        rect = plt.Rectangle((mejor_deteccion['x'], mejor_deteccion['y']), 
                           mejor_deteccion['ancho'], mejor_deteccion['alto'],
                           linewidth=3, edgecolor='lime', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Agregar información de la detección
        plt.text(mejor_deteccion['x'], mejor_deteccion['y']-10, 
                f'MEJOR DETECCIÓN\nConfianza: {mejor_deteccion["confianza"]:.3f}\n'
                f'Escala: {mejor_deteccion["escala"]:.1f}x\n'
                f'Tipo: {mejor_deteccion["version"]}',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lime', alpha=0.8),
                ha='left', va='bottom', fontweight='bold', fontsize=12)
        
        titulo = f'MEJOR DETECCIÓN - {nombre_imagen}\nConfianza: {mejor_deteccion["confianza"]:.3f}'
    else:
        titulo = f'NO SE ENCONTRARON DETECCIONES - {nombre_imagen}'
    
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
    Muestra TODAS las escalas del template superpuestas en la imagen.
    """
    if not mapas_resultado:
        return
        
    nombre_base = os.path.splitext(nombre_imagen)[0]
    
    # Usar TODAS las escalas disponibles (máximo 12 para no saturar)
    num_escalas = min(12, len(mapas_resultado))
    
    # ORDENAR los mapas por escala ascendente para mostrar desde la menor
    mapas_ordenados = sorted(mapas_resultado, key=lambda x: x[1])  # x[1] es la escala
    
    # DEBUG: Verificar que las escalas estén ordenadas correctamente
    escalas_disponibles = [mapa[1] for mapa in mapas_ordenados[:num_escalas]]
    print(f"DEBUG - Escalas a mostrar en {nombre_imagen}: {escalas_disponibles}")
    
    # Calcular distribución de subplots
    if num_escalas <= 3:
        filas, cols = 1, num_escalas
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
        mapa, escala, version = mapas_ordenados[i]
        
        # Mostrar imagen original como fondo
        axes[i].imshow(imagen_original, cmap='gray', alpha=0.7)
        
        # Redimensionar template a la escala actual
        nuevo_ancho = int(template_original.shape[1] * escala)
        nuevo_alto = int(template_original.shape[0] * escala)
        
        # QUITAR RESTRICCIONES - permitir cualquier tamaño
        try:
            if nuevo_ancho > 0 and nuevo_alto > 0:
                template_escalado = cv2.resize(template_original, (nuevo_ancho, nuevo_alto))
                
                # Posición centrada para superponer el template
                pos_x = max(0, (imagen_original.shape[1] - nuevo_ancho) // 2)
                pos_y = max(0, (imagen_original.shape[0] - nuevo_alto) // 2)
                
                # Asegurar que no se salga de la imagen
                pos_x = min(pos_x, imagen_original.shape[1] - 1)
                pos_y = min(pos_y, imagen_original.shape[0] - 1)
                
                # Ajustar tamaños si el template es más grande que la imagen
                ancho_final = min(nuevo_ancho, imagen_original.shape[1] - pos_x)
                alto_final = min(nuevo_alto, imagen_original.shape[0] - pos_y)
                
                if ancho_final > 0 and alto_final > 0:
                    # Recortar template si es necesario
                    template_recortado = template_escalado[:alto_final, :ancho_final]
                    
                    # Crear máscara para el template escalado
                    template_mask = np.zeros_like(imagen_original)
                    template_mask[pos_y:pos_y+alto_final, pos_x:pos_x+ancho_final] = template_recortado
                    
                    # Superponer template con transparencia
                    axes[i].imshow(template_mask, cmap='Reds', alpha=0.5)
                    
                    # Dibujar rectángulo del template
                    rect = plt.Rectangle((pos_x, pos_y), ancho_final, alto_final,
                                       linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                    axes[i].add_patch(rect)
        except Exception as e:
            # Si hay error, continuar sin el template superpuesto
            pass
        
        # Configurar subplot
        axes[i].set_title(f'Escala: {escala:.1f}x\nTamaño: {nuevo_ancho}x{nuevo_alto}px', 
                         fontsize=10, weight='bold')
        axes[i].axis('off')
        
        # Agregar información del matching
        try:
            if METODO_MATCHING in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                valor_optimo = np.min(mapa)
                color_info = 'green' if valor_optimo < UMBRAL_MATCHING else 'red'
            else:
                valor_optimo = np.max(mapa)
                color_info = 'green' if valor_optimo > UMBRAL_MATCHING else 'red'
            
            axes[i].text(0.02, 0.98, f'Match: {valor_optimo:.3f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_info, alpha=0.7),
                        va='top', fontweight='bold', color='white')
        except:
            pass
    
    # Ocultar subplots vacíos
    for i in range(num_escalas, filas * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Guardar comparación de escalas
    plt.savefig(f'{CARPETA_RESULTADOS}/{nombre_base}_04_comparacion_escalas.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()

# ===================================================================
# FUNCIÓN DE ANÁLISIS EXPLORATORIO
# ===================================================================


def analisis_exploratorio():
    """
    Realiza análisis exploratorio automático para encontrar la mejor configuración
    que produzca exactamente 1 detección por imagen sin falsos positivos.
    """
    print("=" * 80)
    print("=                  ANÁLISIS EXPLORATORIO AUTOMÁTICO                  =")
    print("=" * 80)
    print("OBJETIVO: Encontrar configuración que dé 1 detección por imagen sin falsos positivos")
    print()

    # Configuraciones a probar
    configuraciones = [
        # [metodo_preprocesamiento, metodo_matching, umbral_matching, descripcion]
        [1, cv2.TM_CCOEFF_NORMED, 0.8, "Grises + CCOEFF_NORMED"],
        [1, cv2.TM_CCORR_NORMED, 0.8, "Grises + CCORR_NORMED"],
        [1, cv2.TM_SQDIFF_NORMED, 0.2, "Grises + SQDIFF_NORMED"],

        [2, cv2.TM_CCOEFF_NORMED, 0.7, "Binario + CCOEFF_NORMED"],
        [2, cv2.TM_CCORR_NORMED, 0.7, "Binario + CCORR_NORMED"],
        [2, cv2.TM_SQDIFF_NORMED, 0.3, "Binario + SQDIFF_NORMED"],

        [3, cv2.TM_CCOEFF_NORMED, 0.6, "Canny + CCOEFF_NORMED"],
        [3, cv2.TM_CCORR_NORMED, 0.6, "Canny + CCORR_NORMED"],
        [3, cv2.TM_SQDIFF_NORMED, 0.4, "Canny + SQDIFF_NORMED"],

        [4, cv2.TM_CCOEFF_NORMED, 0.7, "Binario+Canny + CCOEFF_NORMED"],
        [4, cv2.TM_CCORR_NORMED, 0.7, "Binario+Canny + CCORR_NORMED"],
        [4, cv2.TM_SQDIFF_NORMED, 0.3, "Binario+Canny + SQDIFF_NORMED"],

        [5, cv2.TM_CCOEFF_NORMED, 0.7, "Laplaciano + CCOEFF_NORMED"],
        [5, cv2.TM_CCORR_NORMED, 0.7, "Laplaciano + CCORR_NORMED"],
        [5, cv2.TM_SQDIFF_NORMED, 0.3, "Laplaciano + SQDIFF_NORMED"],
    ]

    # Variables globales originales para restaurar después
    original_metodo_prep = globals()['METODO_PREPROCESAMIENTO']
    original_metodo_match = globals()['METODO_MATCHING']
    original_umbral = globals()['UMBRAL_MATCHING']

    # Obtener imágenes objetivo
    imagenes = obtener_imagenes_objetivo()
    total_imagenes = len(imagenes)

    # Cargar template una vez
    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')

    resultados_analisis = []

    print(f"Probando {len(configuraciones)} configuraciones en {total_imagenes} imágenes...")
    print()
    
    # Deshabilitar progreso interno para el análisis exploratorio
    original_mostrar_progreso = globals()['MOSTRAR_PROGRESO']
    globals()['MOSTRAR_PROGRESO'] = False

    # Usar tqdm para el bucle principal de configuraciones
    for config in tqdm(configuraciones, desc="Configuraciones", unit="config"):
        metodo_prep, metodo_match, umbral, descripcion = config

        # Configurar parámetros globales
        globals()['METODO_PREPROCESAMIENTO'] = metodo_prep
        globals()['METODO_MATCHING'] = metodo_match
        globals()['UMBRAL_MATCHING'] = umbral

        try:
            # Cargar template con nueva configuración (suprimir output)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            template_data = cargar_template(ruta_template)
            
            sys.stdout = old_stdout

            detecciones_por_imagen = {}
            total_detecciones = 0

            # Procesar cada imagen con barra de progreso
            for ruta_imagen in tqdm(imagenes, desc=f"Imgs ({descripcion[:20]})", 
                                   unit="img", leave=False):
                nombre_imagen = os.path.basename(ruta_imagen)

                # Preprocesar imagen
                imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)

                # Búsqueda multi-escala (optimizada)
                detecciones, _ = buscar_coincidencias_multiescala(
                    imagen_procesada, template_data[0], template_data[1], template_data[2]
                )

                # Aplicar NMS
                detecciones_filtradas = aplicar_nms(detecciones)

                detecciones_por_imagen[nombre_imagen] = len(detecciones_filtradas)
                total_detecciones += len(detecciones_filtradas)

            # Calcular métricas
            imagenes_con_deteccion = sum(
                1 for count in detecciones_por_imagen.values() if count > 0)
            imagenes_con_una_deteccion = sum(
                1 for count in detecciones_por_imagen.values() if count == 1)
            imagenes_con_multiples = sum(
                1 for count in detecciones_por_imagen.values() if count > 1)

            precision = imagenes_con_una_deteccion / max(imagenes_con_deteccion, 1)
            recall = imagenes_con_deteccion / total_imagenes

            # Puntuación compuesta (prioriza 1 detección por imagen)
            score = (imagenes_con_una_deteccion * 2) - imagenes_con_multiples

            resultado = {
                'config': descripcion,
                'metodo_prep': metodo_prep,
                'metodo_match': metodo_match,
                'umbral': umbral,
                'total_detecciones': total_detecciones,
                'imagenes_con_deteccion': imagenes_con_deteccion,
                'imagenes_con_una': imagenes_con_una_deteccion,
                'imagenes_con_multiples': imagenes_con_multiples,
                'precision': precision,
                'recall': recall,
                'score': score,
                'detecciones_detalle': detecciones_por_imagen.copy()
            }

            resultados_analisis.append(resultado)

        except Exception as e:
            # Error silencioso para no interrumpir el análisis
            continue

    # Restaurar configuración de progreso
    globals()['MOSTRAR_PROGRESO'] = original_mostrar_progreso

    # Ordenar por score (mejor configuración primero)
    resultados_analisis.sort(key=lambda x: x['score'], reverse=True)

    # Mostrar resultados
    print("\n" + "=" * 80)
    print("=                        RESULTADOS DEL ANÁLISIS                        =")
    print("=" * 80)
    print()

    print("TOP 5 MEJORES CONFIGURACIONES:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Configuración':<35} {'Total':<6} {'1 Det':<5} {'Multi':<5} {'Score':<6}")
    print("-" * 80)

    for i, resultado in enumerate(resultados_analisis[:5], 1):
        print(f"{i:<4} {resultado['config']:<35} {resultado['total_detecciones']:<6} "
              f"{resultado['imagenes_con_una']:<5} {resultado['imagenes_con_multiples']:<5} "
              f"{resultado['score']:<6}")

    print()

    # Mostrar la mejor configuración en detalle
    if resultados_analisis:
        mejor = resultados_analisis[0]
        print("MEJOR CONFIGURACIÓN ENCONTRADA:")
        print("-" * 50)
        print(f"Método: {mejor['config']}")
        print(f"Total detecciones: {mejor['total_detecciones']}")
        print(
            f"Imágenes con 1 detección: {mejor['imagenes_con_una']}/{total_imagenes}")
        print(
            f"Imágenes con múltiples detecciones: {mejor['imagenes_con_multiples']}")
        print(f"Precisión: {mejor['precision']:.2f}")
        print(f"Recall: {mejor['recall']:.2f}")
        print(f"Score: {mejor['score']}")
        print()

        print("DETALLE POR IMAGEN:")
        for img, count in mejor['detecciones_detalle'].items():
            status = "✓" if count == 1 else "✗" if count == 0 else f"✗({count})"
            print(f"  {status} {img}: {count} detecciones")

        print()
        print("CONFIGURACIÓN RECOMENDADA:")
        print(f"METODO_PREPROCESAMIENTO = {mejor['metodo_prep']}")
        print(
            f"METODO_MATCHING = cv2.{['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED'][mejor['metodo_match']]}")
        print(f"UMBRAL_MATCHING = {mejor['umbral']}")
        
        # GUARDAR RESULTADOS DEL ANÁLISIS COMO IMÁGENES
        print()
        print("GUARDANDO RESULTADOS DEL ANÁLISIS EXPLORATORIO...")
        
        # Crear carpeta específica para análisis
        carpeta_analisis = 'resultados_analisis_exploratorio'
        os.makedirs(carpeta_analisis, exist_ok=True)
        
        # 1. Guardar gráfico de comparación de configuraciones
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Gráfico 1: Score por configuración
        configs = [r['config'] for r in resultados_analisis[:10]]  # Top 10
        scores = [r['score'] for r in resultados_analisis[:10]]
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores]
        
        ax1.barh(range(len(configs)), scores, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(configs)))
        ax1.set_yticklabels([c[:25] + '...' if len(c) > 25 else c for c in configs], fontsize=9)
        ax1.set_xlabel('Score (Mayor = Mejor)')
        ax1.set_title('TOP 10 CONFIGURACIONES\nScore por Método', weight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Destacar la mejor
        ax1.barh(0, scores[0], color='gold', alpha=0.9, edgecolor='black', linewidth=2)
        
        # Gráfico 2: Detecciones por configuración
        total_dets = [r['total_detecciones'] for r in resultados_analisis[:10]]
        una_det = [r['imagenes_con_una'] for r in resultados_analisis[:10]]
        multi_det = [r['imagenes_con_multiples'] for r in resultados_analisis[:10]]
        
        x = range(len(configs))
        width = 0.25
        
        ax2.bar([i - width for i in x], total_dets, width, label='Total detecciones', alpha=0.7, color='lightblue')
        ax2.bar([i for i in x], una_det, width, label='1 detección (ideal)', alpha=0.7, color='green')
        ax2.bar([i + width for i in x], multi_det, width, label='Múltiples (problema)', alpha=0.7, color='red')
        
        ax2.set_xlabel('Configuraciones')
        ax2.set_ylabel('Número de imágenes')
        ax2.set_title('DETECCIONES POR CONFIGURACIÓN', weight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in configs], rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{carpeta_analisis}/01_comparacion_configuraciones.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Guardar tabla de resultados detallada
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Crear tabla de datos
        tabla_datos = []
        headers = ['Rank', 'Configuración', 'Total', '1 Det', 'Multi', 'Score', 'Precisión', 'Recall']
        
        for i, r in enumerate(resultados_analisis[:10], 1):
            fila = [
                str(i),
                r['config'][:30] + '...' if len(r['config']) > 30 else r['config'],
                str(r['total_detecciones']),
                str(r['imagenes_con_una']),
                str(r['imagenes_con_multiples']),
                str(r['score']),
                f"{r['precision']:.2f}",
                f"{r['recall']:.2f}"
            ]
            tabla_datos.append(fila)
        
        # Crear tabla
        tabla = ax.table(cellText=tabla_datos, colLabels=headers, 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.2, 2)
        
        # Colorear la mejor configuración
        for j in range(len(headers)):
            tabla[(1, j)].set_facecolor('#FFD700')  # Dorado para la mejor
            tabla[(1, j)].set_text_props(weight='bold')
        
        # Colorear headers
        for j in range(len(headers)):
            tabla[(0, j)].set_facecolor('#4472C4')
            tabla[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.title('TABLA DETALLADA DE RESULTADOS - ANÁLISIS EXPLORATORIO', 
                 fontsize=16, weight='bold', pad=20)
        plt.savefig(f'{carpeta_analisis}/02_tabla_resultados.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Probar la mejor configuración y guardar resultados de muestra
        print("Aplicando mejor configuración y guardando ejemplos...")
        
        # Configurar con la mejor configuración encontrada
        globals()['METODO_PREPROCESAMIENTO'] = mejor['metodo_prep']
        globals()['METODO_MATCHING'] = mejor['metodo_match']
        globals()['UMBRAL_MATCHING'] = mejor['umbral']
        
        # Cargar template con la mejor configuración
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        template_data_mejor = cargar_template(ruta_template)
        
        sys.stdout = old_stdout
        
        # PROCESAR TODAS LAS IMÁGENES CON LA MEJOR CONFIGURACIÓN Y GUARDAR TODO
        # Cambiar carpeta de resultados temporalmente
        carpeta_original = globals()['CARPETA_RESULTADOS']
        globals()['CARPETA_RESULTADOS'] = f'{carpeta_analisis}/mejor_configuracion_resultados'
        os.makedirs(globals()['CARPETA_RESULTADOS'], exist_ok=True)
        
        print(f"Procesando todas las imágenes con la mejor configuración...")
        print(f"Guardando preprocesamiento, mapas de calor y resultados finales...")
        
        # Procesar cada imagen con la mejor configuración y guardar TODO
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.basename(ruta_imagen)
            print(f"  [{i}/{len(imagenes)}] Procesando: {nombre_imagen}")
            
            try:
                # Preprocesar imagen
                imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)
                
                # Visualizar preprocesamiento (IGUAL QUE EL MÉTODO PRINCIPAL)
                visualizar_preprocesamiento(
                    template_data_mejor[0], template_data_mejor[1], template_data_mejor[2],
                    imagen_original, imagen_procesada, nombre_imagen
                )
                
                # Búsqueda multi-escala
                detecciones, mapas_resultado = buscar_coincidencias_multiescala(
                    imagen_procesada, template_data_mejor[0], template_data_mejor[1], template_data_mejor[2]
                )
                
                # Visualizar mapas de coincidencias (IGUAL QUE EL MÉTODO PRINCIPAL)
                visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)
                
                # Aplicar NMS
                detecciones_filtradas = aplicar_nms(detecciones)
                
                # Visualizar resultado final (IGUAL QUE EL MÉTODO PRINCIPAL)
                visualizar_resultado_final(
                    imagen_original, detecciones, detecciones_filtradas, nombre_imagen
                )
                
            except Exception as e:
                print(f"    ⚠️ Error procesando {nombre_imagen}: {str(e)}")
                continue
        
        # Restaurar carpeta original
        globals()['CARPETA_RESULTADOS'] = carpeta_original
        
        # Guardar resultado de muestra EN LA CARPETA PRINCIPAL DE ANÁLISIS también
        if imagenes:
            imagen_muestra = imagenes[0]  # Primera imagen
            nombre_muestra = os.path.basename(imagen_muestra)
            
            # Procesar con configuración óptima
            imagen_original, imagen_procesada = preprocesar_imagen(imagen_muestra)
            detecciones, mapas_resultado = buscar_coincidencias_multiescala(
                imagen_procesada, template_data_mejor[0], template_data_mejor[1], template_data_mejor[2]
            )
            detecciones_filtradas = aplicar_nms(detecciones)
            
            # Guardar resultado de muestra
            plt.figure(figsize=(12, 8))
            plt.imshow(imagen_original, cmap='gray')
            plt.title(f'RESULTADO CON MEJOR CONFIGURACIÓN\n{mejor["config"]}\n'
                     f'Imagen: {nombre_muestra} - {len(detecciones_filtradas)} detecciones', 
                     fontsize=14, weight='bold')
            
            # Dibujar detecciones con colores diferentes para máximos y mínimos
            maximos = [d for d in detecciones_filtradas if d['version'] == 'directo']
            minimos = [d for d in detecciones_filtradas if d['version'] == 'invertido']
            
            for i, det in enumerate(maximos):
                rect = plt.Rectangle((det['x'], det['y']), det['ancho'], det['alto'],
                                   linewidth=3, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(det['x'], det['y']-5, f'MAX #{i+1}\n{det["confianza"]:.3f}',
                        color='lime', fontsize=10, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            for i, det in enumerate(minimos):
                rect = plt.Rectangle((det['x'], det['y']), det['ancho'], det['alto'],
                                   linewidth=3, edgecolor='magenta', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(det['x'], det['y']-5, f'MIN #{i+1}\n{det["confianza"]:.3f}',
                        color='magenta', fontsize=10, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # Agregar información de la configuración
            info_text = f"""MEJOR CONFIGURACIÓN:
Método: {mejor['config']}
Preprocesamiento: {mejor['metodo_prep']}
Matching: {['SQDIFF', 'SQDIFF_NORMED', 'CCORR', 'CCORR_NORMED', 'CCOEFF', 'CCOEFF_NORMED'][mejor['metodo_match']]}
Umbral: {mejor['umbral']}
Score: {mejor['score']}
Detecciones: {len(maximos)} MAX, {len(minimos)} MIN"""
            
            plt.figtext(0.02, 0.02, info_text, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9))
            
            plt.axis('off')
            plt.savefig(f'{carpeta_analisis}/03_ejemplo_mejor_configuracion.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Resultados del análisis guardados en: {carpeta_analisis}/")
        print("   - 01_comparacion_configuraciones.png: Gráficos comparativos")
        print("   - 02_tabla_resultados.png: Tabla detallada de todas las configuraciones")
        print("   - 03_ejemplo_mejor_configuracion.png: Ejemplo con la mejor configuración")
        print(f"   - mejor_configuracion_resultados/: TODOS los preprocesados, mapas y resultados")
        print(f"     (igual que el método principal pero con la configuración óptima)")

    # Restaurar configuración original
    globals()['METODO_PREPROCESAMIENTO'] = original_metodo_prep
    globals()['METODO_MATCHING'] = original_metodo_match
    globals()['UMBRAL_MATCHING'] = original_umbral

    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO. Use --analisis para ejecutar este modo.")
    print("=" * 80)


# ===================================================================
# FUNCIÓN PRINCIPAL
# ===================================================================


def obtener_imagenes_objetivo() -> List[str]:
    """
    Obtiene la lista de imágenes que contienen 'logo' o 'reto' en el nombre.
    """
    patrones = ['*logo*', '*retro*', '*LOGO*']
    imagenes = []

    for patron in patrones:
        # Buscar tanto PNG como JPG
        imagenes.extend(glob.glob(os.path.join(
            PATH_IMAGENES, patron + '.png')))
        imagenes.extend(glob.glob(os.path.join(
            PATH_IMAGENES, patron + '.jpg')))
        imagenes.extend(glob.glob(os.path.join(
            PATH_IMAGENES, patron + '.jpeg')))

    return imagenes


def procesar_imagen(ruta_imagen: str, template_data: Tuple):
    """
    Procesa una sola imagen con el template.
    """
    nombre_imagen = os.path.basename(ruta_imagen)
    
    # Solo mostrar headers detallados si no hay barra de progreso global
    if not MOSTRAR_PROGRESO:
        print(f"\n" + "="*60)
        print(f"PROCESANDO: {nombre_imagen}")
        print("="*60)

    # Desempaquetar datos del template
    template_procesado, template_procesado_inv, mascara = template_data

    # Cargar template original para comparación de escalas
    ruta_template = os.path.join(PATH_TEMPLATE, 'pattern.png')
    template_original = cv2.imread(ruta_template, cv2.IMREAD_GRAYSCALE)

    # Preprocesar imagen
    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)

    if not MOSTRAR_PROGRESO:
        print(f"Tamaño template: {template_procesado.shape[1]}x{template_procesado.shape[0]} pixeles")
        print(f"Tamaño imagen: {imagen_original.shape[1]}x{imagen_original.shape[0]} pixeles")
        print(f"Buscando template EN la imagen...")

    # Visualizar preprocesamiento
    if not MOSTRAR_PROGRESO:
        print("Guardando preprocesamiento...")
    visualizar_preprocesamiento(
        template_procesado, template_procesado_inv, mascara,
        imagen_original, imagen_procesada, nombre_imagen
    )

    # Búsqueda multi-escala
    if not MOSTRAR_PROGRESO:
        print("Realizando busqueda multi-escala...")
    detecciones, mapas_resultado = buscar_coincidencias_multiescala(
        imagen_procesada, template_procesado, template_procesado_inv, mascara
    )

    # Visualizar mapas de coincidencias (TODOS los mapas de matching en un plot)
    if not MOSTRAR_PROGRESO:
        print("Guardando mapas de matching...")
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)

    # Aplicar NMS
    if not MOSTRAR_PROGRESO:
        print("Aplicando filtro de detecciones (NMS)...")
    detecciones_filtradas = aplicar_nms(detecciones)

    # Visualizar resultado final (solo mejor detección)
    if not MOSTRAR_PROGRESO:
        print("Guardando mejor detección...")
    visualizar_resultado_final(
        imagen_original, detecciones, detecciones_filtradas, nombre_imagen
    )

    # Visualizar comparación de escalas (nuevo plot)
    if not MOSTRAR_PROGRESO:
        print("Guardando comparación de escalas...")
    visualizar_comparacion_escalas(
        imagen_original, template_original, mapas_resultado, nombre_imagen
    )

    return detecciones_filtradas


def main():
    """
    Función principal del script.
    """
    print("=" + "="*58 + "=")
    print("=          TEMPLATE MATCHING - DETECCION DE LOGOS          =")
    print("=" + "="*58 + "=")
    print()
    print("CONFIGURACION ACTUAL:")
    print(f"   Metodo de matching: {METODO_MATCHING}")
    print(f"   Metodo de preprocesamiento: {METODO_PREPROCESAMIENTO}")
    print(f"   Umbral de confianza: {UMBRAL_MATCHING}")
    print(f"   Umbral filtro NMS: {UMBRAL_IOU_NMS}")
    print(
        f"   Rango de escalas: {ESCALA_MIN}x - {ESCALA_MAX}x (paso: {PASO_ESCALA})")
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
    if USAR_MULTIPROCESSING:
        print(f"OPTIMIZACIONES ACTIVAS:")
        print(f"   - Multiprocessing: {NUM_PROCESOS} procesos")
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
        buenas = [d for d in detecciones if d['version'] == 'directo']
        malas = [d for d in detecciones if d['version'] == 'invertido']
        
        icono = "✓" if len(detecciones) > 0 else "✗"
        print(f"{icono} {nombre_img:<25} -> {len(detecciones):>3} total | {len(buenas):>2} BUENAS | {len(malas):>2} MALAS")
        
        total_detecciones += len(detecciones)
        total_buenas += len(buenas)
        total_malas += len(malas)

    print("-" * 60)
    print(f"TOTAL DETECCIONES: {total_detecciones} (BUENAS: {total_buenas}, MALAS: {total_malas})")
    print(f"Método usado: {['SQDIFF', 'SQDIFF_NORMED', 'CCORR', 'CCORR_NORMED', 'CCOEFF', 'CCOEFF_NORMED'][METODO_MATCHING]}")
    if METODO_MATCHING == cv2.TM_SQDIFF_NORMED:
        print("INTERPRETACIÓN: BUENAS = mínimos del mapa (valores cercanos a 0), MALAS = máximos del mapa (valores cercanos a 1)")
    elif METODO_MATCHING == cv2.TM_SQDIFF:
        print("INTERPRETACIÓN: BUENAS = mínimos del mapa (valores bajos), MALAS = máximos del mapa (valores altos)")
    else:
        print("INTERPRETACIÓN: BUENAS = máximos del mapa (valores altos), MALAS = mínimos del mapa (valores bajos)")
    print("Template usado: pattern.png")
    print("Imagenes procesadas: TP3/images/")
    print("Resultados guardados en:", CARPETA_RESULTADOS)
    
    # Información de rendimiento
    if USAR_MULTIPROCESSING:
        print(f"Procesamiento acelerado con {NUM_PROCESOS} cores")
    
    print("Proceso completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    # Verificar si se quiere ejecutar análisis exploratorio
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analisis":
        analisis_exploratorio()
    else:
        main()
