import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
from typing import List, Tuple, Dict

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
# Para Canny (método 3): TM_CCORR_NORMED es el MEJOR (CONFIRMADO)
METODO_MATCHING = cv2.TM_SQDIFF_NORMED

# Parámetros de Pirámide de Escala
ESCALA_MIN = 0.05
ESCALA_MAX = 2.7
PASO_ESCALA = 0.1

# Parámetros de Canny
UMBRAL_CANNY_MIN = 50
UMBRAL_CANNY_MAX = 200

# Parámetros de Filtro de Ruido (antes de Canny)
# Filtro Gaussiano para reducir ruido antes de detectar bordes
FILTRAR_RUIDO = True  # Activar/desactivar filtro de ruido
KERNEL_GAUSSIANO = (5, 5)  # Tamaño del kernel Gaussiano (debe ser impar)
SIGMA_GAUSSIANO = 1.4  # Desviación estándar del filtro Gaussiano

# Parámetros de Detección y NMS
# NOTA: Si usas TM_SQDIFF_NORMED, este umbral debe ser bajo (ej. 0.1).
# Para TM_CCORR_NORMED con Canny: umbral medio-alto (0.4-0.7)
# Para TM_CCOEFF_NORMED: umbral medio (0.3-0.6)
UMBRAL_MATCHING = 0.35  # ÓPTIMO para TM_CCORR_NORMED + Canny
UMBRAL_IOU_NMS = 0.01

# METODOS DE PREPROCESAMIENTO DISPONIBLES
# 1: Solo escala de grises
# 2: Escala de grises + binarización
# 3: Escala de grises + Canny
# 4: Escala de grises + binarización + Canny (RECOMENDADO)
# 5: Pirámides Laplacianas (reduce información, enfoca en bordes importantes)
# Método de preprocesamiento (1-5)
METODO_PREPROCESAMIENTO = 3

# Parámetros para Pirámides Laplacianas (método 5)
NIVELES_PIRAMIDE = 2  # Número de niveles de la pirámide
USAR_PIRAMIDE_GAUSSIANA = True  # Usar Gaussiana antes de Laplaciana

# Parámetros de umbralización para binarización
UMBRAL_BINARIO = 127

# Parámetros de visualización
TAMAÑO_FIGURA = (15, 10)
DPI_FIGURA = 100
CARPETA_RESULTADOS = 'resultados_template_matching'

# ===================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ===================================================================


def crear_piramide_laplaciana(imagen: np.ndarray) -> np.ndarray:
    """
    Crea una pirámide laplaciana para reducir información y enfocar en bordes importantes.

    Args:
        imagen: Imagen en escala de grises

    Returns:
        Imagen procesada con pirámide laplaciana
    """
    # Convertir a float32 para cálculos precisos
    img = imagen.astype(np.float32)

    # Crear pirámide gaussiana
    piramide_gaussiana = [img]

    # Generar niveles de la pirámide gaussiana
    for i in range(NIVELES_PIRAMIDE):
        # Reducir resolución (downsample)
        img = cv2.pyrDown(img)
        piramide_gaussiana.append(img)

    # Crear pirámide laplaciana
    piramide_laplaciana = []

    # Para cada nivel (excepto el último)
    for i in range(NIVELES_PIRAMIDE):
        # Expandir el nivel inferior
        expandido = cv2.pyrUp(piramide_gaussiana[i + 1])

        # Ajustar tamaño si es necesario (puede haber diferencias de 1 pixel)
        if expandido.shape != piramide_gaussiana[i].shape:
            expandido = cv2.resize(expandido,
                                   (piramide_gaussiana[i].shape[1], piramide_gaussiana[i].shape[0]))

        # Calcular diferencia (nivel laplaciano)
        laplaciano = piramide_gaussiana[i] - expandido
        piramide_laplaciana.append(laplaciano)

    # El último nivel es simplemente el último gaussiano
    piramide_laplaciana.append(piramide_gaussiana[-1])

    # Reconstruir imagen combinando niveles laplacianos
    # Comenzar con el nivel más alto
    resultado = piramide_laplaciana[-1]

    # Reconstruir hacia arriba
    for i in range(NIVELES_PIRAMIDE - 1, -1, -1):
        # Expandir resultado actual
        resultado = cv2.pyrUp(resultado)

        # Ajustar tamaño si es necesario
        if resultado.shape != piramide_laplaciana[i].shape:
            resultado = cv2.resize(resultado,
                                   (piramide_laplaciana[i].shape[1], piramide_laplaciana[i].shape[0]))

        # Sumar el nivel laplaciano
        resultado += piramide_laplaciana[i]

    # Normalizar y convertir a formato adecuado
    resultado = np.clip(resultado, 0, 255)

    # Si queremos enfocarnos solo en los cambios significativos,
    # podemos usar solo el primer nivel laplaciano
    if USAR_PIRAMIDE_GAUSSIANA:
        return resultado
    else:
        # Usar solo el primer nivel laplaciano (más enfoque en bordes)
        primer_nivel = piramide_laplaciana[0]
        # Normalizar a rango 0-255
        primer_nivel = cv2.normalize(
            primer_nivel, None, 0, 255, cv2.NORM_MINMAX)
        return primer_nivel.astype(np.float32)


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


def buscar_coincidencias_multiescala(imagen_procesada: np.ndarray,
                                     template_procesado: np.ndarray,
                                     template_procesado_inv: np.ndarray,
                                     mascara: np.ndarray) -> Tuple[List[Dict], List[Tuple]]:
    """
    Realiza búsqueda de template en múltiples escalas.

    Returns:
        Lista de detecciones con información de posición, escala y confianza
    """
    detecciones = []
    mapas_resultado = []

    # Generar escalas
    escalas = np.arange(ESCALA_MIN, ESCALA_MAX + PASO_ESCALA, PASO_ESCALA)
    print(f"Probando {len(escalas)} escalas diferentes...")

    for escala in escalas:
        # Redimensionar templates y máscara
        template_escalado, mascara_escalada = redimensionar_template_y_mascara(
            template_procesado, mascara, escala)
        template_inv_escalado, _ = redimensionar_template_y_mascara(
            template_procesado_inv, mascara, escala)

        if template_escalado is None or template_inv_escalado is None:
            continue

        # Verificar que el template escalado no sea más grande que la imagen
        if (template_escalado.shape[0] > imagen_procesada.shape[0] or
                template_escalado.shape[1] > imagen_procesada.shape[1]):
            continue

        # Template matching para ambas versiones
        for template_actual, nombre_version in [(template_escalado, "directo"),
                                                (template_inv_escalado, "invertido")]:

            try:
                # Convertir a float32 para evitar problemas
                img_match = imagen_procesada.astype(np.float32)
                temp_match = template_actual.astype(np.float32)

                # Intentar con máscara primero, luego sin máscara si falla
                resultado = None
                try:
                    mask_match = mascara_escalada.astype(np.uint8)
                    resultado = cv2.matchTemplate(
                        img_match, temp_match, METODO_MATCHING, mask=mask_match)
                except:
                    # Si falla con máscara, usar sin máscara
                    resultado = cv2.matchTemplate(
                        img_match, temp_match, METODO_MATCHING)

                # Validar resultado
                if resultado is None or np.any(np.isnan(resultado)) or np.any(np.isinf(resultado)):
                    continue

                mapas_resultado.append((resultado, escala, nombre_version))

                # Buscar coincidencias según el método de matching
                if METODO_MATCHING in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    ubicaciones = np.where(resultado <= UMBRAL_MATCHING)
                    valores_confianza = resultado[ubicaciones]
                else:
                    ubicaciones = np.where(resultado >= UMBRAL_MATCHING)
                    valores_confianza = resultado[ubicaciones]

                # Agregar detecciones válidas
                for i, (y, x) in enumerate(zip(ubicaciones[0], ubicaciones[1])):
                    confianza = float(valores_confianza[i])

                    # Validar confianza
                    if np.isnan(confianza) or np.isinf(confianza):
                        continue

                    detecciones.append({
                        'x': int(x),
                        'y': int(y),
                        'ancho': template_actual.shape[1],
                        'alto': template_actual.shape[0],
                        'confianza': confianza,
                        'escala': escala,
                        'version': nombre_version
                    })

            except Exception as e:
                print(
                    f"Error en template matching escala {escala:.1f}, {nombre_version}: {e}")
                continue

    print(f"Total detecciones encontradas: {len(detecciones)}")
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
    Aplica Non-Maximum Suppression a las detecciones.
    """
    if not detecciones:
        return []

    # Ordenar por confianza (mayor a menor para la mayoría de métodos)
    if METODO_MATCHING in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        detecciones_ordenadas = sorted(
            detecciones, key=lambda x: x['confianza'])
    else:
        detecciones_ordenadas = sorted(
            detecciones, key=lambda x: x['confianza'], reverse=True)

    detecciones_filtradas = []

    while detecciones_ordenadas:
        # Tomar la detección con mejor confianza
        mejor_deteccion = detecciones_ordenadas.pop(0)
        detecciones_filtradas.append(mejor_deteccion)

        # Eliminar detecciones superpuestas
        detecciones_restantes = []
        for deteccion in detecciones_ordenadas:
            iou = calcular_iou_modificado(mejor_deteccion, deteccion)
            if iou <= UMBRAL_IOU_NMS:
                detecciones_restantes.append(deteccion)

        detecciones_ordenadas = detecciones_restantes

    return detecciones_filtradas

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
    Visualiza el proceso de preprocesamiento de forma clara y organizada.
    """
    # Crear carpeta si no existe
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=DPI_FIGURA)

    # Título con método de preprocesamiento
    metodos = {
        1: "Escala de Grises",
        2: "Escala de Grises + Binarizacion",
        3: "Escala de Grises + Filtro de Ruido + Canny",
        4: "Escala de Grises + Binarizacion + Filtro de Ruido + Canny",
        5: "Piramides Laplacianas (Reduccion de Informacion)"
    }
    metodo_nombre = metodos.get(METODO_PREPROCESAMIENTO, "Desconocido")
    fig.suptitle(f'PREPROCESAMIENTO - {nombre_imagen}\nMETODO: {metodo_nombre}',
                 fontsize=16, weight='bold')

    # FILA 1: TEMPLATE Y SUS PROCESADOS
    axes[0, 0].imshow(template_procesado, cmap='gray')
    axes[0, 0].set_title('TEMPLATE PROCESADO\n(Lo que buscamos)',
                         fontsize=12, weight='bold', color='blue')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(template_procesado_inv, cmap='gray')
    axes[0, 1].set_title(
        'Template Invertido\n(Version alternativa)', fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mascara, cmap='gray')
    axes[0, 2].set_title(
        'Mascara del Template\n(Area de busqueda)', fontsize=10)
    axes[0, 2].axis('off')

    # FILA 2: IMAGEN OBJETIVO Y SUS PROCESADOS
    axes[1, 0].imshow(imagen_original, cmap='gray')
    axes[1, 0].set_title('IMAGEN ORIGINAL\n(Donde buscamos)',
                         fontsize=12, weight='bold', color='green')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(imagen_procesada, cmap='gray')
    axes[1, 1].set_title('Imagen Procesada\n(Para matching)', fontsize=10)
    axes[1, 1].axis('off')

    # Información del proceso
    filtro_info = f"\nFiltro de ruido: {'SI' if FILTRAR_RUIDO else 'NO'}"
    if FILTRAR_RUIDO:
        filtro_info += f"\nKernel Gaussiano: {KERNEL_GAUSSIANO}\nSigma: {SIGMA_GAUSSIANO}"

    info_text = f"""
METODO {METODO_PREPROCESAMIENTO}: {metodo_nombre}

TAMAÑOS:
Template: {template_procesado.shape[1]}x{template_procesado.shape[0]} px
Imagen: {imagen_original.shape[1]}x{imagen_original.shape[0]} px

PARAMETROS:
Umbral matching: {UMBRAL_MATCHING}
Umbral NMS: {UMBRAL_IOU_NMS}
Escalas: {ESCALA_MIN}-{ESCALA_MAX} (paso {PASO_ESCALA}){filtro_info}
"""
    axes[1, 2].text(0.05, 0.95, info_text, ha='left', va='top',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Guardar en lugar de mostrar
    nombre_base = os.path.splitext(nombre_imagen)[0]
    plt.savefig(f'{CARPETA_RESULTADOS}/01_preprocesamiento_{nombre_base}.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_mapas_coincidencias(mapas_resultado: List[Tuple], nombre_imagen: str):
    """
    Visualiza los mapas de coincidencias de forma clara.
    """
    if not mapas_resultado:
        return

    # Mostrar solo los mejores mapas (3 escalas más representativas)
    num_mapas = min(3, len(mapas_resultado))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI_FIGURA)
    if num_mapas == 1:
        axes = [axes]

    fig.suptitle(f'MAPAS DE CALOR - Imagen: {nombre_imagen}\n'
                 f'(Rojo = Mayor similitud, Azul = Menor similitud)', fontsize=14, weight='bold')

    # Seleccionar mapas representativos (inicio, medio, fin)
    indices_seleccionados = []
    if len(mapas_resultado) >= 3:
        indices_seleccionados = [
            0, len(mapas_resultado)//2, len(mapas_resultado)-1]
    else:
        indices_seleccionados = list(range(len(mapas_resultado)))

    for i, idx in enumerate(indices_seleccionados[:num_mapas]):
        mapa, escala, version = mapas_resultado[idx]

        # Mostrar mapa como imagen de calor
        im = axes[i].imshow(mapa, cmap='hot', interpolation='nearest')
        axes[i].set_title(
            f'Escala: {escala:.1f}x\n({version})', fontsize=12, weight='bold')

        # Encontrar y marcar la mejor coincidencia
        if METODO_MATCHING in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_match = np.unravel_index(np.argmin(mapa), mapa.shape)
            best_value = np.min(mapa)
            axes[i].plot(best_match[1], best_match[0], 'bo', markersize=10,
                         markeredgecolor='white', markeredgewidth=2, label=f'Mejor: {best_value:.3f}')
        else:
            best_match = np.unravel_index(np.argmax(mapa), mapa.shape)
            best_value = np.max(mapa)
            axes[i].plot(best_match[1], best_match[0], 'bo', markersize=10,
                         markeredgecolor='white', markeredgewidth=2, label=f'Mejor: {best_value:.3f}')

        axes[i].legend(loc='upper right')

        # Agregar colorbar
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Similitud', rotation=270, labelpad=15)

    plt.tight_layout()

    # Guardar en lugar de mostrar
    nombre_base = os.path.splitext(nombre_imagen)[0]
    plt.savefig(f'{CARPETA_RESULTADOS}/02_mapas_calor_{nombre_base}.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()


def visualizar_resultado_final(imagen_original: np.ndarray,
                               detecciones_antes_nms: List[Dict],
                               detecciones_despues_nms: List[Dict],
                               nombre_imagen: str):
    """
    Visualiza el resultado final de forma clara y comprensible.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI_FIGURA)
    fig.suptitle(
        f'RESULTADOS FINALES - Imagen: {nombre_imagen}', fontsize=16, weight='bold')

    # IMAGEN 1: Original sin detecciones
    axes[0].imshow(imagen_original, cmap='gray')
    axes[0].set_title('IMAGEN ORIGINAL\n(Sin procesar)',
                      fontsize=12, weight='bold', color='blue')
    axes[0].axis('off')

    # IMAGEN 2: Todas las detecciones encontradas
    axes[1].imshow(imagen_original, cmap='gray')
    axes[1].set_title(f'TODAS LAS DETECCIONES\n({len(detecciones_antes_nms)} encontradas)',
                      fontsize=12, weight='bold', color='orange')

    for i, det in enumerate(detecciones_antes_nms):
        rect = plt.Rectangle((det['x'], det['y']), det['ancho'], det['alto'],
                             linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.7)
        axes[1].add_patch(rect)
        # Mostrar confianza solo para las primeras detecciones para no saturar
        if i < 10:
            axes[1].text(det['x'], det['y']-2, f'{det["confianza"]:.2f}',
                         color='yellow', fontsize=6, weight='bold')

    axes[1].axis('off')

    # IMAGEN 3: Detecciones finales después de filtrar
    axes[2].imshow(imagen_original, cmap='gray')
    axes[2].set_title(f'DETECCIONES FINALES\n({len(detecciones_despues_nms)} despues de filtrar)',
                      fontsize=12, weight='bold', color='green')

    for i, det in enumerate(detecciones_despues_nms):
        rect = plt.Rectangle((det['x'], det['y']), det['ancho'], det['alto'],
                             linewidth=3, edgecolor='lime', facecolor='none')
        axes[2].add_patch(rect)

        # Agregar número de detección y confianza
        axes[2].text(det['x'], det['y']-5, f'#{i+1}\n{det["confianza"]:.3f}\n{det["escala"]:.1f}x',
                     color='lime', fontsize=10, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

    axes[2].axis('off')

    # Agregar información adicional
    info_text = f"""
RESUMEN:
- Template usado: pattern.png
- Metodo: {METODO_MATCHING}
- Escalas probadas: {ESCALA_MIN}-{ESCALA_MAX}x
- Umbral confianza: {UMBRAL_MATCHING}
- Filtro NMS: {UMBRAL_IOU_NMS}
"""
    plt.figtext(0.02, 0.02, info_text, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Guardar en lugar de mostrar
    nombre_base = os.path.splitext(nombre_imagen)[0]
    plt.savefig(f'{CARPETA_RESULTADOS}/03_resultados_finales_{nombre_base}.png',
                bbox_inches='tight', dpi=DPI_FIGURA)
    plt.close()

    # Mostrar detalles de cada detección final
    if detecciones_despues_nms:
        print(f"\nDETALLES DE DETECCIONES FINALES EN {nombre_imagen}:")
        print("="*60)
        for i, det in enumerate(detecciones_despues_nms):
            print(f"Deteccion #{i+1}:")
            print(f"   Posicion: ({det['x']}, {det['y']})")
            print(f"   Tamaño: {det['ancho']}x{det['alto']} px")
            print(f"   Confianza: {det['confianza']:.4f}")
            print(f"   Escala: {det['escala']:.1f}x")
            print(f"   Version: {det['version']}")
            print("-" * 40)

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

    print(
        f"Probando {len(configuraciones)} configuraciones en {total_imagenes} imágenes...")
    print()

    for i, config in enumerate(configuraciones, 1):
        metodo_prep, metodo_match, umbral, descripcion = config

        print(f"[{i}/{len(configuraciones)}] Probando: {descripcion}")

        # Configurar parámetros globales
        globals()['METODO_PREPROCESAMIENTO'] = metodo_prep
        globals()['METODO_MATCHING'] = metodo_match
        globals()['UMBRAL_MATCHING'] = umbral

        try:
            # Cargar template con nueva configuración
            template_data = cargar_template(ruta_template)

            detecciones_por_imagen = {}
            total_detecciones = 0

            # Procesar cada imagen silenciosamente
            for ruta_imagen in imagenes:
                nombre_imagen = os.path.basename(ruta_imagen)

                # Preprocesar imagen
                imagen_original, imagen_procesada = preprocesar_imagen(
                    ruta_imagen)

                # Búsqueda multi-escala (sin prints)
                detecciones, _ = buscar_coincidencias_multiescala(
                    imagen_procesada, template_data[0], template_data[1], template_data[2]
                )

                # Aplicar NMS
                detecciones_filtradas = aplicar_nms(detecciones)

                detecciones_por_imagen[nombre_imagen] = len(
                    detecciones_filtradas)
                total_detecciones += len(detecciones_filtradas)

            # Calcular métricas
            imagenes_con_deteccion = sum(
                1 for count in detecciones_por_imagen.values() if count > 0)
            imagenes_con_una_deteccion = sum(
                1 for count in detecciones_por_imagen.values() if count == 1)
            imagenes_con_multiples = sum(
                1 for count in detecciones_por_imagen.values() if count > 1)

            precision = imagenes_con_una_deteccion / \
                max(imagenes_con_deteccion, 1)
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

            print(
                f"  → {total_detecciones} detecciones totales, {imagenes_con_una_deteccion} imágenes con 1 detección")

        except Exception as e:
            print(f"  → ERROR: {e}")
            continue

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
    print(f"\n" + "="*60)
    print(f"PROCESANDO: {nombre_imagen}")
    print("="*60)

    # Desempaquetar datos del template
    template_procesado, template_procesado_inv, mascara = template_data

    # Preprocesar imagen
    imagen_original, imagen_procesada = preprocesar_imagen(ruta_imagen)

    print(
        f"Tamaño template: {template_procesado.shape[1]}x{template_procesado.shape[0]} pixeles")
    print(
        f"Tamaño imagen: {imagen_original.shape[1]}x{imagen_original.shape[0]} pixeles")
    print(f"Buscando template EN la imagen...")

    # Visualizar preprocesamiento
    print("Guardando preprocesamiento...")
    visualizar_preprocesamiento(
        template_procesado, template_procesado_inv, mascara,
        imagen_original, imagen_procesada, nombre_imagen
    )

    # Búsqueda multi-escala
    print("Realizando busqueda multi-escala...")
    detecciones, mapas_resultado = buscar_coincidencias_multiescala(
        imagen_procesada, template_procesado, template_procesado_inv, mascara
    )

    # Visualizar mapas de coincidencias
    print("Guardando mapas de calor...")
    visualizar_mapas_coincidencias(mapas_resultado, nombre_imagen)

    # Aplicar NMS
    print("Aplicando filtro de detecciones (NMS)...")
    detecciones_filtradas = aplicar_nms(detecciones)

    # Visualizar resultado final
    print("Guardando resultados finales...")
    visualizar_resultado_final(
        imagen_original, detecciones, detecciones_filtradas, nombre_imagen
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

    # Procesar cada imagen
    resultados_totales = {}
    for i, ruta_imagen in enumerate(imagenes, 1):
        print(f"\nPROGRESO: {i}/{len(imagenes)}")
        detecciones = procesar_imagen(ruta_imagen, template_data)
        resultados_totales[os.path.basename(ruta_imagen)] = detecciones

    # Resumen final
    print("\n" + "=" + "="*58 + "=")
    print("=                    RESUMEN FINAL                        =")
    print("=" + "="*58 + "=")

    total_detecciones = 0
    for nombre_img, detecciones in resultados_totales.items():
        icono = "OK" if len(detecciones) > 0 else "NO"
        print(f"{icono} {nombre_img:<25} -> {len(detecciones):>3} detecciones")
        total_detecciones += len(detecciones)

    print("-" * 60)
    print(f"TOTAL DETECCIONES ENCONTRADAS: {total_detecciones}")
    print("Template usado: pattern.png")
    print("Imagenes procesadas: TP3/images/")
    print("Resultados guardados en:", CARPETA_RESULTADOS)
    print("Proceso completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    # Verificar si se quiere ejecutar análisis exploratorio
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analisis":
        analisis_exploratorio()
    else:
        main()
