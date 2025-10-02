import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================================================================
# PARÁMETROS DE CONFIGURACIÓN
# ===================================================================

# Rutas
PATH_TEMPLATE = 'TP3/template/pattern.png'
PATH_IMAGEN = 'TP3/images/coca_multi.png'
PATH_SALIDA = 'resultados_sift/'

# Parámetros SIFT (optimizado para bordes y esquinas)
SIFT_NFEATURES = 300000  # Aún más features para capturar más detalles
SIFT_NOCTAVELAYERS = 6  # Más octavas para múltiples escalas
SIFT_CONTRAST_THRESHOLD = 0.01  # MUY bajo para detectar bordes débiles
SIFT_EDGE_THRESHOLD = 50  # MUY permisivo - acepta prácticamente todos los bordes
SIFT_SIGMA = 1.0  # Más sensible a detalles finos

# Parámetros del algoritmo SIFT (sin ratio test - matching directo)
DISTANCE_THRESHOLD = 320    # Aún más permisivo para más matches
MIN_MATCHES = 4             # Mínimo para homografía
CONFIDENCE_THRESHOLD = 0.1  # Permisivo para múltiples instancias
CLUSTERING_DISTANCE = 100   # Más permisivo para capturar logos completos
MIN_CLUSTER_SIZE = 6        # Mínimo para logos robustos

# Parámetros NMS
IOU_THRESHOLD = 0.5

# Visualización
SAVE_VISUALIZATIONS = True
SHOW_PLOTS = True

# ===================================================================
# FUNCIONES AUXILIARES
# ===================================================================

def create_output_dir():
    if not os.path.exists(PATH_SALIDA):
        os.makedirs(PATH_SALIDA)

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_intersect = max(x1_1, x1_2)
    y1_intersect = max(y1_1, y1_2)
    x2_intersect = min(x2_1, x2_2)
    y2_intersect = min(y2_1, y2_2)
    
    if x1_intersect >= x2_intersect or y1_intersect >= y2_intersect:
        return 0.0
    
    area_intersect = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_union = area_box1 + area_box2 - area_intersect
    
    return area_intersect / area_union if area_union > 0 else 0.0

def non_max_suppression(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        ious = [calculate_iou(boxes[current], boxes[i]) for i in indices[1:]]
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return keep

def resize_template(template, scale):
    h, w = template.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(template, (new_w, new_h))

def preprocess_template(template_gray):
    """Preprocesar el template con contraste suave - sin inversión"""
    # Solo aumento de contraste suave para mejorar keypoints sin deformar
    alpha = 1.4  # Contraste moderado
    beta = 0     # Sin brillo
    enhanced = cv2.convertScaleAbs(template_gray, alpha=alpha, beta=beta)
    
    print(f"Template: original mean={template_gray.mean():.1f} -> procesado mean={enhanced.mean():.1f}")
    return enhanced

def preprocess_image(image_gray):
    """Preprocesar la imagen con contraste suave"""
    # Solo aumento de contraste suave para mejorar keypoints sin artefactos
    alpha = 1.3  # Contraste suave
    beta = 0     # Sin brillo
    enhanced = cv2.convertScaleAbs(image_gray, alpha=alpha, beta=beta)
    
    print(f"Imagen: original mean={image_gray.mean():.1f} -> procesada mean={enhanced.mean():.1f}")
    return enhanced

def cluster_matches(matches, kp1, kp2, distance_threshold=CLUSTERING_DISTANCE):
    """
    Agrupa matches que pertenecen al MISMO logo individual
    Clustering muy estricto para detectar cada logo por separado
    """
    print(f"Clustering {len(matches)} matches con umbral MUY estricto de {distance_threshold} píxeles")
    
    if len(matches) < MIN_CLUSTER_SIZE:
        print(f"Muy pocos matches para clustering (mínimo {MIN_CLUSTER_SIZE})")
        return []
    
    # Extraer coordenadas de los matches en la imagen target
    points = np.array([kp2[m.trainIdx].pt for m in matches])
    
    clusters = []
    visited = np.zeros(len(matches), dtype=bool)
    
    for i in range(len(matches)):
        if visited[i]:
            continue
        
        # Iniciar nuevo cluster para este logo
        cluster_matches = [matches[i]]
        cluster_indices = [i]
        visited[i] = True
        
        # Buscar SOLO matches muy cercanos (mismo logo)
        for j in range(len(matches)):
            if visited[j]:
                continue
                
            distance = np.linalg.norm(points[j] - points[i])
            if distance < distance_threshold:  # Solo matches del mismo logo
                cluster_matches.append(matches[j])
                cluster_indices.append(j)
                visited[j] = True
        
        # Crear cluster solo si tiene suficientes matches para un logo
        if len(cluster_matches) >= MIN_CLUSTER_SIZE:
            clusters.append(cluster_matches)
            print(f"  Logo detectado con {len(cluster_matches)} matches")
    
    print(f"Total de logos individuales detectados: {len(clusters)}")
    return clusters

def calculate_homography_and_bbox(cluster_matches, kp1, kp2, template_shape):
    """Calcula homografía y bounding box para un cluster de matches"""
    if len(cluster_matches) < 4:
        print(f"    Cluster muy pequeño: {len(cluster_matches)} matches (mínimo 4)")
        return None, None, 0
    
    # Extraer puntos correspondientes
    src_pts = np.float32([kp1[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
    
    print(f"    Calculando homografía con {len(cluster_matches)} puntos...")
    
    try:
        # Para pocos puntos, probar primero método más simple
        if len(cluster_matches) == 4:
            # Con exactamente 4 puntos, usar método directo
            M = cv2.getPerspectiveTransform(src_pts.reshape(4, 2), dst_pts.reshape(4, 2))
            mask = np.ones(len(cluster_matches), dtype=np.uint8)
        else:
            # Calcular homografía con RANSAC EXTREMADAMENTE permisivo para pocos matches
            M, mask = cv2.findHomography(src_pts, dst_pts, 
                                       cv2.RANSAC, 
                                       ransacReprojThreshold=25.0,  # EXTREMADAMENTE permisivo
                                       confidence=0.1,   # Muy bajo para aceptar cualquier cosa
                                       maxIters=10000)   # Muchísimas iteraciones
        
        if M is None:
            print("    ✗ Homografía es None")
            return None, None, 0
        
        # Verificar que la matriz no es singular
        det = np.linalg.det(M[:2, :2])
        if abs(det) < 1e-6:
            print(f"    ✗ Matriz singular (det={det})")
            return None, None, 0
        
        # Definir las esquinas del template
        h, w = template_shape[:2]
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transformar las esquinas usando la homografía
        transformed_corners = cv2.perspectiveTransform(template_corners, M)
        
        # Verificar que la transformación es válida
        area = cv2.contourArea(transformed_corners)
        template_area = template_shape[0] * template_shape[1]
        scale_factor = np.sqrt(area / template_area)
        
        print(f"    Área transformada: {area}, template: {template_area}, escala: {scale_factor:.3f}")
        
        if area < 20:  # Área mínima muy permisiva
            print(f"    ✗ Área muy pequeña: {area}")
            return None, None, 0
        
        # Calcular bounding box rectangular
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        # Verificar que el bbox tiene dimensiones razonables
        width, height = x2 - x1, y2 - y1
        if width < 10 or height < 10:
            print(f"    ✗ Bbox muy pequeño: {width}x{height}")
            return None, None, 0
        
        # Contar inliers para calcular confianza
        inliers = mask.sum() if mask is not None else len(cluster_matches)
        confidence = float(inliers) / len(cluster_matches)
        
        print(f"    ✓ Homografía válida: bbox=({x1},{y1},{x2},{y2}), inliers={inliers}/{len(cluster_matches)}, conf={confidence:.3f}")
        
        return (x1, y1, x2, y2), transformed_corners, confidence
        
    except Exception as e:
        print(f"    ✗ Error en homografía: {e}")
        return None, None, 0

# ===================================================================
# FUNCIONES PRINCIPALES
# ===================================================================

def detect_sift_keypoints(image, template):
    sift = cv2.SIFT_create(
        nfeatures=SIFT_NFEATURES,
        nOctaveLayers=SIFT_NOCTAVELAYERS,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
        sigma=SIFT_SIGMA
    )
    
    # Aplicar preprocesamiento
    template_processed = preprocess_template(template)
    image_processed = preprocess_image(image)
    
    kp1, des1 = sift.detectAndCompute(template_processed, None)
    kp2, des2 = sift.detectAndCompute(image_processed, None)
    
    # Asegurar que los descriptores son válidos para FLANN
    if des1 is not None:
        des1 = des1.astype(np.float32)
    if des2 is not None:
        des2 = des2.astype(np.float32)
    
    return kp1, des1, kp2, des2, sift

def match_features(des1, des2):
    """Matching más agresivo sin ratio test - mejor para múltiples instancias"""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print(f"Error: descriptores insuficientes - Template: {len(des1) if des1 is not None else 0}, Imagen: {len(des2) if des2 is not None else 0}")
        return []
    
    print(f"Iniciando matching directo - Template descriptores: {len(des1)}, Imagen descriptores: {len(des2)}")
    
    # Usar BruteForce matcher directo (más simple y efectivo para múltiples instancias)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    try:
        # Asegurar que los descriptores sean float32
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
        
        # Obtener todos los matches sin ratio test
        matches = bf.match(des1, des2)
        
        # Ordenar por distancia (mejores primero)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filtrar solo por distancia directa (más permisivo)
        good_matches = [m for m in matches if m.distance < DISTANCE_THRESHOLD]
        
        print(f"Matches totales encontrados: {len(matches)}")
        print(f"Matches después del filtro de distancia (<{DISTANCE_THRESHOLD}): {len(good_matches)}")
        
        # Mostrar estadísticas de distancias
        if matches:
            distances = [m.distance for m in matches]
            print(f"Distancias: min={min(distances):.1f}, max={max(distances):.1f}, media={np.mean(distances):.1f}")
            print(f"Mejores 5 distancias: {[f'{d:.1f}' for d in distances[:5]]}")
        
        return good_matches
        
    except Exception as e:
        print(f"Error en matching: {e}")
        return []

# ===================================================================
# FUNCIONES PRINCIPALES
# ===================================================================

def find_multiple_templates(image, template):
    """
    Detecta múltiples instancias del template usando SIFT con clustering de matches
    Basado en la investigación proporcionada sobre homografías y bounding boxes
    """
    print("=== DETECCIÓN MÚLTIPLE CON SIFT Y CLUSTERING ===")
    
    # Detectar keypoints y descriptores
    kp_template, des_template, kp_image, des_image, _ = detect_sift_keypoints(image, template)
    
    print(f"Keypoints detectados - Template: {len(kp_template)}, Imagen: {len(kp_image)}")
    
    if des_template is None or des_image is None:
        print("Error: No se pudieron generar descriptores")
        return [], [], kp_template, kp_image, 1.0
    
    # Realizar matching con ratio test de Lowe
    good_matches = match_features(des_template, des_image)
    print(f"Matches después del ratio test: {len(good_matches)}")
    
    if len(good_matches) < MIN_MATCHES:
        print(f"Insuficientes matches para clustering (mínimo {MIN_MATCHES})")
        return [], good_matches, kp_template, kp_image, 1.0
    
    # Clustering de matches para detectar múltiples instancias
    print("Aplicando clustering espacial a los matches...")
    clusters = cluster_matches(good_matches, kp_template, kp_image)
    print(f"Clusters encontrados: {len(clusters)}")
    
    detections = []
    template_shape = template.shape
    
    # Procesar cada cluster
    for i, cluster in enumerate(clusters):
        print(f"\nProcesando cluster {i+1} con {len(cluster)} matches:")
        
        # Calcular homografía y bounding box para este cluster
        bbox, corners, confidence = calculate_homography_and_bbox(
            cluster, kp_template, kp_image, template_shape
        )
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # Verificar que el bbox es válido
            if (x1 < x2 and y1 < y2 and 
                x1 >= 0 and y1 >= 0 and 
                x2 <= image.shape[1] and y2 <= image.shape[0]):
                
                area = (x2 - x1) * (y2 - y1)
                
                detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'matches': len(cluster),
                    'inliers': int(confidence * len(cluster)),
                    'area': area,
                    'corners': corners,
                    'cluster_id': i,
                    'method': 'sift_clustering'
                }
                
                detections.append(detection)
                print(f"  ✓ Detección válida: bbox{bbox}, confianza: {confidence:.3f}, área: {area}")
            else:
                print(f"  ✗ Bbox inválido: {bbox}")
        else:
            print(f"  ✗ No se pudo calcular homografía")
    
    print(f"\n=== RESUMEN: {len(detections)} detecciones válidas ===")
    
    # Retornar el mejor cluster para visualización
    best_cluster = max(clusters, key=len) if clusters else good_matches
    
    return detections, best_cluster, kp_template, kp_image, 1.0

def visualize_keypoints(image, template, kp1, kp2, filename):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Template original y keypoints
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_processed = preprocess_template(template_gray)
    template_with_kp = cv2.drawKeypoints(template_processed, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    axes[0,0].imshow(template_gray, cmap='gray')
    axes[0,0].set_title('Template Original (Grayscale)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(template_processed, cmap='gray')
    axes[0,1].set_title(f'Template Procesado + Keypoints: {len(kp1)}')
    axes[0,1].axis('off')
    for kp in kp1:
        circle = plt.Circle((kp.pt[0], kp.pt[1]), 2, color='red', fill=False)
        axes[0,1].add_patch(circle)
    
    # Imagen original y keypoints
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_processed = preprocess_image(image_gray)
    
    axes[1,0].imshow(image_gray, cmap='gray')
    axes[1,0].set_title('Imagen Original (Grayscale)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(image_processed, cmap='gray')
    axes[1,1].set_title(f'Imagen Procesada + Keypoints: {len(kp2)}')
    axes[1,1].axis('off')
    # Mostrar solo algunos keypoints para no saturar la visualización
    kp2_sample = kp2[::10] if len(kp2) > 100 else kp2
    for kp in kp2_sample:
        circle = plt.Circle((kp.pt[0], kp.pt[1]), 1, color='red', fill=False)
        axes[1,1].add_patch(circle)
    
    plt.tight_layout()
    if SAVE_VISUALIZATIONS:
        plt.savefig(os.path.join(PATH_SALIDA, filename), dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def visualize_matches(image, template, kp1, kp2, good_matches, filename):
    img_matches = cv2.drawMatches(template, kp1, image, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(20, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Matches: {len(good_matches)}')
    plt.axis('off')
    
    if SAVE_VISUALIZATIONS:
        plt.savefig(os.path.join(PATH_SALIDA, filename), dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def visualize_detections_before_nms(image, detections, filename):
    img_copy = image.copy()
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f'{confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f'Detecciones antes de NMS: {len(detections)}')
    plt.axis('off')
    
    if SAVE_VISUALIZATIONS:
        plt.savefig(os.path.join(PATH_SALIDA, filename), dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def visualize_final_detections(image, detections, filename):
    img_copy = image.copy()
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img_copy, f'Det {i+1}: {confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f'Detecciones finales después de NMS: {len(detections)}')
    plt.axis('off')
    
    if SAVE_VISUALIZATIONS:
        plt.savefig(os.path.join(PATH_SALIDA, filename), dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def analyze_intensity_differences(template, image, filename):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Primera fila: Template
    axes[0,0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Template Original (Color)')
    axes[0,0].axis('off')
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    axes[0,1].imshow(template_gray, cmap='gray')
    axes[0,1].set_title('Template Grayscale')
    axes[0,1].axis('off')
    
    template_processed = preprocess_template(template_gray)
    axes[0,2].imshow(template_processed, cmap='gray')
    axes[0,2].set_title('Template Invertido (Final)')
    axes[0,2].axis('off')
    
    # Segunda fila: Imagen
    axes[1,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('Imagen Original (Color)')
    axes[1,0].axis('off')
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    axes[1,1].imshow(image_gray, cmap='gray')
    axes[1,1].set_title('Imagen Grayscale')
    axes[1,1].axis('off')
    
    image_processed = preprocess_image(image_gray)
    axes[1,2].imshow(image_processed, cmap='gray')
    axes[1,2].set_title('Imagen Sin Procesar (Final)')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    if SAVE_VISUALIZATIONS:
        plt.savefig(os.path.join(PATH_SALIDA, filename), dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    print(f"Template shape: {template.shape}")
    print(f"Image shape: {image.shape}")
    print(f"Template gray mean: {template_gray.mean():.1f}, std: {template_gray.std():.1f}")
    print(f"Template invertido mean: {template_processed.mean():.1f}, std: {template_processed.std():.1f}")
    print(f"Image gray mean: {image_gray.mean():.1f}, std: {image_gray.std():.1f}")
    print(f"Imagen procesada mean: {image_processed.mean():.1f}, std: {image_processed.std():.1f}")

# ===================================================================
# FUNCIÓN PRINCIPAL
# ===================================================================

def main():
    create_output_dir()
    
    print("Cargando imágenes...")
    template = cv2.imread(PATH_TEMPLATE)
    image = cv2.imread(PATH_IMAGEN)
    
    if template is None or image is None:
        print("Error: No se pudieron cargar las imágenes")
        return
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("Analizando diferencias de intensidad...")
    analyze_intensity_differences(template, image, '00_analisis_intensidad.png')
    
    print("Detectando keypoints SIFT...")
    kp1, des1, kp2, des2, sift = detect_sift_keypoints(image_gray, template_gray)
    
    print(f"Keypoints template: {len(kp1)}")
    print(f"Keypoints imagen: {len(kp2)}")
    
    visualize_keypoints(image, template, kp1, kp2, '01_keypoints_detection.png')
    
    print("Realizando matching inicial...")
    good_matches = match_features(des1, des2)
    print(f"Matches encontrados: {len(good_matches)}")
    
    if len(good_matches) > 0:
        visualize_matches(image, template, kp1, kp2, good_matches, '02_feature_matching.png')
    
    print("Buscando múltiples instancias del template...")
    detections, best_cluster, kp_template, kp_image, _ = find_multiple_templates(image_gray, template_gray)
    print(f"Detecciones encontradas: {len(detections)}")
    
    # Siempre mostrar el mejor matching encontrado
    if len(best_cluster) > 0:
        print(f"Mostrando mejor clustering con {len(best_cluster)} matches")
        visualize_matches(image, template, kp_template, kp_image, best_cluster, '02_mejor_matching.png')
    else:
        print("No se encontraron matches SIFT válidos")
    
    if detections:
        visualize_detections_before_nms(image, detections, '03_detecciones_antes_nms.png')
        
        print("Aplicando Non-Maximum Suppression...")
        boxes = [d['bbox'] for d in detections]
        scores = [d['confidence'] for d in detections]
        
        keep_indices = non_max_suppression(boxes, scores, IOU_THRESHOLD)
        final_detections = [detections[i] for i in keep_indices]
        
        print(f"Detecciones finales después de NMS: {len(final_detections)}")
        
        visualize_final_detections(image, final_detections, '04_detecciones_finales.png')
        
        print("\nResultados finales:")
        for i, detection in enumerate(final_detections):
            x1, y1, x2, y2 = detection['bbox']
            print(f"Detección {i+1}:")
            print(f"  Bbox: ({x1}, {y1}, {x2}, {y2})")
            print(f"  Confianza: {detection['confidence']:.3f}")
            print(f"  Método: {detection.get('method', 'sift_clustering')}")
            if detection.get('matches', 0) > 0:
                print(f"  Matches: {detection['matches']}")
                print(f"  Inliers: {detection['inliers']}")
            if 'cluster_id' in detection:
                print(f"  Cluster ID: {detection['cluster_id']}")
    else:
        print("No se encontraron detecciones válidas")
    
    print(f"\nVisualizaciones guardadas en: {PATH_SALIDA}")

if __name__ == "__main__":
    main()