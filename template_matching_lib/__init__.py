from .preprocessing import cargar_template, preprocesar_imagen, redimensionar_template
from .template_matching import (
    procesar_escala_individual,
    buscar_coincidencias_multiescala,
    procesar_escala_individual_multi,
    buscar_coincidencias_multiescala_multi
)
from .nms import (
    aplicar_nms,
    aplicar_nms_por_escala,
    aplicar_nms_multi_deteccion,
    calcular_iou,
    agrupar_detecciones_por_clustering,
    normalizar_detecciones_globalmente
)
from .visualization import (
    visualizar_preprocesamiento,
    visualizar_mapas_coincidencias,
    visualizar_resultado_final,
    visualizar_comparacion_escalas,
    visualizar_todas_las_detecciones,
    visualizar_detecciones_finales_numeradas
)
from .utils import obtener_imagenes, procesar_imagen, procesar_imagen_multi