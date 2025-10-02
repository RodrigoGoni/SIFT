from tqdm import tqdm
from template_matching_lib.config import ConfigSingle, crear_config_dict
from template_matching_lib.preprocessing import cargar_template
from template_matching_lib.utils import obtener_imagenes_objetivo, procesar_imagen, crear_directorio_resultados


def main():
    config = crear_config_dict(ConfigSingle)
    crear_directorio_resultados(config)
    
    template_data = cargar_template(f"{config['PATH_TEMPLATE']}pattern.png", config)
    imagenes = obtener_imagenes_objetivo(config=config)
    
    if not imagenes:
        print("No se encontraron imágenes para procesar.")
        return

    resultados = {}
    for ruta_imagen in tqdm(imagenes, desc="Procesando imágenes"):
        nombre = ruta_imagen.split('/')[-1]
        detecciones = procesar_imagen(ruta_imagen, template_data, config)
        resultados[nombre] = detecciones

    total = sum(len(det) for det in resultados.values())
    print(f"Proceso completado: {total} detecciones en {len(resultados)} imágenes")


if __name__ == "__main__":
    main()