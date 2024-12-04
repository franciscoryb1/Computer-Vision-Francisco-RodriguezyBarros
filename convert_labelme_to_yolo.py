import os
import json

def convert_labelme_to_yolo(json_dir, output_dir, img_width, img_height):
    """
    Convierte archivos JSON de LabelMe a formato YOLO.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            output_txt = os.path.join(output_dir, json_file.replace('.json', '.txt'))
            with open(output_txt, 'w') as out_file:
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']
                    
                    # Calcular las coordenadas de la bounding box
                    x_min = min([p[0] for p in points])
                    y_min = min([p[1] for p in points])
                    x_max = max([p[0] for p in points])
                    y_max = max([p[1] for p in points])
                    
                    # Normalizar coordenadas
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # Obtener el ID de la clase
                    class_id = get_class_id(label)
                    out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def get_class_id(label):
    """
    Devuelve el ID de la clase basado en el nombre del objeto.
    """
    class_names = ['10_pesos', '20_pesos', '50_pesos', '100_pesos', '200_pesos', 
                   '500_pesos', '1000_pesos', '10000_pesos', '1_dolares', 
                   '5_dolares', '10_dolares', '20_dolares', '50_dolares', '100_dolares']
    return class_names.index(label)

# Directorios y dimensiones de la imagen
json_dir = 'dataset/labels/train'
output_dir = 'dataset/images/train'
img_width = 1200
img_height = 1600 

# Ejecutar la conversión
convert_labelme_to_yolo(json_dir, output_dir, img_width, img_height)
