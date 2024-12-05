from ultralytics import YOLO
import cv2
import os

# Paso 1: Configurar rutas
model_path = "model/yolov8n_trained.pt"  # Cambia esto a la ubicación de tu modelo
test_images_path = "test/images"  # Carpeta que contiene las imágenes de prueba
results_path = "results"  # Carpeta para guardar las imágenes procesadas

# Crear carpeta de resultados si no existe
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Paso 2: Cargar el modelo entrenado
model = YOLO(model_path)

# Paso 3: Procesar todas las imágenes de la carpeta de prueba
for img_file in os.listdir(test_images_path):
    # Verificar si el archivo es una imagen
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        # Ruta completa de la imagen
        image_path = os.path.join(test_images_path, img_file)
        
        # Cargar la imagen
        image = cv2.imread(image_path)
        
        # Realizar predicción
        results = model.predict(source=image_path, save=False)
        
        # Dibujar los bounding boxes en la imagen
        annotated_frame = results[0].plot()
        
        # Guardar la imagen procesada en la carpeta de resultados
        output_path = os.path.join(results_path, img_file)
        cv2.imwrite(output_path, annotated_frame)

        print(f"Procesada y guardada: {output_path}")

print("\n¡Procesamiento completado! Todas las imágenes con bounding boxes están en la carpeta 'results'.")
