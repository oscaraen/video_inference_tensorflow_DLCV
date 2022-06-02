"""
Script de funciones y utilidades para las inferencias
"""
# bibliotecas necesarias
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def cargar_modelo(ruta_modelo):
    """ Funcion para leer modelo....."""
    modelo = tf.saved_model.load(ruta_modelo) # carga modelo
    obj_inferencias = modelo.signatures["serving_default"]
    return modelo, obj_inferencias

def imagen_a_tensor(imagen):
    """
    recibe np array como imagen y retorna tensor
    :param imagen:
    :return: ttf tensor
    """
    imagen = np.expand_dims(imagen, 0)
    imagen = tf.convert_to_tensor(imagen, dtype=tf.uint8)
    return imagen

def reescalar_imagen(imagen, tam_salida=512):
    imagen_reescalada = cv2.resize(imagen, (tam_salida, tam_salida))
    return imagen_reescalada

def postprocesado_salida_modelo(inferencias, threshold=0.3):
    """
    toma las inferencias del modelo y entrega una lista
    con los boundig boxes de mayor probabilidad
    :param threshold:
    :param inferencias:
    :return:
    """
    scores = inferencias['detection_scores'].numpy()
    boxes = inferencias['detection_boxes'].numpy()
    classes = inferencias['detection_classes'].numpy()

    # obtener las predicciones sobre el umbral

    indices = scores > threshold
    scores = scores[indices]
    classes = classes[indices]
    boxes = boxes[indices]

    return scores, boxes, classes

def dibujar_bboxes(imagen ,scores, boxes, classes):
    # iterar para ir dibujando
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    verde = (0,255,0)
    for box, score, clase in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box[0] *512, box[1] *512, box[2] *512, box[3] *512
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
        cv2.rectangle(imagen, (xmin, ymin),(xmax, ymax), verde, 1)
        cv2.putText(imagen, f"{score}", (xmin, ymin), fuente, 1, (0,255,0), 3, cv2.LINE_AA)
    return imagen


if __name__ == "__main__":
    ruta_modelo = "models/saved_model"
    ruta_imagen = "ciclistas.jpg"
    modelo, obj_inferencias = cargar_modelo(ruta_modelo)
    #leer una imagen (para pruebas)
    imagen = cv2.imread(ruta_imagen)
    # reescalamos la imagen
    img_reescalada = reescalar_imagen(imagen)
    # descomentar paravisualizar con color correcto
    # mantener comentada para enviar al modelo (BGR)
    #img_reescalada = cv2.cvtColor(img_reescalada, cv2.COLOR_BGR2RGB)
    plt.imshow(img_reescalada)
    plt.show()

    # ahora volveremos la img un tensor para procesar
    img_tensor = imagen_a_tensor(img_reescalada)
    # ahora si, obtener las inferencias
    resultado = obj_inferencias(img_tensor)
    # postprocesar salida y dibujar
    scores, boxes, classes = postprocesado_salida_modelo(resultado)
    # dibujar
    imagen_boxes = dibujar_bboxes(img_reescalada,scores, boxes, classes)
    #mostrar
    cv2.imshow("imagen con bboxes", imagen_boxes)
    cv2.waitKey(0)









