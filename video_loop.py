"""
 Demo de lectura de video con opencv y procesamiento por frames


 En este script se detalla cómo utilizar el objeto VideoCapture de OpenCV para la extracción de los frames de un video.
 Se emplea el método .isOpened() para verificar que el video haya sido leido sin problemas, ya que en muchas ocasiones
 pueden haber decodificadores de video faltantes, por lo que no todos los videos pueden abrirse. (se recomienda revisar
 e instalar librerías como klite codec pack o ffmpeg en caso de problemas con formatos).

 Adicionalmente, se utilizan los valores ret y frame que retorna el método .read(), el cual permite determinar en la
 variable ret (True or False) si hay o no un frame retornado. En caso de que exista, el frame se procesa.

 El procesamiento se hace como un ciclo while que termina cuando el usuario pulse q (quit, salir) o cuando
 no existan más frames por leer en el video.

"""
# librerias necesarias
import numpy as np
from cv2 import cv2
from utilidades.inferencias import *

# crear capturador de video
ruta_video = "videotest.mp4" # ruta en el disco duro en donde se encuentra el video
capture = cv2.VideoCapture(ruta_video)
font = cv2.FONT_HERSHEY_SIMPLEX  # variable con la fuente para renderizar sobre el video
ruta_modelo = "models/saved_model"
# cargar el modelo guardado
modelo, obj_inferencias = cargar_modelo(ruta_modelo)

while capture.isOpened():
    ret, frame = capture.read() # si se pudo abrir el video, leer frames
    if ret: # si en el frame hay una imágen:
        # reescalar frame
        frame_cuadrado = reescalar_imagen(frame)
        # convertir en tensor
        tensor_frame = imagen_a_tensor(frame_cuadrado)
        # mandarlo al modelo
        resultado = obj_inferencias(tensor_frame)
        # postprocesar
        scores, boxes, classes = postprocesado_salida_modelo(resultado)
        # dibujar los bboxes encontrados
        frame_salida = dibujar_bboxes(frame_cuadrado, scores, boxes, classes)
        # mostrar imágen actual con opencv
        cv2.imshow("Mi video", frame_salida)

        if cv2.waitKey(1) & 0xff == ord('q'): # romper el ciclo si se presiona la letra Q
            break
    else: # si no hay más frames, terminar
        #salir del ciclo
        break
#liberamos la memoria y el video
capture.release()
cv2.destroyAllWindows()