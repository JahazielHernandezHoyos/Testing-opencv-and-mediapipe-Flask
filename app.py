# Importamos librerias
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from flask_cors import CORS

# Creamos nuestra funcion de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

# Realizamos la Videocaptura
cap = cv2.VideoCapture(0)

# Creamos la app
app = Flask(__name__)

# Mostramos el video en RT
def gen_frame():
    # Empezamos
    while True:
        # Leemos la VideoCaptura
        ret, frame = cap.read()

        # Si tenemos un error
        if not ret:
            break

        else:

            # Correccion de color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Observamos los resultados
            resultados = MallaFacial.process(frameRGB)

            # Si tenemos rostros
            if resultados.multi_face_landmarks:
                # Iteramos
                for rostros in resultados.multi_face_landmarks:
                    # Dibujamos
                    mpDibujo.draw_landmarks(
                        frame,
                        rostros,
                        mpMallaFacial.FACEMESH_TESSELATION,
                        ConfDibu,
                        ConfDibu,
                    )

            # Codificamos nuestro video en Bytes
            suc, encode = cv2.imencode(".jpg", frame)
            frame = encode.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

CORS(app, resources={r"/*": {"origins": "*"}})
# Ruta de aplicacion 'principal'
@app.route("/")
def index():
    return render_template("index.html")


# Ruta del video
@app.route("/video")
def video():
    return Response(gen_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)

#configuarion para deploy o no deploy en render.com
if __name__ == "__main__":
    app.run(host="216.24.57.3", port=443, debug=False)


