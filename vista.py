from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox, QLabel, QVBoxLayout, QLineEdit, QMessageBox, QTableWidget, QMainWindow, QSlider, QHBoxLayout, QApplication, QFileDialog, QHBoxLayout, QSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from modelo import ImageProcessor, MorfologiaImagen, BinarizacionImagen, crear_db, crear_tabla_imagenes, insertar_imagen, obtener_imagenes_procesadas
import os
import numpy as np
import pydicom
import sys
import cv2
import pandas as pd


class VisorMatUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizador de Señales")
        self.resize(800, 600)
        self.combo_llaves = QComboBox()
        self.btn_graficar = QPushButton("Graficar Señal Completa")
        self.campo_intervalo = QLineEdit()
        self.campo_intervalo.setPlaceholderText("Ej: 0-100")
        self.btn_graficar_intervalo = QPushButton("Graficar Intervalo")
        self.btn_promedio = QPushButton("Calcular Promedio")
        self.spin_canal = QSpinBox()
        self.spin_canal.setRange(0, 100)

        self.spin_ensayo = QSpinBox()
        self.spin_ensayo.setRange(0, 100)

        self.figura = Figure()
        self.canvas = FigureCanvas(self.figura)
        self.btn_cargar_archivo = QPushButton("Cargar Archivo .MAT")
        layout = QVBoxLayout()
        layout.addWidget(self.btn_cargar_archivo)
        layout.addWidget(QLabel("Seleccione una llave:"))
        layout.addWidget(self.combo_llaves)
        layout.addWidget(QLabel("Canal:"))
        layout.addWidget(self.spin_canal)
        layout.addWidget(QLabel("Ensayo:"))
        layout.addWidget(self.spin_ensayo)
        layout.addWidget(self.btn_graficar)
        layout.addWidget(self.campo_intervalo)
        layout.addWidget(self.btn_graficar_intervalo)
        layout.addWidget(self.btn_promedio)
        layout.addWidget(self.canvas)
        self.setLayout(layout)     


    def mostrar_error(self, mensaje):
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText(mensaje)
        msg.exec_()

class VisorCSVUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizador de CSV")
        self.resize(900, 600)

        self.tabla = QTableWidget()
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.btn_graficar = QPushButton("Graficar Scatter")
        self.figura = Figure()
        self.canvas = FigureCanvas(self.figura)
        self.btn_cargar_csv = QPushButton("Cargar Archivo CSV")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Datos CSV:"))
        layout.addWidget(self.btn_cargar_csv)
        layout.addWidget(self.tabla)
        layout.addWidget(QLabel("Eje X:"))
        layout.addWidget(self.combo_x)
        layout.addWidget(QLabel("Eje Y:"))
        layout.addWidget(self.combo_y)
        layout.addWidget(self.btn_graficar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def mostrar_error(self, mensaje):
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText(mensaje)
        msg.exec_()


    def mostrar_error(self, mensaje):
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText(mensaje)
        msg.exec_()


class DicomViewer(QMainWindow):
    def __init__(self, carpeta_dicom):
        super().__init__()
        self.setWindowTitle("Visualización DICOM")
        self.setGeometry(100, 100, 1200, 600)

        self.volumen = self.cargar_volumen(carpeta_dicom)

        self.num_axial = self.volumen.shape[0]
        self.num_coronal = self.volumen.shape[1]
        self.num_sagittal = self.volumen.shape[2]

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        self.figures = []
        self.canvases = []
        self.sliders = []
        self.labels = []

        planos = ["Axial", "Coronal", "Sagittal"]

        for i, plano in enumerate(planos):
            vbox = QVBoxLayout()
            fig = Figure(figsize=(4,4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.axis('off')

            slider = QSlider(Qt.Horizontal)
            max_index = [
                self.num_axial - 1,
                self.num_coronal - 1,
                self.num_sagittal - 1
            ][i]
            slider.setMaximum(max_index)
            slider.setValue(0)

            label = QLabel(f"{plano} Slice: 0")
            slider.valueChanged.connect(lambda value, p=plano: self.actualizar_imagen(p, value))

            vbox.addWidget(canvas)
            vbox.addWidget(label)
            vbox.addWidget(slider)

            layout.addLayout(vbox)

            self.figures.append(fig)
            self.canvases.append(canvas)
            self.sliders.append(slider)
            self.labels.append(label)

        self.actualizar_todas()

    def cargar_volumen(self, carpeta):
        archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta)]
        slices = [pydicom.dcmread(f) for f in archivos]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        volumen = np.stack([s.pixel_array for s in slices])
        return volumen

    def actualizar_imagen(self, plano, index):
        if plano == "Axial":
            img = self.volumen[index,:,:]
            i = 0
        elif plano == "Coronal":
            img = self.volumen[:,index,:]
            i = 1
        else:
            img = self.volumen[:,:,index]
            i = 2

        self.figures[i].clear()
        ax = self.figures[i].add_subplot(111)
        ax.imshow(img, cmap="gray")
        ax.axis('off')
        self.canvases[i].draw()
        self.labels[i].setText(f"{plano} Slice: {index}")

    def actualizar_todas(self):
        self.actualizar_imagen("Axial", 0)
        self.actualizar_imagen("Coronal", 0)
        self.actualizar_imagen("Sagittal", 0)


class InterfazGrafica(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imágenes Biomédicas")
        self.setGeometry(100, 100, 1000, 600)

        self.image_label = QLabel("Cargue una imagen")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.imagen_actual = None
        self.procesador = None

        self.boton_cargar = QPushButton("Cargar Imagen")
        self.boton_histograma = QPushButton("Histograma RGB")
        self.boton_ecualizar = QPushButton("Ecualizar Histograma")
        self.boton_conteo = QPushButton("Conteo de Células")
        self.boton_binarizar = QPushButton("Binarizar Imagen")
        self.boton_morfologia = QPushButton("Operación Morfológica")
        self.boton_esqueleto = QPushButton("Esqueletizar Imagen")
        self.boton_bordes = QPushButton("Detección de Bordes (Canny)")
        self.boton_historial = QPushButton("Ver Historial")

        # Combo boxes
        self.combo_binarizacion = QComboBox()
        self.combo_binarizacion.addItems(["binario", "binario_inv", "truncado", "tozero", "tozero_inv"])

        self.combo_morfologia = QComboBox()
        self.combo_morfologia.addItems(["erosion", "dilatacion", "apertura", "cierre", "gradiente", "tophat", "blackhat"])

        self.spin_kernel = QSpinBox()
        self.spin_kernel.setValue(5)
        self.spin_kernel.setMinimum(1)
        self.spin_kernel.setMaximum(21)

        self.spin_umbral = QSpinBox()
        self.spin_umbral.setValue(125)
        self.spin_umbral.setRange(0, 255)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        botones1 = QHBoxLayout()
        botones1.addWidget(self.boton_cargar)
        botones1.addWidget(self.boton_histograma)
        botones1.addWidget(self.boton_ecualizar)
        layout.addLayout(botones1)

        botones2 = QHBoxLayout()
        botones2.addWidget(self.boton_binarizar)
        botones2.addWidget(self.combo_binarizacion)
        botones2.addWidget(QLabel("Umbral:"))
        botones2.addWidget(self.spin_umbral)
        layout.addLayout(botones2)

        botones3 = QHBoxLayout()
        botones3.addWidget(self.boton_morfologia)
        botones3.addWidget(self.combo_morfologia)
        botones3.addWidget(QLabel("Tamaño:"))
        botones3.addWidget(self.spin_kernel)
        layout.addLayout(botones3)

        botones4 = QHBoxLayout()
        botones4.addWidget(self.boton_esqueleto)
        botones4.addWidget(self.boton_bordes)
        botones4.addWidget(self.boton_conteo)
        botones4.addWidget(self.boton_historial)
        layout.addLayout(botones4)

        self.setLayout(layout)

        self.boton_cargar.clicked.connect(self.cargar_imagen)
        self.boton_histograma.clicked.connect(self.ver_histograma)
        self.boton_ecualizar.clicked.connect(self.ecualizar)
        self.boton_binarizar.clicked.connect(self.binarizar)
        self.boton_morfologia.clicked.connect(self.aplicar_morfologia)
        self.boton_esqueleto.clicked.connect(self.esqueletizar)
        self.boton_bordes.clicked.connect(self.bordes)
        self.boton_conteo.clicked.connect(self.conteo)
        self.boton_historial.clicked.connect(self.historial)

    def mostrar_imagen(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        altura, ancho, canal = img_rgb.shape
        bytes_img = canal * ancho
        qimg = QImage(img_rgb.data, ancho, altura, bytes_img, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def cargar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Imagenes (*.png *.jpg *.jpeg)")
        if ruta:
            self.procesador = ImageProcessor(ruta)
            self.imagen_actual = self.procesador.get_image()
            self.mostrar_imagen(self.imagen_actual)

    def ver_histograma(self):
        if self.procesador:
            self.procesador.calcular_histograma_manual()
            self.procesador.graficar_histograma()

    def ecualizar(self):
        if self.procesador:
            ecualizada = self.procesador.ecualizar_histograma()
            self.mostrar_imagen(cv2.cvtColor(ecualizada, cv2.COLOR_GRAY2RGB))

    def binarizar(self):
        if self.procesador:
            gray = cv2.cvtColor(self.imagen_actual, cv2.COLOR_RGB2GRAY)
            metodo = self.combo_binarizacion.currentText()
            umbral = self.spin_umbral.value()
            binarizador = BinarizacionImagen(gray)
            resultado = getattr(binarizador, metodo)(umbral=umbral)
            self.mostrar_imagen(cv2.cvtColor(resultado, cv2.COLOR_GRAY2RGB))

    def aplicar_morfologia(self):
        if self.procesador:
            metodo = self.combo_morfologia.currentText()
            tam = self.spin_kernel.value()
            gray = cv2.cvtColor(self.imagen_actual, cv2.COLOR_RGB2GRAY)
            _, binaria = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
            morfo = MorfologiaImagen(binaria)
            resultado = getattr(morfo, metodo)(kernel_size=(tam, tam))
            self.mostrar_imagen(cv2.cvtColor(resultado, cv2.COLOR_GRAY2RGB))

    def esqueletizar(self):
        if self.procesador:
            gray = cv2.cvtColor(self.imagen_actual, cv2.COLOR_RGB2GRAY)
            _, binaria = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
            morfo = MorfologiaImagen(binaria)
            skel = morfo.esqueletizar(metodo='skimage')
            self.mostrar_imagen(cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB))

    def bordes(self):
        if self.procesador:
            edges = self.procesador.deteccion_bordes_canny()
            self.mostrar_imagen(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def conteo(self):
        if self.procesador:
            cerrada, _, total = self.procesador.conteo_celulas()
            QMessageBox.information(self, "Conteo", f"Total de células detectadas: {total}")
            self.mostrar_imagen(cv2.cvtColor(cerrada, cv2.COLOR_GRAY2RGB))
            insertar_imagen(self.procesador.get_metadata())

    def historial(self):
        registros = obtener_imagenes_procesadas()
        mensaje = "\n".join([f"{r['fecha']} - {r['nombre_archivo']} - {r['tecnica_usada']}" for r in registros])
        QMessageBox.information(self, "Historial de Imágenes", mensaje)

class LoginUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login - Proyecto Informática II")
        self.setGeometry(100, 100, 300, 150)

        self.label_user = QLabel("Usuario:")
        self.input_user = QLineEdit()

        self.label_pass = QLabel("Contraseña:")
        self.input_pass = QLineEdit()
        self.input_pass.setEchoMode(QLineEdit.Password)

        self.btn_login = QPushButton("Ingresar")

        layout = QVBoxLayout()
        layout.addWidget(self.label_user)
        layout.addWidget(self.input_user)
        layout.addWidget(self.label_pass)
        layout.addWidget(self.input_pass)
        layout.addWidget(self.btn_login)

        self.setLayout(layout)

class MenuImagenUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú - Experto en Imágenes")
        self.setGeometry(100, 100, 400, 300)

        self.btn_jpg_png = QPushButton("Procesar JPG/PNG")
        self.btn_dicom = QPushButton("Visualizar DICOM")
        self.btn_convertir_nifti = QPushButton("Convertir DICOM a NIFTI")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Seleccione una acción:"))
        layout.addWidget(self.btn_jpg_png)
        layout.addWidget(self.btn_dicom)
        layout.addWidget(self.btn_convertir_nifti)
        self.setLayout(layout)

class MenuSenalUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menu - Experto en Senales")
        self.setGeometry(100, 100, 300, 200)

        self.btn_mat = QPushButton("Procesar Senales .MAT")
        self.btn_csv = QPushButton("Procesar Datos .CSV")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Seleccione el tipo de datos a procesar:"))
        layout.addWidget(self.btn_mat)
        layout.addWidget(self.btn_csv)
        self.setLayout(layout)



if __name__ == '__main__':
    crear_db()
    crear_tabla_imagenes()
    app = QApplication(sys.argv)
    ventana = InterfazGrafica()
    ventana.show()
    sys.exit(app.exec_())



