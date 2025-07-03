
from modelo import ModeloSenales, ModeloCSV, MySQLDatabase, DicomProcessor, ImageProcessor, BinarizacionImagen, MorfologiaImagen, insertar_imagen, obtener_imagenes_procesadas, insertar_usuarios_por_defecto, validar_credenciales
from vista import VisorMatUI, VisorCSVUI, DicomViewer, LoginUI, InterfazGrafica, VisorMatUI, LoginUI
from PyQt5.QtWidgets import QTableWidgetItem, QApplication, QMessageBox, QFileDialog
import numpy as np
import sys
import scipy.io


class VisorMatController:
    def __init__(self, vista):
        self.vista = vista
        self.mat_data = None

        # Conectar botones a sus métodos
        self.vista.btn_graficar.clicked.connect(self.graficar_senal_completa)
        self.vista.btn_graficar_intervalo.clicked.connect(self.graficar_intervalo)
        self.vista.btn_promedio.clicked.connect(self.calcular_promedio)

        # Abrir el archivo MAT al inicio
        self.cargar_archivo()

    def cargar_archivo(self):
        ruta_archivo, _ = QFileDialog.getOpenFileName(
            self.vista, "Seleccionar archivo .mat", "", "Archivos MAT (*.mat)"
        )
        if not ruta_archivo:
            self.vista.mostrar_error("No se seleccionó ningún archivo.")
            return

        try:
            mat = scipy.io.loadmat(ruta_archivo)
            llaves = [k for k in mat.keys() if not k.startswith("__")]

            if not llaves:
                self.vista.mostrar_error("El archivo no contiene variables válidas.")
                return

            self.mat_data = mat
            self.vista.combo_llaves.clear()
            self.vista.combo_llaves.addItems(llaves)

        except Exception as e:
            self.vista.mostrar_error(f"Error al cargar el archivo: {str(e)}")

    def graficar_senal_completa(self):
        if self.mat_data is None:
            self.vista.mostrar_error("No hay datos cargados.")
            return

        llave = self.vista.combo_llaves.currentText()
        datos = np.squeeze(self.mat_data[llave])

        self.vista.figura.clear()
        ax = self.vista.figura.add_subplot(111)
        ax.plot(datos)
        ax.set_title(f"Señal completa: {llave}")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valor")
        self.vista.canvas.draw()

    def graficar_intervalo(self):
        if self.mat_data is None:
            self.vista.mostrar_error("No hay datos cargados.")
            return

        texto = self.vista.campo_intervalo.text()
        if "-" not in texto:
            self.vista.mostrar_error("Intervalo inválido. Use el formato inicio-fin (ej: 0-100).")
            return

        try:
            inicio, fin = map(int, texto.split("-"))
        except ValueError:
            self.vista.mostrar_error("Intervalo inválido. Debe usar dos números separados por '-'.")
            return

        llave = self.vista.combo_llaves.currentText()
        datos = np.squeeze(self.mat_data[llave])

        if inicio < 0 or fin > len(datos) or inicio >= fin:
            self.vista.mostrar_error("Rango fuera de los límites de la señal.")
            return

        self.vista.figura.clear()
        ax = self.vista.figura.add_subplot(111)
        ax.plot(range(inicio, fin), datos[inicio:fin])
        ax.set_title(f"Señal {llave} ({inicio}-{fin})")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valor")
        self.vista.canvas.draw()

    def calcular_promedio(self):
        if self.mat_data is None:
            self.vista.mostrar_error("No hay datos cargados.")
            return

        llave = self.vista.combo_llaves.currentText()
        datos = np.squeeze(self.mat_data[llave])
        promedio = np.mean(datos)

        self.vista.mostrar_error(f"El promedio de la señal '{llave}' es: {promedio:.3f}")


class ControladorCSV:
    def __init__(self):
        self.vista = VisorCSVUI()
        self.modelo = ModeloCSV()

        self.vista.btn_graficar.clicked.connect(self.graficar_dispersión)

    def cargar_archivo(self, ruta):
        self.modelo.cargar_csv(ruta)
        columnas = self.modelo.get_columnas()
        self.vista.combo_x.clear()
        self.vista.combo_y.clear()
        self.vista.combo_x.addItems(columnas)
        self.vista.combo_y.addItems(columnas)
        self.llenar_tabla()
        nombre_archivo = ruta.split("/")[-1]
        self.modelo.guardar_registro(nombre_archivo, ruta)

    def llenar_tabla(self):
        df = self.modelo.df
        self.vista.tabla.setRowCount(len(df))
        self.vista.tabla.setColumnCount(len(df.columns))
        self.vista.tabla.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.vista.tabla.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def graficar_dispersión(self):
        x_col = self.vista.combo_x.currentText()
        y_col = self.vista.combo_y.currentText()
        x = self.modelo.get_columna(x_col)
        y = self.modelo.get_columna(y_col)

        self.vista.figura.clear()
        ax = self.vista.figura.add_subplot(111)
        ax.scatter(x, y)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Gráfico de dispersión: {x_col} vs {y_col}")
        self.vista.canvas.draw()


class DicomController:
    def __init__(self, db, processor):
        self.db = db
        self.processor = processor

    def procesar_metadatos(self, dicoms):
        for ds in dicoms:
            archivo = ds.filename.split("/")[-1] 
            datos = self.processor.extraer_metadatos(ds, archivo)
            self.db.insertar_metadatos(datos)
        print("Metadatos insertados.")

    def convertir_y_registrar_nifti(self, carpeta_salida):
        nifti_path = self.processor.convertir_a_nifti(carpeta_salida)
        dicoms = self.processor.leer_dicoms()
        ds = dicoms[0]
        datos = (
            str(ds.get("PatientID", "")),
            nifti_path,
            len(dicoms),
            str(ds.get("PatientBirthDate", "")),
            str(ds.get("PatientSex", "")),
            str(ds.get("PatientAge", ""))
        )
        self.db.insertar_nifti(datos)
        print("Nifti registrado.")

class ControladorImagen:
    def __init__(self, ruta_imagen):
        self.procesador = ImageProcessor(ruta_imagen)
        self.imagen_original = self.procesador.get_image()

    def obtener_imagen_original(self):
        return self.imagen_original

    def obtener_histograma(self):
        return self.procesador.calcular_histograma_manual()

    def obtener_imagen_ecualizada(self):
        return self.procesador.ecualizar_histograma()

    def convertir_espacio_color(self, espacio):
        return self.procesador.convertir_espacio_color(espacio)

    def detectar_bordes(self, min_val=100, max_val=200):
        return self.procesador.deteccion_bordes_canny(min_val, max_val)

    def conteo_celulas(self, umbral=130, kernel_size=(3,3), iteraciones=7):
        return self.procesador.conteo_celulas(umbral, kernel_size, iteraciones)

    def aplicar_binarizacion(self, metodo='binario', umbral=125):
        """
        Aplica el método de binarización indicado sobre la imagen en escala de grises.
        """
        img_gray = self.procesador.ecualizar_histograma()
        binarizador = BinarizacionImagen(img_gray)

        metodo_func = getattr(binarizador, metodo, None)
        if metodo_func is None:
            raise ValueError(f"Método de binarización no válido: {metodo}")
        return metodo_func(umbral=umbral)

    def aplicar_morfologia(self, operacion='apertura', kernel=(5,5), iteraciones=1, umbral=125, metodo_bin='binario'):
        """
        Binariza la imagen y luego aplica una operación morfológica.
        """
        img_bin = self.aplicar_binarizacion(metodo=metodo_bin, umbral=umbral)
        morf = MorfologiaImagen(img_bin)

        operacion_func = getattr(morf, operacion, None)
        if operacion_func is None:
            raise ValueError(f"Operación morfológica no válida: {operacion}")
        return operacion_func(kernel_size=kernel, iteraciones=iteraciones)

    def esqueletizar(self, umbral=125, metodo_bin='binario', metodo_esqueleto='skimage'):
        """
        Binariza la imagen y la convierte en su esqueleto.
        """
        img_bin = self.aplicar_binarizacion(metodo=metodo_bin, umbral=umbral)
        morf = MorfologiaImagen(img_bin)
        return morf.esqueletizar(metodo=metodo_esqueleto)

    def procesar_y_guardar(self):
        """
        Procesa la imagen y guarda sus metadatos en la base de datos.
        """
        self.procesador.calcular_histograma_manual()
        self.procesador.ecualizar_histograma()
        _, _, total = self.procesador.conteo_celulas()

        metadata = self.procesador.get_metadata()
        insertar_imagen(metadata)
        return metadata, total

    def obtener_historial(self):
        return obtener_imagenes_procesadas()


class InsertarUsuariosController:
    def __init__(self):
        insertar_usuarios_por_defecto()
        print("Usuarios creados correctamente.")

class LoginController:
    def __init__(self):
        self.login_view = LoginUI()
        self.login_view.btn_login.clicked.connect(self.handle_login)
        self.login_view.show()

    def handle_login(self):
        username = self.login_view.input_user.text()
        password = self.login_view.input_pass.text()

        rol = validar_credenciales(username, password)

        if rol:
            QMessageBox.information(
                self.login_view, "Bienvenido", f"Login correcto. Rol: {rol}"
            )
            self.login_view.close()
            self.abrir_interfaz_por_rol(rol)
        else:
            QMessageBox.warning(
                self.login_view, "Error", "Usuario o contraseña incorrectos."
            )

    def abrir_interfaz_por_rol(self, rol):
        if rol == "imagen":
            self.interfaz = InterfazGrafica()
            self.interfaz.show()
        elif rol == "senal":
            self.vista_mat = VisorMatUI()
            self.controlador_mat = VisorMatController(self.vista_mat)
            self.vista_mat.show()
        else:
            QMessageBox.warning(None, "Error", "Rol no reconocido.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_controller = LoginController()
    sys.exit(app.exec_())
