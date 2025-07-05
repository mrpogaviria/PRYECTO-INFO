from scipy.io import loadmat
from datetime import datetime
from skimage.morphology import skeletonize
import numpy as np
import pandas as pd
import os
import pydicom
import dicom2nifti
import mysql.connector
import cv2
import matplotlib.pyplot as plt


class ModeloSenales:
    def __init__(self):
        self.datos = None

    def cargar_mat(self, ruta):
        self.datos = loadmat(ruta)
        return list(self.datos.keys())

    def obtener_array(self, llave):
        try:
            array = self.datos[llave]
            if isinstance(array, np.ndarray):
                return array
            else:
                return None
        except KeyError:
            return None

    def guardar_registro(self, nombre_archivo, ruta):
        try:
            conexion = mysql.connector.connect(
                host='127.0.0.1',
                user='informatica2',
                password='info20251',
                database='info2_PF'
            )
            cursor = conexion.cursor()
            query = """
            INSERT INTO otros_archivos (tipo_archivo, nombre_archivo, ruta_archivo)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, ('csv', nombre_archivo, ruta))
            conexion.commit()
            cursor.close()
            conexion.close()
        except mysql.connector.Error as e:
            print("Error al guardar CSV en la base de datos:", e)

class ModeloCSV:
    def __init__(self):
        self.df = None

    def cargar_csv(self, ruta):
        self.df = pd.read_csv(ruta)
        return list(self.df.columns)

    def obtener_datos(self):
        return self.df

    def guardar_registro(self, nombre_archivo, ruta):
        try:
            conexion = mysql.connector.connect(
                host='127.0.0.1',
                user='informatica2',
                password='info20251',
                database='info2_PF'
            )
            cursor = conexion.cursor()
            query = """
            INSERT INTO otros_archivos (tipo_archivo, nombre_archivo, ruta_archivo)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, ('mat', nombre_archivo, ruta))
            conexion.commit()
            cursor.close()
            conexion.close()
        except mysql.connector.Error as e:
            print("Error al guardar archivo MAT en la base de datos:", e)
    def get_columnas(self):
        return list(self.df.columns)

    def get_columna(self, nombre):
        return self.df[nombre]



class MySQLDatabase:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()
        print("Conexión MySQL establecida.")

    def crear_tabla_metadatos(self):
        sql = '''
        CREATE TABLE IF NOT EXISTS metadatos (
            id INT AUTO_INCREMENT PRIMARY KEY,
            archivo VARCHAR(255),
            patient_id VARCHAR(100),
            patient_name VARCHAR(255),
            birth_date VARCHAR(20),
            sex VARCHAR(10),
            age VARCHAR(10)
        )
        '''
        self.cursor.execute(sql)
        self.conn.commit()
        print("Tabla metadatos creada.")

    def insertar_metadatos(self, datos):
        sql = '''
        INSERT INTO metadatos (
            archivo, patient_id, patient_name, birth_date, sex, age
        ) VALUES (%s, %s, %s, %s, %s, %s)
        '''
        self.cursor.execute(sql, datos)
        self.conn.commit()

    def crear_tabla_nifti(self):
        sql = '''
        CREATE TABLE IF NOT EXISTS nifti_conversion (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id VARCHAR(100),
            nifti_path VARCHAR(255),
            dicom_folder VARCHAR(255),
            num_dicoms INT,
            birth_date VARCHAR(20),
            sex VARCHAR(10),
            age VARCHAR(10)
        )
        '''
        self.cursor.execute(sql)
        self.conn.commit()
        print("Tabla nifti creada.")

        try:
            self.cursor.execute("ALTER TABLE nifti_conversion CHANGE dicom_path dicom_folder VARCHAR(255)")
            self.conn.commit()
            print("Columna renombrada a dicom_folder.")
        except Exception as e:
            print(f"No se pudo cambiar el nombre de la columna (posiblemente ya existe): {e}")

        print("Tabla nifti creada.")

    def insertar_nifti(self, datos):
        sql = '''
        INSERT INTO nifti_conversion (
            patient_id, nifti_path, dicom_folder, num_dicoms, birth_date, sex, age
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        '''
        self.cursor.execute(sql, datos)
        self.conn.commit()

    def cerrar(self):
        self.cursor.close()
        self.conn.close()
        print("Conexión MySQL cerrada.")


class DicomProcessor:
    def __init__(self, carpeta):
        self.carpeta = carpeta

    def leer_dicoms(self):
        dicoms = []
        for archivo in os.listdir(self.carpeta):
            ruta = os.path.join(self.carpeta, archivo)
            try:
                ds = pydicom.dcmread(ruta)
                dicoms.append(ds)
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")
        return dicoms

    def extraer_metadatos(self, ds, archivo):
        patient_id = str(ds.get("PatientID", ""))
        patient_name = str(ds.get("PatientName", ""))
        birth_date = str(ds.get("PatientBirthDate", ""))
        sex = str(ds.get("PatientSex", ""))
        age = str(ds.get("PatientAge", ""))
        return (archivo, patient_id, patient_name, birth_date, sex, age)

    def convertir_a_nifti(self, carpeta_salida):
        os.makedirs(carpeta_salida, exist_ok=True)
        dicom2nifti.convert_directory(
            self.carpeta,
            carpeta_salida,
            compression=True,
            reorient=True
        )
        nifti_files = [f for f in os.listdir(carpeta_salida) if f.endswith(".nii.gz")]
        if not nifti_files:
            raise FileNotFoundError("No se encontró Nifti generado.")
        nifti_path = os.path.join(carpeta_salida, nifti_files[0])
        return nifti_path
    

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.filename = os.path.basename(image_path)
        self.img = self.load_image()
        self.histR = None
        self.histG = None
        self.histB = None

    def load_image(self):
        """Carga la imagen JPG o PNG y la convierte a RGB."""
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {self.image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def calcular_histograma_manual(self):
        """Calcula el histograma R, G y B manualmente"""
        histR = np.zeros(256)
        histG = np.zeros(256)
        histB = np.zeros(256)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                r = self.img[i, j, 0]
                g = self.img[i, j, 1]
                b = self.img[i, j, 2]
                histR[r] += 1
                histG[g] += 1
                histB[b] += 1

        self.histR, self.histG, self.histB = histR, histG, histB
        return histR, histG, histB

    def graficar_histograma(self):
        """Grafica el histograma RGB usando matplotlib."""
        if self.histR is None or self.histG is None or self.histB is None:
            raise ValueError("Primero debes calcular el histograma con calcular_histograma_manual()")

        plt.figure(figsize=(10, 5))
        plt.plot(self.histR, color='r', label='Rojo')
        plt.plot(self.histG, color='g', label='Verde')
        plt.plot(self.histB, color='b', label='Azul')
        plt.title('Histograma de Color')
        plt.xlabel('Nivel de Intensidad (0-255)')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 255])
        plt.tight_layout()
        plt.show()

    def get_image(self):
        """Devuelve la imagen RGB cargada."""
        return self.img

    def ecualizar_histograma(self):
        """
        Ecualiza el histograma de la imagen (requiere que esté en RGB).
        Devuelve la imagen ecualizada en escala de grises.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        equ = cv2.equalizeHist(gray)
        return equ

    def conteo_celulas(self, umbral=130, kernel_size=(3, 3), iteraciones=7):
        """
        Conteo de células a partir del canal azul de la imagen.
        Devuelve:
        - Imagen con cierre morfológico
        - Máscara de componentes conectadas
        - Número de células detectadas
        """
        blue = self.img[:, :, 2]
        _, binaria = cv2.threshold(blue, umbral, 255, cv2.THRESH_BINARY)
        kernel = np.ones(kernel_size, np.uint8)
        cerrada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
        num_labels, mask = cv2.connectedComponents(cerrada)
        total_celulas = num_labels - 1  # sin contar el fondo
        return cerrada, mask, total_celulas

    def convertir_espacio_color(self, espacio='HSV'):
            """Convierte a otro espacio de color (HSV, LAB, YCrCb)"""
            espacios = {
                'HSV': cv2.COLOR_RGB2HSV,
                'LAB': cv2.COLOR_RGB2LAB,
                'YCrCb': cv2.COLOR_RGB2YCrCb
            }
            if espacio not in espacios:
                raise ValueError("Espacio inválido. Usa 'HSV', 'LAB', o 'YCrCb'.")

            return cv2.cvtColor(self.img, espacios[espacio])


    def deteccion_bordes_canny(self, min_val=100, max_val=200):
        """Aplica la detección de bordes de Canny (método no visto en clase)"""
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        bordes = cv2.Canny(gray, min_val, max_val)
        return bordes
    

    def get_metadata(self):
        """Devuelve información para  la base de datos."""
        return {
            "nombre_archivo": self.filename,
            "formato": self.filename.split('.')[-1].lower(),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ruta": self.image_path,
            "tecnica_usada": "Histograma manual RGB"
        }

class MorfologiaImagen:
    def __init__(self, img):
        """
        Recibe una imagen binaria (en escala de grises, ya umbralizada).
        """
        self.img = img
        if len(img.shape) != 2:
            raise ValueError("La imagen debe estar en escala de grises (binaria)")

    def erosion(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        erosionada = cv2.erode(self.img, kernel, iterations=iteraciones)
        return erosionada

    def dilatacion(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        dilatada = cv2.dilate(self.img, kernel, iterations=iteraciones)
        return dilatada

    def apertura(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        abierta = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
        return abierta

    def cierre(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        cerrada = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
        return cerrada

    def gradiente(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        grad = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, kernel, iterations=iteraciones)
        return grad

    def tophat(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        th = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, kernel, iterations=iteraciones)
        return th

    def blackhat(self, kernel_size=(5, 5), iteraciones=1):
        kernel = np.ones(kernel_size, np.uint8)
        bh = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, kernel, iterations=iteraciones)
        return bh

    def esqueletizar(self, metodo='skimage'):
        """
        Esqueletiza la imagen. Se puede usar:
        - metodo='skimage': usa skeletonize() de skimage
        - metodo='opencv': implementación manual con erosión y dilatación
        """
        if metodo == 'skimage':
            skeleton = skeletonize(self.img // 255)  # dividir entre 255 para convertir a 0-1
            return (skeleton * 255).astype(np.uint8)
        elif metodo == 'opencv':
            skel = np.zeros(self.img.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp_img = self.img.copy()

            while True:
                eroded = cv2.erode(temp_img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(temp_img, temp)
                skel = cv2.bitwise_or(skel, temp)
                temp_img = eroded.copy()

                if cv2.countNonZero(temp_img) == 0:
                    break

            return skel
        else:
            raise ValueError("Método no reconocido. Usa 'skimage' o 'opencv'.")


class BinarizacionImagen:
    def __init__(self, imagen_gray):
        if len(imagen_gray.shape) != 2:
            raise ValueError("La imagen debe estar en escala de grises.")
        self.img = imagen_gray

    def binario(self, umbral=125, max_val=255):
        _, img_bin = cv2.threshold(self.img, umbral, max_val, cv2.THRESH_BINARY)
        return img_bin

    def binario_inv(self, umbral=125, max_val=255):
        _, img_bin = cv2.threshold(self.img, umbral, max_val, cv2.THRESH_BINARY_INV)
        return img_bin

    def truncado(self, umbral=125, max_val=255):
        _, img_bin = cv2.threshold(self.img, umbral, max_val, cv2.THRESH_TRUNC)
        return img_bin

    def tozero(self, umbral=125, max_val=255):
        _, img_bin = cv2.threshold(self.img, umbral, max_val, cv2.THRESH_TOZERO)
        return img_bin

    def tozero_inv(self, umbral=125, max_val=255):
        _, img_bin = cv2.threshold(self.img, umbral, max_val, cv2.THRESH_TOZERO_INV)
        return img_bin

#HAGO OTRA CLASE???
# Crear base de datos
def crear_db():
    """
    Crea la base de datos 'info2_PF' si no existe.
    """
    conexion = mysql.connector.connect(user='informatica2', password='info20251', host='127.0.0.1')
    cursor = conexion.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS info2_PF")
    conexion.commit()
    cursor.close()
    conexion.close()
    print('Base de datos creada')

# Crear tabla de imágenes procesadas
def crear_tabla_imagenes():
    """
    Crea la tabla 'ImagenesProcesadas' para almacenar los metadatos de procesamiento.
    """
    conexion = mysql.connector.connect(user='informatica2', password='info20251', host='127.0.0.1', database='info2_PF')
    cursor = conexion.cursor()
    sql = '''
    CREATE TABLE IF NOT EXISTS ImagenesProcesadas (
        id INT AUTO_INCREMENT PRIMARY KEY,
        nombre_archivo VARCHAR(255) NOT NULL,
        formato VARCHAR(10) NOT NULL,
        ruta TEXT NOT NULL,
        tecnica_usada TEXT,
        fecha DATETIME NOT NULL
    )
    '''
    cursor.execute(sql)
    conexion.commit()
    cursor.close()
    conexion.close()

# Insertar un registro de imagen procesada
def insertar_imagen(metadata):
    """
    Inserta un nuevo registro de imagen procesada en la base de datos.
    metadata: diccionario con las claves 'nombre_archivo', 'formato', 'ruta', 'tecnica_usada', 'fecha'
    """
    conexion = mysql.connector.connect(user='informatica2', password='info20251', host='127.0.0.1', database='info2_PF')
    cursor = conexion.cursor()
    sql = '''
        INSERT INTO ImagenesProcesadas 
        (nombre_archivo, formato, ruta, tecnica_usada, fecha) 
        VALUES (%s, %s, %s, %s, %s)
    '''
    valores = (
        metadata['nombre_archivo'],
        metadata['formato'],
        metadata['ruta'],
        metadata['tecnica_usada'],
        metadata['fecha']
    )
    cursor.execute(sql, valores)
    conexion.commit()
    cursor.close()
    conexion.close()

# Consultar todos los registros
def obtener_imagenes_procesadas():
    """
    Retorna una lista con todos los registros almacenados en ImagenesProcesadas.
    """
    conexion = mysql.connector.connect(user='informatica2', password='info20251', host='127.0.0.1', database='info2_PF')
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM ImagenesProcesadas")
    resultados = cursor.fetchall()
    cursor.close()
    conexion.close()
    return resultados

def crear_tabla_usuarios():
    conexion = mysql.connector.connect(user='informatica2', password='info20251', host='127.0.0.1', database='info2_PF')
    cursor = conexion.cursor()
    sql = '''
    CREATE TABLE IF NOT EXISTS usuarios (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(100) NOT NULL,
        rol ENUM('imagen', 'senal') NOT NULL
    )
    '''

def crear_todas_las_tablas():
    conexion = mysql.connector.connect(
        user='informatica2',
        password='info20251',
        host='127.0.0.1',
        database='info2_PF'
    )
    cursor = conexion.cursor()

    tablas_sql = [
        '''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL,
            rol ENUM('imagen', 'senal') NOT NULL
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS metadatos (
            id INT AUTO_INCREMENT PRIMARY KEY,
            archivo VARCHAR(255),
            patient_id VARCHAR(100),
            patient_name VARCHAR(255),
            birth_date VARCHAR(20),
            sex VARCHAR(10),
            age VARCHAR(10)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS nifti_conversion (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id VARCHAR(100),
            nifti_path VARCHAR(255),
            dicom_folder VARCHAR(255),
            num_dicoms INT,
            birth_date VARCHAR(20),
            sex VARCHAR(10),
            age VARCHAR(10)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS ImagenesProcesadas (
            id INT AUTO_INCREMENT PRIMARY KEY,
            nombre_archivo VARCHAR(255) NOT NULL,
            formato VARCHAR(10) NOT NULL,
            ruta TEXT NOT NULL,
            tecnica_usada TEXT,
            fecha DATETIME NOT NULL
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS otros_archivos (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tipo_archivo ENUM('csv','mat') NOT NULL,
            nombre_archivo VARCHAR(255) NOT NULL,
            ruta_archivo TEXT NOT NULL,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    ]

    for sql in tablas_sql:
        cursor.execute(sql)

    conexion.commit()
    cursor.close()
    conexion.close()

def insertar_usuarios_por_defecto():
    """
    Inserta un conjunto de usuarios predeterminados si no existen ya.
    """
    conexion = mysql.connector.connect(
        user="informatica2",
        password="info20251",
        host="127.0.0.1",
        database="info2_PF"
    )
    cursor = conexion.cursor()

    sql = "INSERT IGNORE INTO usuarios (username, password, rol) VALUES (%s, %s, %s)"
    usuarios = [
        ("experto_imagen", "clave123", "imagen"),
        ("experto_senal", "clave456", "senal")
    ]

    cursor.executemany(sql, usuarios)
    conexion.commit()
    cursor.close()
    conexion.close()


def validar_credenciales(username, password):
    try:
        conexion = mysql.connector.connect(
            user="informatica2",
            password="info20251",
            host="127.0.0.1",
            database="info2_PF"
        )
        cursor = conexion.cursor(dictionary=True)
        query = "SELECT rol FROM usuarios WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        resultado = cursor.fetchone()
        cursor.fetchall()  # limpia cualquier resultado restante
        cursor.close()
        conexion.close()

        if resultado:
            return resultado["rol"]
        else:
            return None
    except mysql.connector.Error as e:
        print("ERROR en validar_credenciales:", e)
        return None