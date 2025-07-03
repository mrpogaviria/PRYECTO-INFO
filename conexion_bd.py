import mysql.connector

class ConexionBD:
    def __init__(self):
        self.host = "%"
        self.user = "informatica2"
        self.password = "info20251"
        self.database = "proyecto_final"
        self.conn = None
        self.cursor = None

    def conectar(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor(dictionary=True)
            print(" Conexión exitosa a la base de datos.")
        except mysql.connector.Error as e:
            print(f" Error de conexión: {e}")

    def cerrar(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print(" Conexión cerrada.")