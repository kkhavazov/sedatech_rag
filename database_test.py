import pyodbc

server = "192.168.125.216"
database = "ShopCenterSL2014"
pwd = "seda08154711tech"

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={server};"
    "DATABASE=YourDatabase;"
    "UID=rag_user;"
    "PWD=YourPassword;"
    "Encrypt=no;"
)

cursor = conn.cursor()

cursor.execute("SELECT SYSTEM_USER")
print("Logged in as:", cursor.fetchone()[0])

cursor.execute("SELECT TOP (5) * FROM YourTable")

for row in cursor.fetchall():
    print(row)

conn.close()