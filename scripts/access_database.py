import pyodbc

print([x for x in pyodbc.drivers()])

# Replace the file path with the path to your .accdb file
access_database_file = r'BaseMaterials/SampleDatabase.accdb'
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=' + access_database_file + ';'
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Replace 'TableName' with the name of a table in your database
# table_name = 'TableName'
# cursor.execute(f'SELECT * FROM {table_name}')

cursor.execute("SELECT Name FROM MSysObjects WHERE Type=1 AND Flags=0")

for row in cursor.fetchall():
    print(row)

conn.close()
