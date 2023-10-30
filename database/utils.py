import sqlite3

def school():
    conn = sqlite3.connect("database/tuvantuyensinh.db")
    cursor = conn.execute("SELECT * FROM School")
    data = cursor.fetchall()
    conn.close()
    return data


if __name__ == '__main__':
    print(school())