import sqlite3

def school():
    conn = sqlite3.connect("database/tuvantuyensinh.db")
    cursor = conn.execute("SELECT * FROM School")
    data = cursor.fetchone()
    conn.close()
    return data

def get_job_data():
    conn = sqlite3.connect("database/tuvantuyensinh.db")
    cursor = conn.execute("SELECT * FROM Job")
    data = cursor.fetchall()
    conn.close()
    return data

def get_scholarship_data():
    conn = sqlite3.connect("database/tuvantuyensinh.db")
    cursor = conn.execute("SELECT * FROM Scholarship")
    data = cursor.fetchall()
    conn.close()
    return data

import sqlite3

def get_tuition_data():
    conn = sqlite3.connect("database/tuvantuyensinh.db")
    cursor = conn.execute("SELECT * FROM Tuition")
    data = cursor.fetchall()
    conn.close()
    return data

def get_target_data():
    conn = sqlite3.connect("database/tuvantuyensinh.db")
    cursor = conn.execute("SELECT * FROM Target")
    data = cursor.fetchall()
    conn.close()
    return data

if __name__ == '__main__':
    print(school())