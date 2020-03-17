import mysql.connector


def main():
    cnx = mysql.connector.connect(user='root', password='abcd1234',
                                  host='127.0.0.1',
                                  database='wiki')

    cursor = cnx.cursor()

    query = f"""
        select * from wiki.langlinks limit 100
    """

    cursor.execute(query)

    for p in cursor:
        print(p)

    cursor.close()

    cnx.close()

