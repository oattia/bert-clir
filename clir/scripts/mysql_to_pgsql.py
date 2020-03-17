import mysql.connector
from clir.db.pgsql import DB


def main():
    db = DB()

    cnx = mysql.connector.connect(user='root', password='abcd1234',
                                  host='127.0.0.1',
                                  database='wiki')

    db.drop_table("wiki.en_langlinks")

    db.execute_update(f"""
        CREATE TABLE wiki.en_langlinks (
            ll_from     int,
            ll_lang     bytea,
            ll_title    bytea
        )
    """)

    cursor = cnx.cursor()

    query = f"""
        select * from wiki.langlinks
    """

    cursor.execute(query)

    t = []
    for p in cursor:
        t.append((int(p[0]), str(p[1]), str(p[2])))

    db.insert_records_parallel(records=t, schema_name='wiki', table_name='en_langlinks')

    cursor.close()

    cnx.close()


if __name__ == "__main__":
    main()
