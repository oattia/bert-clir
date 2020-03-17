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
            ll_lang     text
        )
    """)

    cursor = cnx.cursor()

    query = f"""
        select ll_from, ll_lang from wiki.langlinks
    """

    cursor.execute(query)

    t = []
    for p in cursor:
        # print(type(p[0]))
        # print(type(p[1]))
        # print(type(p[2]))
        # # (int(p[0]), str(p[1]), str(p[2]))
        t.append(p)

    db.insert_records_parallel(records=t, schema_name='wiki', table_name='en_langlinks')

    cursor.close()

    cnx.close()


if __name__ == "__main__":
    main()
