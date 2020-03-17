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
            ll_lang     text,
            ll_title    text
        )
    """)

    cursor = cnx.cursor()

    query = f"""
        select ll_from, ll_lang, ll_title 
        from wiki.langlinks
        where ll_lang in ('ar', 'de')
        order by ll_from
    """

    cursor.execute(query)

    t = []
    for p in cursor:
        t.append(p)

    db.insert_records_parallel(records=t, schema_name='wiki', table_name='en_langlinks')

    cursor.close()

    cnx.close()


if __name__ == "__main__":
    main()
