import sys
import json

from clir.db.pgsql import DB

"""
inside: /home/oattia/workplace/dlnlp/extractors/wikiextractor

python ./WikiExtractor.py ../../dumps/en/enwiki-20200301-pages-articles-multistream.xml.bz2  -c --json -b 400G --processes 16 --filter_disambig_pages -o ../../dumps/en/enwiki_extracted
"""

def main(lang, file_path):
    db = DB()
    tuples = []

    db.drop_table(f"wiki.{lang}_wiki")

    db.execute_update(f"""
        CREATE TABLE wiki.{lang}_wiki (
            id       int,
            title    text,
            length   int,
            content  text
        )
    """)

    with open(file_path) as f:
        for line in f:
            ob = json.loads(line)
            idd = int(ob["id"])
            title = str(ob["title"])
            text = str(ob["text"])
            length = len(text.split())
            t = (idd, title, length, text)
            tuples.append(t)

    db.insert_records_parallel(records=tuples, schema_name="wiki", table_name=f"{lang}_wiki")

    db.execute_update(f"CREATE INDEX ON wiki.{lang}_wiki(id)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_wiki(title)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_wiki(length)")


if __name__ == "__main__":
    lang = sys.argv[1]
    file_path = sys.argv[2]
    main(lang, file_path)
