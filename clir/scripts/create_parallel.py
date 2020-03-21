import sys

from clir.db.pgsql import DB


def main(lang):
    db = DB()

    db.drop_table(f"wiki.{lang}_en_par")
    db.execute_update(f"""
        CREATE TABLE wiki.{lang}_en_par (
            {lang}_id       bigint,
            en_id           bigint, 
            {lang}_title    text,
            en_title        text, 
            {lang}_length   int, 
            en_length       int,
            {lang}_content  text,
            en_content      text
        )
    """)

    par_q = f"""
        INSERT INTO wiki.{lang}_en_par
        WITH {lang}_en_par_temp AS (
            SELECT DISTINCT a.id {lang}_id, ll.ll_from en_id, a.title {lang}_title, b.title en_title
            FROM wiki.{lang}_wiki a, wiki.en_wiki b, wiki.en_langlinks ll 
            WHERE a.title = ll.ll_title 
              AND b.id = ll.ll_from
        )
        SELECT p.{lang}_id, p.en_id, p.{lang}_title, p.en_title, 
               a.length {lang}_length, b.length en_length,
               a.content {lang}_content, b.content en_content
        FROM {lang}_en_par_temp p, wiki.{lang}_wiki a, wiki.en_wiki b
        WHERE p.{lang}_id = a.id
          AND p.en_id = b.id
    """
    db.execute_update(par_q)

    db.execute_update(f"CREATE INDEX ON wiki.{lang}_wiki(id)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_wiki(title)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_wiki(length)")


if __name__ == "__main__":
    lang = sys.argv[1]
    main(lang)
