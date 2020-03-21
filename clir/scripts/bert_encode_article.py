import re
import sys

from bert_serving.client import BertClient

from clir.db.pgsql import DB

f"""
    docker run --gpus all -p 5554:5554 -p 5555:5555 -p 5556:5556 -p 5557:5557 -v ~/workplace/dlnlp/models/cased_L-12_H-768_A-12/:/model_en -v ~/workplace/dlnlp/models/multi_cased_L-12_H-768_A-12/:/model_multi -t bert-as-service

    en:
    bert-serving-start -model_dir=/model_en  -num_worker=1 -max_seq_len=NONE -cased_tokenization -gpu_memory_fraction=0.47 -port=5554 -port_out=5555

    multi:
    bert-serving-start -model_dir=/model_multi -num_worker=1 -max_seq_len=NONE -cased_tokenization -gpu_memory_fraction=0.47 -port=5556 -port_out=5557
"""


def clean_text(text: str) -> str:
    text = re.sub("'", "", text)
    text = re.sub("(\\W)+", " ", text)
    return text


def main(lang):
    db = DB()
    en_bc = BertClient(output_fmt='list', port=5554, port_out=5555)
    ml_bc = BertClient(output_fmt='list', port=5556, port_out=5557)

    read_q = f"""
        SELECT {lang}_id, en_id, {lang}_title, en_title
        FROM wiki.{lang}_en_par
        WHERE en_length <= 4 * {lang}_length
        ORDER BY {lang}_length
        LIMIT 100
    """

    result = db.execute_query(read_q)

    lang_ids = [p[0] for p in result]
    en_ids = [p[1] for p in result]
    lang_text = [p[2] for p in result]
    en_text = [p[3] for p in result]

    ml_bc.encode(lang_text, blocking=False)
    en_bc.encode(en_text, blocking=False)

    ml_embs = ml_bc.fetch_all()
    en_embs = en_bc.fetch_all()

    db.drop_table(f"wiki.{lang}_en_titles_embs")
    db.execute_update(f"""
        CREATE TABLE wiki.{lang}_en_titles_embs (
            {lang}_id   bigint,
            en_id       bigint,
            {lang}_emb  decimal[],
            en_emb      decimal[]            
        )
    """)

    records = list(zip(lang_ids, en_ids, ml_embs, en_embs))
    bs = 1000
    j = 0
    for i in range(start=0, step=bs, stop=len(records)):
        b = records[i:i + bs]
        try:
            db.insert_records_parallel(records=b,
                                       schema_name="wiki",
                                       table_name=f"{lang}_en_titles_embs",
                                       columns=[f"{lang}_id", "en_id", f"{lang}_emb", "en_emb"])
            print(f"Inserted {bs} successfully")
        except:
            j += 1
            print(f"Failed to insert {bs} for time {j}")


if __name__ == "__main__":
    lang = sys.argv[1]
    main(lang)
