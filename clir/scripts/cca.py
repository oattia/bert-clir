import sys
import numpy as np

from clir.cca.linear_cca import LinearCCAModel
from clir.db.pgsql import DB
from clir.sim.cosine_sim import CosineSimilarity


def main(lang):
    db = DB()

    read_q = f"""
        SELECT {lang}_id, en_id, {lang}_emb, en_emb, en_title, {lang}_title
        FROM wiki.{lang}_en_titles_embs e, wiki.{lang}_en_par p
        WHERE e.{lang}_id = p.{lang}_id AND e.en_id = p.en_id
        ORDER BY random()
    """

    result = db.execute_query(read_q)

    lang_ids = [p[0] for p in result]
    en_ids = [p[1] for p in result]
    lang_embs = [p[2] for p in result]
    en_embs = [p[3] for p in result]
    en_titles = [p[4] for p in result]
    lang_titles = [p[5] for p in result]

    train_size = int(len(result) / 2)

    lang_ids_train = lang_ids[:train_size]
    en_ids_train = en_ids[:train_size]
    lang_embs_train = lang_embs[:train_size]
    en_embs_train = en_embs[:train_size]
    en_titles_train = en_titles[:train_size]
    lang_titles_train = lang_titles[:train_size]

    trainids = set(en_ids_train)

    assert len(lang_embs_train) == len(en_embs_train)

    cca = LinearCCAModel(num_components=50)
    sim = CosineSimilarity()

    cca.train(en_embs_train, lang_embs_train)

    # construct test sets:
    lang_ids_train = lang_ids[train_size:]
    en_ids_train = en_ids[train_size:]
    lang_embs_test = lang_embs[train_size:]
    en_embs_test = en_embs[train_size:]
    en_titles_test = en_titles[train_size:]
    lang_titles_test = lang_titles[train_size:]

    assert len(lang_embs_test) == len(en_embs_test)

    records = []
    for i in range(len(lang_ids)):
        random_perm = [idx for idx in np.random.permutation(len(lang_embs))[:5] if idx != i]
        candidates_embs = [lang_embs[i]] + [lang_embs[idx] for idx in random_perm]

        np.random.shuffle(candidates_embs)
        en_x, lang_x = cca.predict(en_embs_test[i], candidates_embs)
        sims = sim.compute(en_x, lang_x)

        en_id = en_ids[i]
        cands_ids = [lang_ids[idx] for idx in random_perm]
        cands_titles = [lang_titles[idx] for idx in random_perm]

        record = (en_id, en_titles[i], en_id in trainids, cands_ids, cands_titles, sims)
        records.append(record)

    db.drop_table(f"wiki.{lang}_en_output_{cca}_{sim}")

    db.execute_update(f"""
        CREATE TABLE wiki.{lang}_en_output_{cca}_{sim} (
            en_id               int,
            en_title            text,
            train               boolean,
            {lang}_cands_ids    int[],
            {lang}_cand_titles  text[],
            sim                 decimal[]
        )
    """)

    bs = 10000
    j = 0
    for i in range(0, len(records), bs):
        b = records[i:i + bs]
        try:
            db.insert_records_parallel(records=b,
                                       schema_name="wiki",
                                       table_name=f"{lang}_en_output_{cca}_{sim}")
            print(f"Inserted {bs} successfully: {i}/{len(records)} done")
        except:
            j += 1
            print(f"Failed to insert {bs} for time {j}")

    db.execute_update(f"CREATE INDEX ON wiki.{lang}_en_output_{cca}_{sim}(en_id)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_en_output_{cca}_{sim}(en_title)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_en_output_{cca}_{sim}({lang}_cands_ids)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_en_output_{cca}_{sim}({lang}_cand_titles)")
    db.execute_update(f"CREATE INDEX ON wiki.{lang}_en_output_{cca}_{sim}(train)")


if __name__ == "__main__":
    lang = sys.argv[1]
    main(lang)
