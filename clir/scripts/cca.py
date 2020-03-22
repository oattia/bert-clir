import sys
import numpy as np

from clir.cca.linear_cca import LinearCCAModel
from clir.db.pgsql import DB
from clir.sim.cosine_sim import CosineSimilarity


def main(lang):
    db = DB()

    read_q = f"""
        SELECT {lang}_id, en_id, {lang}_emb, en_emb
        FROM wiki.{lang}_en_titles_embs
        ORDER BY random()
    """

    result = db.execute_query(read_q)

    lang_ids = [p[0] for p in result]
    en_ids = [p[1] for p in result]

    lang_embs = [p[2] for p in result]
    en_embs = [p[3] for p in result]

    train_size = int(len(result) / 2)

    lang_ids_train = lang_ids[:train_size]
    en_ids_train = en_ids[:train_size]
    lang_embs_train = lang_embs[:train_size]
    en_embs_train = en_embs[:train_size]

    assert len(lang_embs_train) == len(en_embs_train)

    cca = LinearCCAModel(num_components=50)
    sim = CosineSimilarity()

    cca.train(en_embs_train, lang_embs_train)

    # construct test sets:
    lang_ids_train = lang_ids[train_size:]
    en_ids_train = en_ids[train_size:]
    lang_embs_test = lang_embs[train_size:]
    en_embs_test = en_embs[train_size:]

    assert len(lang_embs_test) == len(en_embs_test)

    test_ds = []
    for i in range(len(lang_embs_test)):
        random_perm = [idx for idx in np.random.permutation(len(lang_embs_test))[:5] if idx != i]
        candidates = [lang_embs_test[i]] + [lang_embs_test[idx] for idx in random_perm]
        np.random.shuffle(candidates)
        pair = (en_embs_test[i], candidates)
        test_ds.append(pair)

        cca.predict(en_embs_train, lang_embs_train)


    db.drop_table(f"wiki.{lang}_en_output")

    db.execute_update(f"""
        CREATE UNLOGGED TABLE wiki.{lang}_en_output_{cca}_{sim} (
            en_id               int,
            {lang}_cands_ids    int[],
            {lang}_emb  decimal[],
            en_emb      decimal[]       
        )
    """)


if __name__ == "__main__":
    lang = sys.argv[1]
    main(lang)
