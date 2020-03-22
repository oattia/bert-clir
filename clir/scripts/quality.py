import sys
import numpy as np

from tqdm import tqdm

from clir.cca.linear_cca import LinearCCAModel
from clir.db.pgsql import DB
from clir.sim.cosine_sim import CosineSimilarity


def main(lang):
    db = DB()

    cca = LinearCCAModel(num_components=50)
    sim = CosineSimilarity()
    f"""
            SELECT * 
            FROM (
                SELECT  en_id,
                        en_title,
                        {lang}_cands_ids,
                        {lang}_cand_titles,
                        sim,
                        ord,
                        rank() over (partition by en_id order by sim desc)
                FROM (
                    SELECT en_id,
                           en_title,
                           unnest({lang}_cands_ids) {lang}_cands_ids,
                           unnest({lang}_cand_titles) {lang}_cand_titles,
                           unnest(sim) sim,
                           generate_subscripts(sim, 1) ord
                    FROM  wiki.{lang}_en_output_{cca}_{sim}
                    WHERE en_title <> {lang}_cand_titles[1] and not train
                ) t1
            ) t2 
            WHERE ord = 1 and rank = 1
    """
    


if __name__ == "__main__":
    lang = sys.argv[1]
    main(lang)
