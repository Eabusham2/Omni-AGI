"""Bounded actual region/band/pathway pages from the authenticated index."""
from __future__ import annotations


def _cluster(row):
    kind, identifier, label, region, source, target, count, active, total, maximum, negative, zero, positive, band = row
    return {"id": identifier, "label": label, "kind": kind, "count": int(count), "activeCount": int(active),
        "meanActivation": float(total) / count if count else 0., "maxActivation": float(maximum),
        "effectiveWeights": {"negative": int(negative), "zero": int(zero), "positive": int(positive)},
        **({"region": region} if region is not None else {}),
        **({"activationBand": band} if band is not None else {}),
        **({"sourceRegion": source, "targetRegion": target} if source is not None else {})}


def aggregate_page(index, *, zoom, region, search, offset, page_size):
    """SQL disk-backed groups; returned endpoint anchors are real region totals."""
    view = index.view
    index.admit(1024 * 1024 + page_size * 8192, "inspection bounded aggregate viewport")
    index.session.reserve_disk(65536 + index.path.stat().st_size * 2, "inspection disk-backed aggregate scratch")
    with index.connection() as connection:
        connection.execute("CREATE TEMP TABLE selected_bands(region TEXT,band TEXT,count INTEGER,active INTEGER,total REAL,maximum REAL)")
        if view.attention_legacy_raw_active:
            connection.execute("INSERT INTO selected_bands SELECT * FROM raw_bands")
        else:
            connection.execute("CREATE TEMP TABLE active_ids(id TEXT PRIMARY KEY) WITHOUT ROWID")
            for number, identifier in enumerate(view.attention_active_neuron_ids):
                index.check()
                if number % 128 == 0: index.session.reserve_disk(65536, "inspection attention index window")
                connection.execute("INSERT INTO active_ids VALUES(?)", (identifier,))
            actual = connection.execute("SELECT count(*) FROM active_ids JOIN neuron_activity USING(id)").fetchone()[0]
            if actual != len(view.attention_active_neuron_ids): raise ValueError("inspection attention overlay names unknown neurons")
            connection.execute("INSERT INTO selected_bands SELECT n.region,CASE WHEN n.activation>=0.66 THEN 'high' WHEN n.activation>=0.2 THEN 'medium' ELSE 'quiet' END,"
                               "count(*),sum(n.activation>=0.1),sum(n.activation),max(n.activation) FROM neuron_activity AS n JOIN active_ids AS a USING(id) GROUP BY n.region,2")
            connection.execute("INSERT INTO selected_bands SELECT r.region,'quiet',r.total-count(a.id),0,0,0 FROM region_totals AS r "
                               "LEFT JOIN neuron_activity AS n ON n.region=r.region LEFT JOIN active_ids AS a ON a.id=n.id GROUP BY r.region HAVING r.total-count(a.id)>0")
        connection.execute("CREATE TEMP TABLE clusters(kind TEXT,id TEXT,label TEXT,region TEXT,source_region TEXT,target_region TEXT,"
                           "count INTEGER,active INTEGER,total REAL,maximum REAL,negative INTEGER,zero_count INTEGER,positive INTEGER,band TEXT)")
        # Both queries use the same actual masked band source; zoom changes
        # grouping only, never manufactures a representative contact/activation.
        if zoom < .35:
            connection.execute("INSERT INTO clusters SELECT 'region','region:'||region,region,region,NULL,NULL,sum(count),sum(active),sum(total),max(maximum),0,0,0,NULL FROM selected_bands GROUP BY region")
        else:
            connection.execute("INSERT INTO clusters SELECT 'activation-band','region:'||region||':'||band,region||' · '||band,region,NULL,NULL,sum(count),sum(active),sum(total),max(maximum),0,0,0,band FROM selected_bands GROUP BY region,band")
        connection.execute("INSERT INTO clusters SELECT 'pathway','pathway:'||source_region||'>'||target_region,source_region||' → '||target_region,NULL,source_region,target_region,count,0,0,0,negative,zero_count,positive,NULL FROM pathway_groups")
        clauses, arguments = [], []
        if region:
            clauses.append("(omni_casefold(region)=? OR omni_casefold(source_region)=? OR omni_casefold(target_region)=?)")
            arguments.extend((region.casefold(),) * 3)
        if search:
            clauses.append("INSTR(omni_casefold(id||' '||label),?)>0"); arguments.append(search.casefold())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        matched = connection.execute("SELECT count(*) FROM clusters" + where, arguments).fetchone()[0]
        page = [_cluster(row) for row in connection.execute("SELECT * FROM clusters" + where + " ORDER BY kind,count DESC,id LIMIT ? OFFSET ?", (*arguments, page_size, offset))]
        anchors, requested = [], set()
        for item in page:
            for endpoint in ([item["sourceRegion"], item["targetRegion"]] if item["kind"] == "pathway" else [item["region"]]):
                if endpoint in requested: continue
                requested.add(endpoint); index.check()
                row = connection.execute("SELECT 'region','region:'||region,region,region,NULL,NULL,sum(count),sum(active),sum(total),max(maximum),0,0,0,NULL FROM selected_bands WHERE region=? GROUP BY region", (endpoint,)).fetchone()
                if row is not None: anchors.append(_cluster(row))
        return page, anchors, int(matched)
