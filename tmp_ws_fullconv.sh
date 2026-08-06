#!/usr/bin/env bash
set -e
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
SE="$WS/search_engine"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONPATH="$WS"
export NLTK_DISABLE_IMPORT_SECURITY=1
cat > "$SE/convert_product_file_format_full.py" <<'PYEOF'
import sys, json
from tqdm import tqdm
sys.path.insert(0, '../')
from web_agent_site.utils import BASE_DIR
from web_agent_site.engine.engine import load_products

FULL = f'{BASE_DIR}/../data/items_shuffle.json'
all_products, *_ = load_products(filepath=FULL)
print('total products after filter:', len(all_products))

docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    for option_name, option_contents in (p.get('options') or {}).items():
        option_texts.append(f'{option_name}: {", ".join(option_contents)}')
    option_text = ', and '.join(option_texts)
    bullet = p['BulletPoints']
    if isinstance(bullet, list):
        bullet = bullet[0] if bullet else ''
    doc = {
        'id': p['asin'],
        'contents': ' '.join([p['Title'], p['Description'], str(bullet), option_text]).lower(),
        'product': p,
    }
    docs.append(doc)

with open('./resources/documents.jsonl', 'w+') as f:
    for doc in docs:
        f.write(json.dumps(doc) + '\n')
print('wrote resources/documents.jsonl with', len(docs), 'docs')
PYEOF
echo "---RUN FULL CONVERT---"
cd "$SE"
$PY convert_product_file_format_full.py 2>&1 | grep -vE "Keys cleaned|Products loaded|Attributes loaded|it/s\]" | tail -15
echo "---FULL DOC COUNT---"
wc -l "$SE/resources/documents.jsonl"