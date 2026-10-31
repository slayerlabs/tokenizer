import json, sys, urllib.request, urllib.parse
from tokenizers import Tokenizer
titles = ["Polska","Kraków","Warszawa","Wisła","Mikołaj Kopernik","Maria Skłodowska-Curie","Fryderyk Chopin","Adam Mickiewicz","Język polski","Tatry","Gdańsk","Bitwa pod Grunwaldem","Pierogi","Fotosynteza","Komputer","Internet","Piłka nożna","Unia Europejska","Konstytucja Rzeczypospolitej Polskiej","Henryk Sienkiewicz"]
texts = []
import time
for t in titles:
    time.sleep(2)
    q = urllib.parse.urlencode({"action":"query","prop":"extracts","explaintext":1,"titles":t,"format":"json","redirects":1})
    req = urllib.request.Request("https://pl.wikipedia.org/w/api.php?"+q, headers={"User-Agent":"tokenizer-fertility-check/0.1"})
    pages = json.load(urllib.request.urlopen(req))["query"]["pages"]
    texts += [p for pg in pages.values() for p in pg.get("extract","").split("\n") if len(p) > 80]
base = Tokenizer.from_file(sys.argv[1]); new = Tokenizer.from_file(sys.argv[2])
words = sum(len(t.split()) for t in texts); chars = sum(map(len, texts))
def n(tok): return sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
nb, nn = n(base), n(new)
changed = sum(a.ids != b.ids for a, b in zip(base.encode_batch(texts, add_special_tokens=False), new.encode_batch(texts, add_special_tokens=False)))
import tiktoken
r = {"paragraphs": len(texts), "words": words,
     "bielik_tokens_per_word": round(nb/words,3), "szl_tokens_per_word": round(nn/words,3),
     "paragraphs_with_changed_segmentation": changed}
for name in ("cl100k_base","o200k_base"):
    enc = tiktoken.get_encoding(name); r[name+"_tokens_per_word"] = round(sum(map(len, enc.encode_ordinary_batch(texts)))/words,3)
print(json.dumps(r, indent=1))
