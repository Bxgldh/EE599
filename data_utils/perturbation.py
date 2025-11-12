import spacy, random, re
nlp = spacy.load("en_core_web_sm")
SYN_DICT = {"rise":["increase","climb"],"fall":["decline","drop"]}
def synonym_replace(text,max_repl=2):
    doc=nlp(text);idx=[i for i,t in enumerate(doc) if t.lemma_.lower() in SYN_DICT]
    random.shuffle(idx);out=[t.text for t in doc]
    for i in idx[:max_repl]: out[i]=random.choice(SYN_DICT[doc[i].lemma_.lower()])
    return " ".join(out)
