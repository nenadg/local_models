import spacy
from rake_nltk import Rake
from spacy.lang.en.stop_words import STOP_WORDS

nlp  = spacy.load("en_core_web_sm")
rake = Rake()
CUSTOM_STOP = {
    "show", "find", "list", "give", "me", "recommend",
    "please", "that", "are", "let"
}

def nl_to_query(text: str, max_phrases: int = 3) -> str:
    doc = nlp(text)

    # 1) Entities: TIME/DATE and GPE/LOC (with "open" pre‑prefix)
    time_ents = []
    loc_ents  = []
    for ent in doc.ents:
        val = ent.text.lower()
        if ent.label_ in ("TIME", "DATE"):
            if ent.start > 0 and doc[ent.start-1].lemma_.lower() == "open":
                val = f"open {val}"
            time_ents.append(val)
        elif ent.label_ in ("GPE", "LOC"):
            loc_ents.append(val)
    time_ents = list(dict.fromkeys(time_ents))
    loc_ents  = list(dict.fromkeys(loc_ents))

    # 2) RAKE phrases, filtered
    rake.extract_keywords_from_text(text)
    raw_phrases = rake.get_ranked_phrases()[:max_phrases]
    phrases = []
    for p in raw_phrases:
        lp = p.lower()
        if any(w in CUSTOM_STOP for w in lp.split()):
            continue
        if any(ent in lp for ent in time_ents + loc_ents):
            continue
        phrases.append(lp)
    phrases = list(dict.fromkeys(phrases))

    # 3) Build lemma‑based filters
    #    so that 'restaurants' → 'restaurant'
    phrase_lemmas = {
        tok.lemma_.lower()
        for ph in phrases
        for tok in nlp(ph)
    }
    entity_lemmas = {
        tok.lemma_.lower()
        for ent in time_ents + loc_ents
        for tok in nlp(ent)
    }

    # 4) Fallback tokens (NOUN/PROPN/ADJ), filter by lemma
    tokens = []
    seen   = set()
    for tok in doc:
        tl = tok.lemma_.lower()
        if (
            tok.pos_ in ("NOUN", "PROPN", "ADJ")
            and tl not in STOP_WORDS
            and tl not in CUSTOM_STOP
            and tl not in phrase_lemmas
            and tl not in entity_lemmas
            and tl not in seen
        ):
            tokens.append(tl)
            seen.add(tl)

    # 5) Assemble in order: time → loc → phrases → tokens
    ordered = time_ents + loc_ents + phrases + tokens
    # 6) Quote multi‑word terms
    out = [f'"{t}"' if " " in t else t for t in ordered]
    return " ".join(out)

# --- Test it ---
if __name__ == "__main__":
    # q = nl_to_query(
    #     "Show me affordable sushi restaurants in Tokyo that are open past midnight"
    # )

    q = nl_to_query("Let's create a javascript react dashboard")
    print(q)
    # → "open past midnight" tokyo "affordable sushi restaurants"
