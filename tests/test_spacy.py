import spacy
nlp = spacy.load("en_core_web_sm")
text = """who was the main character in 1997 movie \"As good as it gets\" """
doc = nlp(text)
print(doc.ents)