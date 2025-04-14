import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from rake_nltk import Rake
rake_nltk_var = Rake()
text = """who was the main character in 1997 movie \"As good as it gets\" """
rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()[:5]
print(keyword_extracted)