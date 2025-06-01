import re
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV

class QuestionClassifier:
    """
    A classifier that categorizes questions into knowledge categories.
    Focuses exclusively on knowledge-based categories without original domain categories.
    """

    def __init__(self):
        """Initialize the classifier with pre-defined categories and training data"""
        # Use consecutive indices starting from 0 to avoid index out of bounds errors
        self.categories = {
            'unknown': 0,
            'declarative': 0,        # Facts and information
            'procedural_knowledge': 1,   # How to do something
            'experiential': 2,       # Personal experiences
            'tacit': 3,              # Intuition, insights
            'explicit': 4,           # Articulated knowledge
            'conceptual_knowledge': 5,  # Principles and theories
            'contextual': 6         # Environmental understanding
        }

        # Map from category name to internal index (for scikit-learn)
        self.category_to_index = {
            'unknown': 0,
            'declarative': 0,
            'procedural_knowledge': 1,
            'experiential': 2,
            'tacit': 3,
            'explicit': 4,
            'conceptual_knowledge': 5,
            'contextual': 6
        }

        # Map from internal index to category name
        self.index_to_category = {
            0: 'declarative',
            1: 'procedural_knowledge',
            2: 'experiential',
            3: 'tacit',
            4: 'explicit',
            5: 'conceptual_knowledge',
            6: 'contextual'
        }

        # Subcategories dictionary
        self.subcategories = {
            'declarative': [
                'historical', 'scientific', 'geographic', 'mathematical',
                'linguistic', 'cultural', 'biographical', 'legal',
                'technological', 'literary'
            ],
            'procedural_knowledge': [
                'cooking', 'programming', 'mechanical', 'artistic',
                'musical', 'sports', 'medical', 'crafting',
                'language_usage', 'problem_solving'
            ],
            'experiential': [
                'travel', 'work', 'social', 'personal',
                'educational', 'cultural_immersion', 'volunteer',
                'leadership', 'creative', 'life_lessons'
            ],
            'tacit': [
                'intuition', 'emotional_intelligence', 'leadership_insights',
                'problem_solving_instincts', 'decision_making',
                'creative_insights', 'negotiation', 'communication',
                'relationship_building', 'personal_judgment'
            ],
            'explicit': [
                'manuals', 'textbooks', 'research_papers', 'tutorials',
                'instructional_videos', 'policy_documents', 'training_materials',
                'academic_articles', 'software_documentation', 'legal_contracts'
            ],
            'conceptual_knowledge': [
                'scientific_theories', 'philosophical_concepts', 'mathematical_theories',
                'economic_theories', 'sociological_theories', 'psychological_theories',
                'political_theories', 'ethical_concepts', 'literary_themes', 'design_principles'
            ],
            'contextual': [
                'cultural_norms', 'historical_context', 'economic_environment',
                'political_climate', 'social_dynamics', 'geographic_influences',
                'technological_landscape', 'organizational_culture',
                'market_trends', 'legal_framework'
            ]
        }

        # Create the classifier pipeline
        self.clf = Pipeline([
            ('vect', CountVectorizer(analyzer='word', ngram_range=(1, 2))),
            ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced')))
        ])

        # Training data - simple examples for each category
        self.training_data = []
        self.training_labels = []

        # Initialize with predefined examples
        self._initialize_training_data()

        # Train the classifier
        self._train_classifier()

        # Pattern matching rules (for high confidence cases)
        self.patterns = {
            # Knowledge category patterns
            'declarative': [
                r'\b(what is|who is|where is|when did|tell me about)\b',
                r'\b(facts|information|details|data)\b.*\b(about|on|regarding)\b',
                r'\bdefine\b.*',
                r'\bexplain\b.*\b(concept|term|meaning)\b'
            ],
            'procedural_knowledge': [
                r'\b(how to|how do I|steps to|guide for|instructions for)\b',
                r'\b(process|method|technique|approach|procedure)\b',
                r'\b(steps|instructions|guide)\b.*\b(for|to)\b'
            ],
            'experiential': [
                r'\b(what was it like|how did you feel|your experience|share experience)\b',
                r'\b(personal|firsthand|lived|direct)\b.*\b(experience|account|story)\b',
                r'\bhave you ever\b',
                r'\bwhat happens when\b'
            ],
            'tacit': [
                r'\b(intuition|gut feeling|judgment|sense)\b',
                r'\b(what do you think|in your opinion|your perspective)\b',
                r'\bwhat would you do\b',
                r'\badvice\b.*\b(on|about|for)\b'
            ],
            'explicit': [
                r'\b(manual|guide|documentation|specifications)\b',
                r'\b(according to|as stated in|referenced in)\b',
                r'\b(published|written|documented)\b',
                r'\b(article|paper|book|journal)\b.*\b(say|state|mention)\b'
            ],
            'conceptual_knowledge': [
                r'\b(theory|principle|concept|framework)\b',
                r'\b(why|reason for|explanation for)\b',
                r'\b(philosophy|ideology|doctrine)\b',
                r'\b(fundamentals|basics|essence)\b.*\b(of|about)\b'
            ],
            'contextual': [
                r'\b(context|environment|setting|circumstances)\b',
                r'\b(cultural|social|historical|political)\b.*\b(context|background|framework)\b',
                r'\b(during|in the context of|considering)\b',
                r'\b(given|under|within)\b.*\b(conditions|circumstances|situation)\b'
            ]
        }

    def _initialize_training_data(self):
        """Initialize with predefined training examples for knowledge categories"""

        # Declarative knowledge examples
        declarative_examples = [
            "what is the capital of France?",
            "who was the first president of the United States?",
            "when did World War II end?",
            "where is the Great Barrier Reef located?",
            "what is the speed of light?",
            "tell me about quantum physics",
            "what are the components of DNA?",
            "who invented the telephone?",
            "what is the population of Tokyo?",
            "what are the main export products of Brazil?",
            "define photosynthesis",
            "what is the meaning of the word 'ephemeral'?",
            "what causes lightning?",
            "who wrote War and Peace?",
            "what is the height of Mount Everest?",
            "what is 5 + 3?",
            "calculate 10 - 7",
            "what's 7 times 8",
            "what's the result of 12 * 4?",
            "divide 100 by 5"
        ]

        # Procedural knowledge examples
        procedural_knowledge_examples = [
            "how to bake a chocolate cake?",
            "what are the steps to change a car tire?",
            "how do I reset my password?",
            "guide for installing Python on Windows",
            "how to fix a leaky faucet?",
            "what's the process for applying to a university?",
            "instructions for assembling IKEA furniture",
            "how to grow tomatoes from seeds?",
            "steps to create a website",
            "how do I perform CPR?",
            "what's the technique for making perfect sushi rice?",
            "how to train a dog to sit?",
            "what's the proper way to meditate?",
            "how do I create a budget spreadsheet?",
            "what's the method for solving quadratic equations?"
        ]

        # Experiential knowledge examples
        experiential_examples = [
            "what is it like to climb Mount Everest?",
            "how does it feel to win an Olympic medal?",
            "what was your experience visiting Japan?",
            "share your experience of childbirth",
            "what happens when you skydive for the first time?",
            "have you ever been to the Amazon rainforest?",
            "what's it like to work at Google?",
            "how does depression feel?",
            "what was it like growing up in the 1980s?",
            "describe the experience of running a marathon",
            "what's it like to live in New York City?",
            "how does it feel to learn a new language?",
            "what was it like to witness a solar eclipse?",
            "share your experience with meditation",
            "what happens during acupuncture?"
        ]

        # Tacit knowledge examples
        tacit_examples = [
            "what's your intuition about the stock market?",
            "what do you think about this painting?",
            "should I quit my job to travel the world?",
            "how can I tell if someone is lying?",
            "what's your perspective on artificial intelligence ethics?",
            "how do I know if I'm making the right decision?",
            "what would you do in my situation?",
            "how can I become a better listener?",
            "what's your advice on public speaking?",
            "how do you know when to trust someone?",
            "what's your gut feeling about this business proposal?",
            "how can I improve my leadership skills?",
            "what makes a good teacher?",
            "how do you handle difficult conversations?",
            "what's the secret to maintaining work-life balance?"
        ]

        # Explicit knowledge examples
        explicit_examples = [
            "what does the user manual say about resetting the device?",
            "according to the Constitution, who has the power to declare war?",
            "what are the specifications of the iPhone 13?",
            "what does the Chicago Manual of Style say about Oxford commas?",
            "as stated in the company policy, what is the vacation allowance?",
            "what are the instructions in the assembly guide?",
            "what does the research paper conclude about climate change?",
            "according to the recipe, how much flour should I use?",
            "what does the API documentation specify for this function?",
            "what are the rules of chess according to the official handbook?",
            "what does the tax code say about deducting home office expenses?",
            "according to safety regulations, what protective equipment is required?",
            "what does the textbook explain about photosynthesis?",
            "what are the terms and conditions for this service?",
            "what does the contract specify about termination?",
            "how do you say hello in French?",
            "translate book to Spanish",
            "how to say thank you in Japanese",
            "what is the German word for water?"
        ]

        # Conceptual knowledge examples
        conceptual_knowledge_examples = [
            "explain the theory of relativity",
            "what is the principle of natural selection?",
            "why does supply and demand affect prices?",
            "what is the concept of justice in philosophy?",
            "explain the principles of object-oriented programming",
            "what is the fundamental idea behind democracy?",
            "explain the concept of sustainability",
            "what are the main theories of personality in psychology?",
            "what is the philosophical concept of existentialism?",
            "explain the principles of quantum mechanics",
            "what is the theory of multiple intelligences?",
            "why do planes fly? Explain the principles of aerodynamics",
            "what is the concept of opportunity cost in economics?",
            "explain the philosophical idea of moral relativism",
            "what are the fundamental principles of design?"
        ]

        # Contextual knowledge examples
        contextual_examples = [
            "how did the Great Depression influence American politics?",
            "what was the cultural context of Shakespeare's plays?",
            "how does the legal framework affect healthcare delivery?",
            "what is the political climate in Brazil right now?",
            "how does geographic location influence architectural styles?",
            "what was the social context of the Civil Rights Movement?",
            "how does the economic environment affect small businesses?",
            "what is the historical context of the French Revolution?",
            "how does organizational culture impact employee productivity?",
            "what's the technological landscape of renewable energy?",
            "how do market trends affect investment strategies?",
            "what is the regulatory environment for cryptocurrencies?",
            "how does the educational context differ between countries?",
            "what's the environmental context of sustainable agriculture?",
            "how does the social dynamic influence teenage behavior?"
        ]

        # Add all examples to training data
        for examples, category in [
            (declarative_examples, 'declarative'),
            (procedural_knowledge_examples, 'procedural_knowledge'),
            (experiential_examples, 'experiential'),
            (tacit_examples, 'tacit'),
            (explicit_examples, 'explicit'),
            (conceptual_knowledge_examples, 'conceptual_knowledge'),
            (contextual_examples, 'contextual')
        ]:
            for example in examples:
                self.training_data.append(example)
                self.training_labels.append(self.category_to_index[category])

                # Add variations with different question forms
                variations = self._generate_variations(example)
                for variation in variations:
                    self.training_data.append(variation)
                    self.training_labels.append(self.category_to_index[category])

    def _generate_variations(self, question):
        """Generate variations of a question to improve training robustness"""
        variations = []

        # Convert to lowercase for consistent processing
        q = question.lower()

        # Remove question marks
        if q.endswith('?'):
            variations.append(q[:-1])

        # Add different question prefixes
        if not q.startswith(('what', 'how', 'who', 'when', 'where', 'why')):
            variations.append(f"what is {q}")

        # Add "I want to know" prefix
        variations.append(f"I want to know {q}")

        # Add "tell me" prefix
        variations.append(f"tell me {q}")

        # Add "I need to" prefix for procedural questions
        if "how to" in q or "how do" in q:
            variations.append(q.replace("how to", "I need to"))
            variations.append(q.replace("how do", "I need to know how to"))

        return variations

    def _train_classifier(self):
        """Train the SVM classifier on the prepared data"""
        if not self.training_data:
            return

        # Convert to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)

        # Fit the classifier
        self.clf.fit(X, y)

    def classify(self, question):
        """
        Classify a question into one of the knowledge categories.
        Returns both the category and a confidence score.

        Args:
            question: The question text to classify

        Returns:
            tuple: (category_name, confidence_score)
        """
        # First check for strong pattern matches
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, question.lower()):
                    subcategory, subcategory_confidence = self.identify_subcategory(question, category)
                    return category, 0.95, subcategory, subcategory_confidence  # High confidence for pattern matches

        # If no strong pattern match, use the trained classifier
        try:
            # Get prediction and probability using internal indices
            internal_idx = self.clf.predict([question])[0]
            probabilities = self.clf.predict_proba([question])[0]
            confidence = probabilities[internal_idx]

            # Map internal index to category name
            category_name = self.index_to_category.get(internal_idx, "unknown")
            subcategory, subcategory_confidence = self.identify_subcategory(question, category_name)

            return category_name, confidence, subcategory, subcategory_confidence

        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0, "unknown", 0.0

    def identify_subcategory(self, text, main_category):
        """
        Identify the most likely subcategory for a given text and main category.

        Args:
            text: The text to analyze
            main_category: The main knowledge category

        Returns:
            tuple: (subcategory, confidence)
        """
        if main_category not in self.subcategories:
            return None, 0.0

        subcategories = self.subcategories[main_category]
        best_subcategory = None
        best_score = 0.0

        # Convert text to lowercase for matching
        text_lower = text.lower()

        for subcategory in subcategories:
            # Convert subcategory to readable format for matching
            readable = subcategory.replace('_', ' ')

            # Initial score based on direct keyword match
            score = 0.0

            # Check for direct mention of the subcategory
            if readable in text_lower:
                score += 0.7

            # Check for related terms based on subcategory
            related_terms = self._get_related_terms(subcategory)
            for term in related_terms:
                if term in text_lower:
                    score += 0.3
                    break

            if score > best_score:
                best_score = score
                best_subcategory = subcategory

        # If no strong match, use a more generic approach
        if best_score < 0.3:
            # Find the subcategory with the most word overlap
            for subcategory in subcategories:
                readable = subcategory.replace('_', ' ')
                words = readable.split()

                overlap = sum(1 for word in words if word in text_lower)
                score = overlap / max(1, len(words))

                if score > best_score:
                    best_score = score
                    best_subcategory = subcategory

        return best_subcategory, best_score

    def _get_related_terms(self, subcategory):
        """Get related terms for a subcategory to improve matching"""
        related_terms = {
            # Declarative subcategories
            'historical': ['history', 'past', 'ancient', 'century', 'era', 'period', 'dynasty', 'date', 'timeline'],
            'scientific': ['science', 'research', 'experiment', 'laboratory', 'hypothesis', 'theory', 'discovery', 'data', 'observation'],
            'geographic': ['geography', 'location', 'region', 'country', 'continent', 'map', 'terrain', 'climate', 'coordinates'],
            'mathematical': ['math', 'calculation', 'formula', 'equation', 'algebra', 'geometry', 'number', 'calculation', 'computation'],
            'linguistic': ['language', 'grammar', 'vocabulary', 'syntax', 'semantics', 'word', 'phrase', 'meaning', 'translation'],
            'cultural': ['culture', 'tradition', 'custom', 'heritage', 'ritual', 'festival', 'celebration', 'practice', 'norm'],
            'biographical': ['biography', 'life', 'person', 'individual', 'career', 'achievement', 'birth', 'death', 'history'],
            'legal': ['law', 'regulation', 'statute', 'legislation', 'court', 'legal', 'judicial', 'rights', 'justice'],
            'technological': ['technology', 'device', 'gadget', 'innovation', 'invention', 'technical', 'product', 'hardware', 'software'],
            'literary': ['literature', 'novel', 'poem', 'author', 'genre', 'fiction', 'narrative', 'book', 'story'],

            # Procedural subcategories
            'cooking': ['cook', 'recipe', 'bake', 'kitchen', 'meal', 'ingredient', 'dish', 'food', 'preparation', 'chef'],
            'programming': ['code', 'program', 'software', 'development', 'app', 'algorithm', 'function', 'coding', 'computer', 'developer'],
            'mechanical': ['machine', 'engine', 'mechanism', 'repair', 'tool', 'fix', 'maintenance', 'parts', 'assembly', 'component'],
            'artistic': ['art', 'draw', 'paint', 'sketch', 'design', 'creative', 'artistic', 'canvas', 'color', 'composition'],
            'musical': ['music', 'instrument', 'play', 'song', 'melody', 'rhythm', 'note', 'tune', 'chord', 'composition'],
            'sports': ['sport', 'game', 'play', 'technique', 'training', 'practice', 'skill', 'exercise', 'competition', 'athlete'],
            'medical': ['medical', 'health', 'treatment', 'procedure', 'therapy', 'medicine', 'clinical', 'patient', 'healing', 'care'],
            'crafting': ['craft', 'make', 'build', 'create', 'material', 'tool', 'project', 'handmade', 'construction', 'assembly'],
            'language_usage': ['speak', 'write', 'grammar', 'usage', 'sentence', 'paragraph', 'communication', 'expression', 'clarity', 'style'],
            'problem_solving': ['solve', 'solution', 'approach', 'strategy', 'method', 'analysis', 'resolution', 'fix', 'address', 'tackle'],

            # Experiential subcategories
            'travel': ['travel', 'trip', 'journey', 'visit', 'destination', 'tourist', 'vacation', 'explore', 'abroad', 'foreign'],
            'work': ['job', 'career', 'profession', 'workplace', 'office', 'colleagues', 'employment', 'business', 'task', 'project'],
            'social': ['social', 'interaction', 'relationship', 'communication', 'friend', 'group', 'community', 'network', 'gathering', 'party'],
            'personal': ['personal', 'individual', 'private', 'self', 'feeling', 'emotion', 'reaction', 'perception', 'response', 'thought'],
            'educational': ['education', 'learning', 'school', 'study', 'course', 'class', 'teach', 'student', 'knowledge', 'curriculum'],
            'cultural_immersion': ['immersion', 'culture', 'adaptation', 'integration', 'foreign', 'tradition', 'local', 'native', 'customs', 'lifestyle'],
            'volunteer': ['volunteer', 'help', 'assist', 'service', 'charity', 'community', 'contribute', 'donate', 'support', 'nonprofit'],
            'leadership': ['lead', 'manage', 'direct', 'guide', 'supervise', 'team', 'organization', 'responsibility', 'vision', 'decision'],
            'creative': ['create', 'imagine', 'design', 'develop', 'innovate', 'artistic', 'inspiration', 'expression', 'originality', 'unique'],
            'life_lessons': ['lesson', 'learn', 'experience', 'wisdom', 'insight', 'realization', 'growth', 'development', 'understanding', 'maturity'],

            # Tacit subcategories
            'intuition': ['intuition', 'gut', 'instinct', 'feeling', 'sense', 'impression', 'hunch', 'perception', 'awareness', 'insight'],
            'emotional_intelligence': ['emotional', 'empathy', 'awareness', 'sensitivity', 'understanding', 'feeling', 'recognizing', 'responding', 'perceiving', 'managing'],
            'leadership_insights': ['leadership', 'vision', 'inspiration', 'motivation', 'guidance', 'influence', 'direction', 'strategy', 'foresight', 'empowerment'],
            'problem_solving_instincts': ['instinct', 'approach', 'solution', 'resolve', 'tackle', 'address', 'fix', 'handle', 'manage', 'overcome'],
            'decision_making': ['decision', 'choice', 'select', 'determine', 'judge', 'evaluate', 'assess', 'opt', 'pick', 'resolve'],
            'creative_insights': ['creative', 'novel', 'original', 'innovative', 'imaginative', 'unique', 'fresh', 'inventive', 'artistic', 'inspired'],
            'negotiation': ['negotiate', 'bargain', 'compromise', 'agreement', 'deal', 'arrangement', 'settlement', 'mediate', 'discuss', 'resolve'],
            'communication': ['communicate', 'express', 'convey', 'articulate', 'present', 'state', 'transmit', 'share', 'relate', 'explain'],
            'relationship_building': ['relationship', 'connection', 'rapport', 'bond', 'network', 'associate', 'alliance', 'partnership', 'collaboration', 'link'],
            'personal_judgment': ['judgment', 'discernment', 'evaluation', 'assessment', 'critique', 'appraisal', 'estimation', 'opinion', 'view', 'perspective'],

            # Explicit subcategories
            'manuals': ['manual', 'guide', 'handbook', 'instruction', 'reference', 'document', 'tutorial', 'how-to', 'guidebook', 'user guide'],
            'textbooks': ['textbook', 'book', 'text', 'educational', 'academic', 'material', 'course', 'study', 'learning', 'teaching'],
            'research_papers': ['research', 'paper', 'study', 'publication', 'journal', 'article', 'investigation', 'analysis', 'finding', 'report'],
            'tutorials': ['tutorial', 'guide', 'instruction', 'walkthrough', 'lesson', 'demonstration', 'example', 'explanation', 'how-to', 'step-by-step'],
            'instructional_videos': ['video', 'instructional', 'demonstration', 'tutorial', 'guide', 'visual', 'presentation', 'show', 'display', 'exhibit'],
            'policy_documents': ['policy', 'document', 'guideline', 'regulation', 'procedure', 'protocol', 'standard', 'rule', 'directive', 'instruction'],
            'training_materials': ['training', 'material', 'resource', 'content', 'module', 'course', 'program', 'curriculum', 'instruction', 'learn'],
            'academic_articles': ['academic', 'article', 'scholarly', 'publication', 'research', 'journal', 'study', 'analysis', 'paper', 'literature'],
            'software_documentation': ['documentation', 'software', 'manual', 'guide', 'reference', 'API', 'specification', 'technical', 'instruction', 'doc'],
            'legal_contracts': ['contract', 'legal', 'agreement', 'document', 'terms', 'condition', 'clause', 'provision', 'stipulation', 'covenant'],

            # Conceptual subcategories
            'scientific_theories': ['theory', 'scientific', 'hypothesis', 'principle', 'law', 'model', 'framework', 'concept', 'paradigm', 'doctrine'],
            'philosophical_concepts': ['philosophy', 'concept', 'idea', 'notion', 'thought', 'principle', 'theory', 'doctrine', 'school', 'perspective'],
            'mathematical_theories': ['mathematical', 'theorem', 'proof', 'axiom', 'postulate', 'equation', 'formula', 'theory', 'law', 'rule'],
            'economic_theories': ['economic', 'theory', 'principle', 'model', 'concept', 'doctrine', 'hypothesis', 'framework', 'approach', 'school'],
            'sociological_theories': ['sociological', 'society', 'social', 'theory', 'structure', 'system', 'organization', 'framework', 'perspective', 'approach'],
            'psychological_theories': ['psychological', 'psychology', 'theory', 'model', 'concept', 'framework', 'perspective', 'approach', 'school', 'hypothesis'],
            'political_theories': ['political', 'politics', 'theory', 'ideology', 'doctrine', 'philosophy', 'principle', 'system', 'model', 'concept'],
            'ethical_concepts': ['ethical', 'ethics', 'moral', 'principle', 'value', 'virtue', 'standard', 'code', 'norm', 'conduct'],
            'literary_themes': ['literary', 'theme', 'motif', 'message', 'subject', 'meaning', 'concept', 'idea', 'symbol', 'representation'],
            'design_principles': ['design', 'principle', 'rule', 'guideline', 'standard', 'concept', 'theory', 'approach', 'method', 'framework'],

            # Contextual subcategories
            'cultural_norms': ['cultural', 'norm', 'custom', 'tradition', 'practice', 'convention', 'value', 'standard', 'behavior', 'etiquette'],
            'historical_context': ['historical', 'history', 'era', 'period', 'time', 'epoch', 'age', 'background', 'setting', 'circumstance'],
            'economic_environment': ['economic', 'economy', 'market', 'financial', 'monetary', 'fiscal', 'commercial', 'trade', 'business', 'industry'],
            'political_climate': ['political', 'politics', 'government', 'administration', 'regime', 'system', 'policy', 'governance', 'authority', 'power'],
            'social_dynamics': ['social', 'interaction', 'relation', 'dynamic', 'behavior', 'structure', 'pattern', 'network', 'connection', 'exchange'],
            'geographic_influences': ['geographic', 'location', 'region', 'terrain', 'landscape', 'environment', 'climate', 'area', 'place', 'spatial'],
            'technological_landscape': ['technological', 'technology', 'digital', 'innovation', 'advancement', 'development', 'trend', 'system', 'infrastructure', 'ecosystem'],
            'organizational_culture': ['organizational', 'culture', 'corporate', 'workplace', 'environment', 'atmosphere', 'ethos', 'climate', 'values', 'practices'],
            'market_trends': ['market', 'trend', 'pattern', 'direction', 'movement', 'shift', 'change', 'development', 'fluctuation', 'evolution'],
            'legal_framework': ['legal', 'law', 'regulation', 'legislation', 'statute', 'code', 'system', 'structure', 'framework', 'rules']
        }

        return related_terms.get(subcategory, [])

    def get_domain_settings(self, question):
        """
        Get domain-specific settings for a question based on knowledge categories.

        Args:
            question: The question text

        Returns:
            dict: Settings including confidence thresholds, etc.
        """
        domain, confidence, subcategory, subcategory_confidence = self.classify(question)

        # Default settings
        settings = {
            'memory_weight': 0.8,            # How much to weight memory vs. model knowledge
            'confidence_threshold': 0.6,     # Minimum confidence to accept answer
            'domain': domain,                # Detected domain
            'domain_confidence': confidence, # Confidence in domain classification
            'post_process': False,           # Whether to apply domain-specific post-processing
            'retrieval_count': 8             # How many memories to retrieve
        }

        if subcategory:
            settings['subcategory'] = subcategory
            settings['subcategory_confidence'] = subcategory_confidence

        # Adjust settings based on domain
        if domain == 'declarative':
            settings.update({
                'memory_weight': 0.9,        # High weight on memory for facts
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 12
            })

        elif domain == 'procedural_knowledge':
            settings.update({
                'memory_weight': 0.7,
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 10
            })

        elif domain == 'experiential':
            settings.update({
                'memory_weight': 0.8,        # High reliance on memory for personal experiences
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 6
            })

        elif domain == 'tacit':
            settings.update({
                'memory_weight': 0.5,        # Balanced mix of model knowledge and memory
                'confidence_threshold': 0.4,  # Lower threshold due to subjective nature
                'post_process': False,
                'retrieval_count': 5
            })

        elif domain == 'explicit':
            settings.update({
                'memory_weight': 0.9,        # Very high reliance on memory for documented knowledge
                'confidence_threshold': 0.6,
                'post_process': False,
                'retrieval_count': 8
            })

        elif domain == 'conceptual_knowledge':
            settings.update({
                'memory_weight': 0.6,         # Balanced approach
                'confidence_threshold': 0.4,  # Lower threshold for concepts
                'post_process': False,
                'retrieval_count': 8
            })

        elif domain == 'contextual':
            settings.update({
                'memory_weight': 0.7,        # Significant memory weight for contextual information
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 10        # More memories to build context
            })
            
        # If confidence in classification is low, fall back to more balanced settings
        if confidence < 0.4:
            settings.update({
                'memory_weight': 0.5,
                'confidence_threshold': 0.6,
                'retrieval_count': 6
            })
            
        return settings