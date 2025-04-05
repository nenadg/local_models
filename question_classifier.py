import re
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV

class QuestionClassifier:
    """
    A lightweight SVM-based classifier that categorizes questions into domains
    such as math, translation, factual knowledge, etc.
    """
    
    def __init__(self):
        """Initialize the classifier with pre-defined categories and training data"""
        self.categories = {
            'arithmetic': 0,
            'translation': 1,
            'factual': 2,
            'conceptual': 3,
            'procedural': 4,
            'unknown': 5
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
            'arithmetic': [
                r'(\d+\s*[\+\-\*\/÷×]\s*\d+)',
                r'(sum|difference|product|quotient|add|subtract|multiply|divide).*?(\d+).*?(\d+)',
                r'(calculate|compute|evaluate|what is|result of).*?(\d+\s*[\+\-\*\/÷×]\s*\d+)'
            ],
            'translation': [
                r'(translate|say|how to say|how do you say).*?(in|to|from).*?(english|french|spanish|german|italian|russian|chinese|japanese|korean|dutch|portuguese|arabic|hindi|serbian|polish|swedish|greek)',
                r'(.*?)(in|to).*?(english|french|spanish|german|italian|russian|chinese|japanese|korean|dutch|portuguese|arabic|hindi|serbian|polish|swedish|greek)'
            ],
            'transliteration': [
                r'(transliterate|convert|write).*?(latin|cyrillic|script)',
                r'(.*?)(to|in).*?(latin|cyrillic|script)'
            ]
        }
        
    def _initialize_training_data(self):
        """Initialize with predefined training examples"""
        
        # Arithmetic examples
        arithmetic_examples = [
            "what is 5 + 3?",
            "calculate 10 - 7",
            "what's the result of 12 * 4?",
            "divide 100 by 5",
            "what is the sum of 23 and 45?",
            "10 plus 20",
            "subtract 5 from 10",
            "multiply 6 by 9",
            "what is 42 divided by 7",
            "123 + 456",
            "calculate 99 - 33",
            "what's 7 times 8"
        ]
        
        # Translation examples
        translation_examples = [
            "how do you say hello in french?",
            "translate book to spanish",
            "how to say thank you in japanese",
            "what is the german word for water?",
            "translate apple from english to russian",
            "how do you say goodbye in italian?",
            "say house in chinese",
            "what's the dutch translation of paper?",
            "how to say car in polish",
            "translate device from english to serbian",
            "how do I say good morning in arabic?"
        ]
        
        # Factual knowledge examples
        factual_examples = [
            "who was the first president of the united states?",
            "what is the capital of france?",
            "when did world war 2 end?",
            "how tall is mount everest?",
            "what is the population of tokyo?",
            "who wrote hamlet?",
            "what's the boiling point of water?",
            "which planet is closest to the sun?",
            "what is the currency of japan?",
            "who invented the telephone?"
        ]
        
        # Conceptual examples
        conceptual_examples = [
            "what is quantum mechanics?",
            "explain photosynthesis",
            "how does gravity work?",
            "what is machine learning?",
            "explain the theory of relativity",
            "what is democracy?",
            "how does the internet work?",
            "what is blockchain technology?",
            "explain the concept of supply and demand",
            "what is ethical philosophy?"
        ]
        
        # Procedural examples
        procedural_examples = [
            "how do i bake a cake?",
            "steps to change a tire",
            "how to install python",
            "what's the process for making coffee?",
            "how do you solve a rubik's cube?",
            "steps to create a website",
            "how to grow tomatoes",
            "what's the procedure for cpr?",
            "how do i reset my password?",
            "how to train a dog"
        ]
        
        # Add all examples to training data
        for examples, category in [
            (arithmetic_examples, 'arithmetic'),
            (translation_examples, 'translation'),
            (factual_examples, 'factual'),
            (conceptual_examples, 'conceptual'),
            (procedural_examples, 'procedural')
        ]:
            for example in examples:
                self.training_data.append(example)
                self.training_labels.append(self.categories[category])
                
                # Add variations with different question forms
                variations = self._generate_variations(example)
                for variation in variations:
                    self.training_data.append(variation)
                    self.training_labels.append(self.categories[category])
    
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
        Classify a question into one of the predefined domains.
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
                    return category, 0.95  # High confidence for pattern matches
        
        # If no strong pattern match, use the trained classifier
        try:
            # Get prediction and probability
            category_id = self.clf.predict([question])[0]
            probabilities = self.clf.predict_proba([question])[0]
            confidence = probabilities[category_id]
            
            # Map back to category name
            category_name = next(cat for cat, idx in self.categories.items() if idx == category_id)
            
            return category_name, confidence
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown", 0.0
            
    def get_domain_settings(self, question):
        """
        Get domain-specific settings for a question.
        
        Args:
            question: The question text
            
        Returns:
            dict: Settings including sharpening parameters, confidence thresholds, etc.
        """
        domain, confidence = self.classify(question)
        
        # Default settings
        settings = {
            'memory_weight': 0.5,            # How much to weight memory vs. model knowledge
            'sharpening_factor': 0.3,        # Default sharpening factor
            'confidence_threshold': 0.6,     # Minimum confidence to accept answer
            'domain': domain,                # Detected domain
            'domain_confidence': confidence, # Confidence in domain classification
            'post_process': False,           # Whether to apply domain-specific post-processing
            'retrieval_count': 8             # How many memories to retrieve
        }
        
        # Adjust settings based on domain
        if domain == 'arithmetic':
            settings.update({
                'memory_weight': 0.2,        # Rely more on model than memory
                'sharpening_factor': 0.1,    # Minimal sharpening
                'confidence_threshold': 0.7,
                'post_process': True,        # Apply arithmetic verification
                'retrieval_count': 4         # Fewer memories needed
            })
            
        elif domain == 'translation':
            settings.update({
                'memory_weight': 0.7,        # More weight on memory for translations
                'sharpening_factor': 0.4,    # More aggressive sharpening
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 10        # More memories for better context
            })
            
        elif domain == 'factual':
            settings.update({
                'memory_weight': 0.8,        # High weight on memory for facts
                'sharpening_factor': 0.5,    # Strong sharpening
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 12
            })
            
        elif domain == 'conceptual':
            settings.update({
                'memory_weight': 0.4,        # Balanced approach
                'sharpening_factor': 0.3,    # Moderate sharpening
                'confidence_threshold': 0.4,  # Lower threshold for concepts
                'post_process': False,
                'retrieval_count': 8
            })
            
        elif domain == 'procedural':
            settings.update({
                'memory_weight': 0.6,
                'sharpening_factor': 0.3,
                'confidence_threshold': 0.5,
                'post_process': False,
                'retrieval_count': 10
            })
            
        # If confidence in classification is low, fall back to more balanced settings
        if confidence < 0.4:
            settings.update({
                'memory_weight': 0.5,
                'sharpening_factor': 0.2,
                'confidence_threshold': 0.6,
                'retrieval_count': 6
            })
            
        return settings