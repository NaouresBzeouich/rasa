import logging
from typing import Any, Dict, List, Optional, Text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import TEXT, INTENT
import gc
import os
import re
import difflib

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class Phi3IntentClassifier(GraphComponent, IntentClassifier):
    """Intent classifier using Phi-3 Mini with improved French language support."""

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        
        # Configuration
        self.model_name = config.get("model_name", "microsoft/Phi-3-mini-4k-instruct")
        self.use_quantization = config.get("use_quantization", True)
        self.cache_dir = config.get("cache_dir", "./model_cache")
        
        # Runtime variables
        self.tokenizer = None
        self.model = None
        self.intent_examples = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized Phi3IntentClassifier on {self.device}")

    @classmethod
    def create(cls, config, model_storage, resource, execution_context):
        return cls(config, model_storage, resource, execution_context)

    def load_model(self):
        """Load Phi-3 model with optimized setup."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading {self.model_name}...")
        
        try:
            # Optimized quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if self.use_quantization and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager"
                )
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                    attn_implementation="eager"
                ).to(self.device)
            
            self.model.eval()
            logger.info("✓ Phi-3 model loaded successfully!")
                
        except Exception as e:
            logger.error(f"Failed to load Phi-3 model: {e}")
            raise

    def train(self, training_data: TrainingData) -> Resource:
        """Store training examples for few-shot prompting."""
        self.intent_examples = {}
        
        # Debug: Check if we have training data
        logger.info(f"Training with {len(training_data.intent_examples)} examples")
        
        for example in training_data.intent_examples:
            intent = example.get(INTENT)
            text = example.get(TEXT)
            
            if intent and text:  # Make sure both exist
                if intent not in self.intent_examples:
                    self.intent_examples[intent] = []
                
                # Store up to 5 examples per intent (increased for better context)
                if len(self.intent_examples[intent]) < 5:
                    self.intent_examples[intent].append(text)
                    logger.debug(f"Added example for {intent}: {text}")
        
        logger.info(f"Stored examples for {len(self.intent_examples)} intents: {list(self.intent_examples.keys())}")
        
        # Store the training data persistently
        with self._model_storage.write_to(self._resource) as model_dir:
            import pickle
            with open(model_dir / "intent_examples.pkl", "wb") as f:
                pickle.dump(self.intent_examples, f)
        
        return self._resource

    def create_phi3_prompt(self, user_message: str) -> str:
        """Create an improved French-optimized prompt."""
        
        # Create examples section with clear formatting
        examples_text = ""
        for intent, examples in self.intent_examples.items():
            examples_text += f"\n{intent}:\n"
            for example in examples[:2]:  # Use only 2 examples to save space
                examples_text += f"  - {example}\n"
        
        # Simplified, clearer prompt in French
        prompt = f"""<|system|>
Tu es un classificateur d'intentions en français. Tu dois répondre UNIQUEMENT avec le nom de l'intention, rien d'autre.

Intentions disponibles avec exemples:{examples_text}

Règles:
- Réponds SEULEMENT avec le nom de l'intention
- Pas d'explication, pas de phrase complète
- Si tu n'es pas sûr, utilise "nlu_fallback"<|end|>
<|user|>
Classe cette phrase: "{user_message}"

Intention:<|end|>
<|assistant|>
"""
        return prompt

    def process(self, messages: List[Message]) -> List[Message]:
        """Process messages with training examples prioritized over keywords."""
        
        for message in messages:
            user_text = message.get(TEXT)
            if not user_text:
                continue
            
            try:
                intent = None
                
                # Step 1: Try training examples first (if we have them)
                if self.intent_examples:
                    logger.info("Trying training examples first...")
                    intent = self._training_based_matching(user_text)
                    
                    # If training examples give good confidence, use it
                    if intent["confidence"] >= 0.7:
                        logger.info(f"Training examples gave good match: {intent}")
                        message.set("intent", intent, add_to_output=True)
                        logger.info(f"'{user_text}' -> {intent['name']} ({intent['confidence']:.3f})")
                        continue
                
                # Step 2: If no good training match, try Phi-3
                if not intent or intent["confidence"] < 0.7:
                    logger.info("Training examples confidence too low, trying Phi-3...")
                    self.load_model()  # Only load model when needed
                    phi3_intent = self._classify_with_phi3(user_text)
                    
                    if not intent or phi3_intent["confidence"] > intent["confidence"]:
                        intent = phi3_intent
                
                # Step 3: Finally, try keyword matching as last resort
                if intent["confidence"] < 0.6:
                    logger.info("Phi-3 confidence too low, trying keyword matching...")
                    keyword_intent = self._simple_keyword_matching(user_text.lower().strip())
                    
                    if keyword_intent["confidence"] > intent["confidence"]:
                        intent = keyword_intent
                
                message.set("intent", intent, add_to_output=True)
                logger.info(f"'{user_text}' -> {intent['name']} ({intent['confidence']:.3f})")
                
            except Exception as e:
                logger.error(f"Classification error: {e}")
                intent = {"name": "nlu_fallback", "confidence": 0.1}
                message.set("intent", intent, add_to_output=True)
        
        return messages

    def _training_based_matching(self, text: str) -> Dict[str, Any]:
        """Match against training examples using multiple similarity metrics."""
        text_lower = text.lower().strip()
        
        logger.info(f"Training-based matching for: '{text_lower}'")
        logger.info(f"Available intents: {list(self.intent_examples.keys())}")
        
        best_match = None
        best_score = 0.0
        
        for intent, examples in self.intent_examples.items():
            intent_max_score = 0.0
            best_example = None
            
            for example in examples:
                example_lower = example.lower()
                
                # Calculate multiple similarity scores
                jaccard_score = self._jaccard_similarity(text_lower, example_lower)
                substring_score = self._substring_similarity(text_lower, example_lower)
                token_score = self._token_overlap_score(text_lower, example_lower)
                
                # Weighted combination of scores
                combined_score = (
                    jaccard_score * 0.4 +
                    substring_score * 0.3 +
                    token_score * 0.3
                )
                
                if combined_score > intent_max_score:
                    intent_max_score = combined_score
                    best_example = example_lower
            
            if intent_max_score > best_score:
                best_score = intent_max_score
                best_match = intent
                logger.info(f"Training match: '{text_lower}' ~ '{best_example}' -> {intent} (score: {best_score:.3f})")
        
        if best_match and best_score > 0.3:  # Lower threshold for training examples
            return {"name": best_match, "confidence": min(best_score + 0.2, 0.95)}  # Boost confidence for training matches
        
        return {"name": "nlu_fallback", "confidence": 0.1}
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _substring_similarity(self, text1: str, text2: str) -> float:
        """Calculate substring similarity."""
        if text1 in text2 or text2 in text1:
            return 0.8
        
        # Check for partial substrings
        words1 = text1.split()
        words2 = text2.split()
        
        matches = 0
        for word1 in words1:
            for word2 in words2:
                if word1 in word2 or word2 in word1:
                    matches += 1
                    break
        
        if not words1:
            return 0.0
        
        return matches / len(words1)
    
    def _token_overlap_score(self, text1: str, text2: str) -> float:
        """Calculate token overlap score with position awareness."""
        tokens1 = text1.split()
        tokens2 = text2.split()
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Exact token matches
        exact_matches = sum(1 for token in tokens1 if token in tokens2)
        
        # Partial token matches
        partial_matches = 0
        for token1 in tokens1:
            for token2 in tokens2:
                if token1 != token2 and (token1 in token2 or token2 in token1):
                    partial_matches += 1
                    break
        
        total_score = exact_matches + (partial_matches * 0.5)
        return total_score / max(len(tokens1), len(tokens2))
    
    def _simple_keyword_matching(self, text: str) -> Dict[str, Any]:
        """Simple keyword matching as fallback."""
        logger.info(f"Keyword matching for: '{text}'")
        
        # Simple keyword mapping - match exact words first
        keyword_map = {
            "greet": ["salut", "bonjour", "coucou", "hey", "heyy", "bonsoir", "hello", "hi"],
            "goodbye": ["au revoir", "bye", "à bientôt", "ciao", "adieu"],
            "mood_happy": ["bien", "super", "génial", "parfait", "excellent", "content", "heureux"],
            "mood_sad": ["mal", "triste", "déprimé", "abattu", "démoralisé"],
            "engagement": ["engagement", "engagé"],
            "absence": ["absent", "absence", "congé", "vacances"],
            "absence_duration": ["durée", "combien", "temps"],
            "turnover_rate": ["rotation", "turnover", "quitter", "partir"],
            "team_turnover": ["équipe"],
            "generate_report": ["rapport", "générer", "créer"]
        }
        
        # Check for direct word matches
        best_match = None
        best_score = 0.0
        
        for intent, keywords in keyword_map.items():
            for keyword in keywords:
                # Check if keyword exists in text
                if keyword in text:
                    # Calculate confidence based on match quality
                    if keyword == text:  # Exact match
                        score = 0.85
                    elif text.startswith(keyword) or text.endswith(keyword):
                        score = 0.75
                    else:  # Substring match
                        score = 0.65
                    
                    if score > best_score:
                        best_score = score
                        best_match = intent
                        logger.info(f"Found keyword match: '{keyword}' -> {intent} (score: {score})")
        
        # Also check phrase patterns
        phrase_patterns = {
            "greet": ["comment ça va", "comment ca va", "ça va", "ca va"],
            "absence": ["sera absent", "est absent", "prendre congé"],
            "absence_duration": ["combien dure", "quelle durée", "pendant combien", "durée", "duree"],
            "engagement": ["score engagement", "taux engagement"]
        }
        
        for intent, phrases in phrase_patterns.items():
            for phrase in phrases:
                if phrase in text:
                    score = 0.70
                    if score > best_score:
                        best_score = score
                        best_match = intent
                        logger.info(f"Found phrase match: '{phrase}' -> {intent}")
        
        if best_match and best_score > 0.5:
            logger.info(f"Keyword matching result: {best_match} (confidence: {best_score})")
            return {"name": best_match, "confidence": best_score}
        
        logger.info("No keyword match found")
        return {"name": "nlu_fallback", "confidence": 0.1}

    def _classify_with_phi3(self, user_text: str) -> Dict[str, Any]:
        """Classify using Phi-3 with improved parsing."""
        prompt = self.create_phi3_prompt(user_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=800,  # Reduced for stability
            padding=False
        )
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        # Generate with conservative settings
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=10,
                do_sample=False,   # Deterministic generation (removes temperature warning)
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.0
            )
        
        # Decode and clean response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Extract intent name with better parsing
        predicted_intent = self._extract_intent_from_response(response)
        
        # Validate and return
        return self._validate_intent(predicted_intent, user_text)

    def _extract_intent_from_response(self, response: str) -> str:
        """Extract intent name from model response with better parsing."""
        # Clean the response
        response = response.strip().lower()
        
        # Remove common prefixes/suffixes
        response = re.sub(r'^(the |l\'|la |le |les |intent:?|intention:?)', '', response)
        response = re.sub(r'(intent|intention)$', '', response)
        
        # Take first line/word
        lines = response.split('\n')
        first_line = lines[0].strip()
        
        # Extract the first word/phrase that could be an intent
        words = first_line.split()
        if words:
            candidate = words[0].strip('.,!?":')
            return candidate
        
        return response

    def _validate_intent(self, predicted_intent: str, original_text: str) -> Dict[str, Any]:
        """Validate predicted intent with fuzzy matching."""
        
        # Direct match
        if predicted_intent in self.intent_examples:
            return {"name": predicted_intent, "confidence": 0.90}
        
        # Fuzzy matching with difflib
        available_intents = list(self.intent_examples.keys())
        close_matches = difflib.get_close_matches(
            predicted_intent, 
            available_intents, 
            n=1, 
            cutoff=0.6
        )
        
        if close_matches:
            return {"name": close_matches[0], "confidence": 0.85}
        
        # Substring matching
        for intent in available_intents:
            if predicted_intent in intent or intent in predicted_intent:
                return {"name": intent, "confidence": 0.75}
        
        return {"name": "nlu_fallback", "confidence": 0.1}

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validate component configuration."""
        pass

    def persist(self) -> None:
        """Clean up GPU memory."""
        if self.device == "cuda":
            if self.model is not None:
                del self.model
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleaned up")

    @classmethod
    def load(cls, config, model_storage, resource, execution_context, **kwargs):
        """Load the trained component."""
        instance = cls(config, model_storage, resource, execution_context)
        
        # Try to load stored training examples
        try:
            with model_storage.read_from(resource) as model_dir:
                import pickle
                with open(model_dir / "intent_examples.pkl", "rb") as f:
                    instance.intent_examples = pickle.load(f)
                    logger.info(f"Loaded examples for {len(instance.intent_examples)} intents: {list(instance.intent_examples.keys())}")
        except Exception as e:
            logger.warning(f"Could not load training examples: {e}")
            instance.intent_examples = {}
        
        return instance