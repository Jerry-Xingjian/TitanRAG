import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants based on the saved model's meta
INPUT_DIM = 405  # Derived from model training: 384(emb) + 14(num) + 7(type)
HIDDEN_DIM = 256

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc5 = nn.Linear(hidden_dim // 4, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x1 = F.gelu(self.bn1(self.fc1(x)))
        x2 = F.gelu(self.bn2(self.fc2(x1))) + x1  # Residual
        x3 = F.gelu(self.bn3(self.fc3(x2)))
        x4 = F.gelu(self.bn4(self.fc4(x3)))
        x4 = self.dropout(x4)
        out = self.fc5(x4)
        return out.squeeze(-1)


class BaseConfidenceScorer:
    """Base class with shared feature extraction logic for RAG confidence scoring.
    
    Subclasses only need to implement model loading and the final prediction step.
    """
    
    def __init__(self, embedder_name="all-MiniLM-L6-v2", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedder_name)
        self.model_loaded = False
        
        self.lb = LabelBinarizer()
        self.lb.fit(["what", "who", "how_many", "which", "why", "when", "other"])
    
    def _resolve_model_path(self, model_path, filename):
        """Resolve model path from multiple possible locations."""
        possible_paths = [
            model_path,
            os.path.join(os.getcwd(), model_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', filename),
        ]
        
        # Notebook context fallback
        notebook_dir = os.getcwd()
        if os.path.basename(notebook_dir) == 'notebooks':
            possible_paths.append(os.path.join(os.path.dirname(notebook_dir), model_path))
        
        for p in possible_paths:
            if os.path.exists(p):
                return p, possible_paths
        return None, possible_paths
    
    def extract_question_type(self, qs):
        """Classify question type and return one-hot encoding."""
        types = []
        for q in qs:
            q = q.strip()
            if q.startswith("什么"):
                types.append("what")
            elif q.startswith("谁"):
                types.append("who")
            elif q.startswith("多少") or q.startswith("几"):
                types.append("how_many")
            elif q.startswith("哪"):
                types.append("which")
            elif q.startswith("为何") or q.startswith("为什么"):
                types.append("why")
            elif q.startswith("何时") or q.startswith("什么时候"):
                types.append("when")
            else:
                types.append("other")
        return self.lb.transform(types)
    
    def _extract_features(self, questions, chunks):
        """Extract feature matrix from (question, chunk) pairs.
        
        Returns:
            np.ndarray: Feature matrix of shape (N, feature_dim).
        """
        if len(questions) != len(chunks):
            raise ValueError("Number of questions and chunks must match.")
        
        # 1. Embeddings
        s_embs = self.embedder.encode(chunks, convert_to_tensor=True, device=self.device).cpu().numpy()
        q_embs = self.embedder.encode(questions, convert_to_tensor=True, device=self.device).cpu().numpy()
        
        # 2. String Lengths
        q_lens = np.array([len(q) for q in questions]).reshape(-1, 1)
        # Dummy gold length (not available at inference time)
        gold_lens = np.ones((len(questions), 1)) * 10
        
        # 3. TF-IDF / BM25
        try:
            vectorizer = TfidfVectorizer().fit(chunks + questions)
            text_tfidf = vectorizer.transform(chunks)
            q_tfidf = vectorizer.transform(questions)
            
            scores = (q_tfidf * text_tfidf.T).toarray()
            bm25_diag = np.diag(scores).reshape(-1, 1)
            bm25_rank = np.argsort(-scores, axis=1)[:, 0].reshape(-1, 1)
            
            sims = cosine_similarity(q_tfidf, text_tfidf)
            tfidf_cos = np.diag(sims).reshape(-1, 1)
        except ValueError:
            n = len(questions)
            bm25_diag = np.zeros((n, 1))
            bm25_rank = np.zeros((n, 1))
            tfidf_cos = np.zeros((n, 1))
        
        # 4. Neural Embedding distances
        def safe_norm(v):
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        
        cos_sim = np.sum(safe_norm(q_embs) * safe_norm(s_embs), axis=1, keepdims=True)
        eu_dist = np.linalg.norm(q_embs - s_embs, axis=1, keepdims=True)
        man_dist = np.sum(np.abs(q_embs - s_embs), axis=1, keepdims=True)
        
        # Jaccard
        def jaccard(a, b):
            sa, sb = set(a.split()), set(b.split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)
        jaccard_sim = np.array([jaccard(q, s) for q, s in zip(questions, chunks)]).reshape(-1, 1)
        
        # 5. Question Type
        q_type_oh = self.extract_question_type(questions)
        
        # Combine numerical features
        num_feats = np.concatenate([
            q_lens, gold_lens, bm25_diag, bm25_rank, cos_sim, eu_dist, man_dist, jaccard_sim, tfidf_cos
        ], axis=1)
        
        # Standardize numerical features
        mean = np.mean(num_feats, axis=0)
        std = np.std(num_feats, axis=0) + 1e-8
        num_feats = (num_feats - mean) / std
        
        features = np.concatenate([s_embs, num_feats, q_type_oh], axis=1)
        return features
    
    def predict_batch(self, questions, chunks):
        """Predict confidence scores. Must be implemented by subclass."""
        raise NotImplementedError
    
    def predict_single(self, question, chunk):
        """Predict confidence score for a single (question, chunk) pair."""
        return self.predict_batch([question], [chunk])[0]


class RAGConfidenceScorer(BaseConfidenceScorer):
    """MLP-based RAG confidence scorer using train_test_model.pt."""
    
    def __init__(self, model_path="is_RAG_able/models/train_test_model.pt",
                 embedder_name="all-MiniLM-L6-v2", device=None):
        super().__init__(embedder_name, device)
        
        final_path, possible_paths = self._resolve_model_path(model_path, 'train_test_model.pt')
        
        # Determine input_dim from meta or state_dict
        input_dim = INPUT_DIM
        state_dict_cache = None
        
        if final_path:
            meta_path = final_path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    input_dim = meta.get("input_dim", INPUT_DIM)
            else:
                state_dict_cache = torch.load(final_path, map_location=self.device)
                if 'fc1.weight' in state_dict_cache:
                    input_dim = state_dict_cache['fc1.weight'].shape[1]
        
        self.model = MLPClassifier(input_dim=input_dim)
        
        if final_path:
            if state_dict_cache is None:
                state_dict_cache = torch.load(final_path, map_location=self.device)
            self.model.load_state_dict(state_dict_cache)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ RAGConfidenceScorer loaded from {final_path} (input_dim={input_dim})")
        else:
            print(f"⚠️ Warning: RAG model not found. Checked paths: {possible_paths}. Will return default scores.")
    
    @torch.no_grad()
    def predict_batch(self, questions, chunks):
        """Predict confidence scores using MLP model.
        
        Returns:
            np.ndarray: Array of probabilities in [0, 1].
        """
        if not self.model_loaded:
            return np.ones(len(questions)) * 0.5
        
        features = self._extract_features(questions, chunks)
        
        # Align dimension to model's expected input
        input_dim = self.model.fc1.in_features
        if features.shape[1] < input_dim:
            pad = np.zeros((features.shape[0], input_dim - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        elif features.shape[1] > input_dim:
            features = features[:, :input_dim]
        
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        logits = self.model(features_t)
        probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()


class XGBConfidenceScorer(BaseConfidenceScorer):
    """XGBoost-based RAG confidence scorer using xgb_model.pkl."""
    
    def __init__(self, model_path="is_RAG_able/models/xgb_model.pkl",
                 embedder_name="all-MiniLM-L6-v2", device=None):
        super().__init__(embedder_name, device)
        import joblib
        
        final_path, possible_paths = self._resolve_model_path(model_path, 'xgb_model.pkl')
        
        if final_path:
            self.model = joblib.load(final_path)
            self.model_loaded = True
            print(f"✅ XGBConfidenceScorer loaded from {final_path}")
        else:
            print(f"⚠️ Warning: XGB model not found. Checked paths: {possible_paths}")
            self.model = None
    
    def predict_batch(self, questions, chunks):
        """Predict confidence scores using XGBoost model.
        
        Returns:
            np.ndarray: Array of probabilities in [0, 1] for "Retrievable" class.
        """
        if not self.model_loaded:
            return np.ones(len(questions)) * 0.5
        
        features = self._extract_features(questions, chunks)
        
        # Align feature dimension to match trained model
        expected_dim = self.model.n_features_in_
        if features.shape[1] < expected_dim:
            pad = np.zeros((features.shape[0], expected_dim - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        elif features.shape[1] > expected_dim:
            features = features[:, :expected_dim]
        
        # XGBoost predict_proba → probability of "Retrievable" (class 1)
        probs = self.model.predict_proba(features)[:, 1]
        
        return probs
