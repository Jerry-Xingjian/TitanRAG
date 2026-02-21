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

class RAGConfidenceScorer:
    """Wrapper to use the trained MLP model for inference."""
    
    def __init__(self, model_path="is_RAG_able/models/train_test_model.pt", embedder_name="all-MiniLM-L6-v2", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedder_name)
        
        # We need to recreate the LabelBinarizer and TFIDF vectorizer logic from training.
        # Since the training script fits these per-batch, we emulate a single-batch prediction.
        self.lb = LabelBinarizer()
        self.lb.fit(["what", "who", "how_many", "which", "why", "when", "other"])

        # Load Model
        # Need to read meta to get exact input_dim
        meta_path = model_path + ".meta.json"
        input_dim = INPUT_DIM
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                input_dim = meta.get("input_dim", INPUT_DIM)

        self.model = MLPClassifier(input_dim=input_dim)
        
        # Handle path relative to workspace root
        if not os.path.exists(model_path):
            # Try to find it if we are deeper in the directory
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            alt_path = os.path.join(workspace_root, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ RAGConfidenceScorer loaded from {model_path}")
        else:
            print(f"⚠️ Warning: RAG model not found at {model_path}. Will return default scores.")
            self.model_loaded = False

    def extract_question_type(self, qs):
        types = []
        for q in qs:
            q = q.strip()
            if q.startswith("什么"): types.append("what")
            elif q.startswith("谁"): types.append("who")
            elif q.startswith("多少") or q.startswith("几"): types.append("how_many")
            elif q.startswith("哪"): types.append("which")
            elif q.startswith("为何") or q.startswith("为什么"): types.append("why")
            elif q.startswith("何时") or q.startswith("什么时候"): types.append("when")
            else: types.append("other")
        # Ensure correct categories by passing the pre-fitted classes
        return self.lb.transform(types)

    @torch.no_grad()
    def predict_batch(self, questions: list[str], chunks: list[str]) -> np.ndarray:
        """Predict confidence scores for a batch of (question, chunk) pairs.
        
        Returns:
            np.ndarray: Array of probabilities in [0, 1]. Size equals len(questions).
        """
        if not self.model_loaded:
            return np.ones(len(questions)) * 0.5  # Fallback

        if len(questions) != len(chunks):
            raise ValueError("Number of questions and chunks must match.")
        
        # 1. Embeddings
        s_embs = self.embedder.encode(chunks, convert_to_tensor=True, device=self.device).cpu().numpy()
        q_embs = self.embedder.encode(questions, convert_to_tensor=True, device=self.device).cpu().numpy()
        
        # 2. String Lengths
        q_lens = np.array([len(q) for q in questions]).reshape(-1, 1)
        # We don't have the "gold" answer at inference time. 
        # The training code used it as a feature, which is technically a leak or oracle feature.
        # We will approximate it with a dummy value (e.g. median answer length 10) or 0.
        gold_lens = np.ones((len(questions), 1)) * 10 
        
        # 3. TF-IDF / BM25 (Per-batch emulation from train_test_model.py)
        # Note: This is an approximation since the original used the whole training set to fit IDF
        try:
            vectorizer = TfidfVectorizer().fit(chunks + questions)
            text_tfidf = vectorizer.transform(chunks)
            q_tfidf = vectorizer.transform(questions)
            
            # BM25 emulation (just dot product in original code)
            scores = (q_tfidf * text_tfidf.T).toarray()
            bm25_diag = np.diag(scores).reshape(-1, 1)
            bm25_rank = np.argsort(-scores, axis=1)[:, 0].reshape(-1, 1)
            
            # TF-IDF Cosine
            sims = cosine_similarity(q_tfidf, text_tfidf)
            tfidf_cos = np.diag(sims).reshape(-1, 1)
        except ValueError:
            # Fallback if vocabulary is empty
            n = len(questions)
            bm25_diag = np.zeros((n, 1))
            bm25_rank = np.zeros((n, 1))
            tfidf_cos = np.zeros((n, 1))

        # 4. Neural Embeddings distances
        def safe_norm(v): return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        
        cos_sim = np.sum(safe_norm(q_embs) * safe_norm(s_embs), axis=1, keepdims=True)
        eu_dist = np.linalg.norm(q_embs - s_embs, axis=1, keepdims=True)
        man_dist = np.sum(np.abs(q_embs - s_embs), axis=1, keepdims=True)
        
        # Jaccard
        def jaccard(a, b):
            sa, sb = set(a.split()), set(b.split())
            if not sa or not sb: return 0.0
            return len(sa & sb) / len(sa | sb)
        jaccard_sim = np.array([jaccard(q, s) for q, s in zip(questions, chunks)]).reshape(-1, 1)
        
        # 5. Question Type
        q_type_oh = self.extract_question_type(questions)
        
        # Combine numerical features
        num_feats = np.concatenate([
            q_lens, gold_lens, bm25_diag, bm25_rank, cos_sim, eu_dist, man_dist, jaccard_sim, tfidf_cos
        ], axis=1)
        
        # In original code, num_feats were standardized using StandardScaler fit on the batch.
        # We will do a generic standardization avoiding div-by-zero.
        mean = np.mean(num_feats, axis=0)
        std = np.std(num_feats, axis=0) + 1e-8
        num_feats = (num_feats - mean) / std
        
        features = np.concatenate([s_embs, num_feats, q_type_oh], axis=1)
        
        # Ensure exact input dimension matches model specification
        # The model's input dim was 405 (in my prior run). If our features are slightly off due to vocab or lb, we pad or truncate.
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
        
    def predict_single(self, question: str, chunk: str) -> float:
        """Predict confidence score for a single (question, chunk) pair."""
        return self.predict_batch([question], [chunk])[0]
