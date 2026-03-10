"""
LLM utilities for text generation.
Shared Flan-T5 loading and generation functions.
"""

import torch


class FlanT5Generator:
    """Flan-T5 based text generator."""
    
    def __init__(self, model_name="google/flan-t5-xl", device=None):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.llm = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if self.device:
            self.llm = self.llm.to(self.device)
            
        self.model_name = model_name
        
        device_info = f" on {self.device}" if self.device else ""
        print(f"✅ Loaded LLM: {model_name}{device_info}")
    
    def generate(self, prompt, max_new_tokens=100, num_beams=4):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width
        
        Returns:
            str: Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # Move inputs to the same device as the model
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if str(self.device).startswith('xla'):
            # Force greedy search on TPU to avoid beam_search hangs (torch.isin issues)
            num_beams = 1
            
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=(num_beams > 1),
                do_sample=False,
                no_repeat_ngram_size=2 if num_beams > 1 else 0
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def generate_qa(self, question, context, max_new_tokens=100, multihop=False):
        """
        Generate answer for a question given context.
        
        Args:
            question: The question
            context: Context/passage to answer from
            max_new_tokens: Maximum tokens to generate
            multihop: If True, use chain-of-thought multi-hop reasoning prompt.
                      If False, use extractive prompt (for SQuAD).
        
        Returns:
            tuple: (answer, prompt)
        """
        if multihop:
            prompt = f"""Answer the question using the context below. The answer may require connecting facts from different paragraphs.
Extract the specific name, number, date, or entity asked for. Answer in a few words only.

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Answer the question based on the context provided.
Extract specific details: exact numbers, names, dates, and technical terms.
Provide the most relevant answer from the context, even if partial.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.generate(prompt, max_new_tokens)
        return answer, prompt

    def extract_bridge_entity(self, question, context, max_new_tokens=50):
        """
        Extract the key intermediate entity from context for iterative retrieval.
        
        For multi-hop questions like "What company merged with the CEMM developer?",
        this extracts the bridge entity (e.g., "Compaq") so we can do a second
        retrieval round with richer context.
        
        Args:
            question: The original multi-hop question
            context: Context retrieved in the first round
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Extracted bridge entity or clue, empty string if extraction fails
        """
        prompt = f"""Read the context and identify the key entity or fact needed to answer the question.
Do NOT answer the question. Instead, extract the most important intermediate fact or entity name.

Context:
{context}

Question: {question}

Key entity or fact:"""
        
        entity = self.generate(prompt, max_new_tokens)
        # Clean up: take only the first line/sentence
        entity = entity.strip().split('\n')[0].strip().rstrip('.')
        return entity

