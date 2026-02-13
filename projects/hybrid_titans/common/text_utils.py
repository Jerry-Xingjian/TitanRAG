"""
Text utilities for document processing.
Shared chunking and text processing functions.
"""

import re


def split_into_chunks(text, min_length=30, sentences_per_chunk=3, overlap_sentences=1):
    """
    Split text into semantic chunks for retrieval with overlapping windows.
    
    Strategy:
    - Headers are grouped with their following sentences
    - Chunks overlap by 1 sentence to avoid boundary information loss
    - Each chunk contains 3 sentences for better context
    
    Args:
        text: Input text (can contain markdown headers)
        min_length: Minimum chunk length to keep
        sentences_per_chunk: Number of sentences per chunk (default: 3)
        overlap_sentences: Number of sentences to overlap between chunks (default: 1)
        
    Returns:
        List of text chunks
    """
    lines = text.split('\n')
    chunks = []
    current_header = ""
    all_sentences = []  # Collect all sentences first
    
    # First pass: collect all sentences with their headers
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a header
        if line.startswith('#'):
            # If we have accumulated sentences, create chunk before new header
            if all_sentences:
                _create_chunks_from_sentences(
                    chunks, current_header, all_sentences, 
                    sentences_per_chunk, overlap_sentences, min_length
                )
                all_sentences = []
            current_header = line
            continue
        
        # Handle regular content - split into sentences
        # Protect abbreviations
        protected = line.replace('Dr.', 'Dr§').replace('Mr.', 'Mr§').replace('Ms.', 'Ms§')
        protected = protected.replace('etc.', 'etc§').replace('e.g.', 'eg§').replace('i.e.', 'ie§')
        
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', protected)
        
        for part in parts:
            part = part.strip()
            # Restore abbreviations
            part = part.replace('Dr§', 'Dr.').replace('Mr§', 'Mr.').replace('Ms§', 'Ms.')
            part = part.replace('etc§', 'etc.').replace('eg§', 'e.g.').replace('ie§', 'i.e.')
            
            if len(part) >= 10:  # Minimum sentence length
                all_sentences.append(part)
    
    # Flush remaining sentences
    if all_sentences or current_header:
        _create_chunks_from_sentences(
            chunks, current_header, all_sentences, 
            sentences_per_chunk, overlap_sentences, min_length
        )
    
    return chunks


def _create_chunks_from_sentences(chunks, header, sentences, 
                                   sentences_per_chunk, overlap_sentences, min_length):
    """Helper: Create overlapping chunks from a list of sentences."""
    if not sentences:
        if header and len(header) > 5:
            chunks.append(header)
        return
    
    # If we have fewer sentences than chunk size, create one chunk
    if len(sentences) <= sentences_per_chunk:
        content = " ".join(sentences)
        if header:
            chunk = f"{header} {content}"
        else:
            chunk = content
        if len(chunk) >= min_length:
            chunks.append(chunk)
        return
    
    # Create overlapping chunks using sliding window
    step = sentences_per_chunk - overlap_sentences
    step = max(1, step)  # Ensure at least 1 step
    
    for i in range(0, len(sentences), step):
        window = sentences[i:i + sentences_per_chunk]
        if not window:
            break
            
        content = " ".join(window)
        
        # Include header only in first chunk of section
        if i == 0 and header:
            chunk = f"{header} {content}"
        else:
            chunk = content
            
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        # Stop if we've processed all sentences
        if i + sentences_per_chunk >= len(sentences):
            break


def simple_split(text):
    """Simple line-by-line split (for comparison)."""
    return [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]


def chunk_context(context):
    """Split a context string into chunks, respecting document boundaries.

    For multi-document contexts (HotpotQA style with '# Title' headers),
    chunks are created within each document to avoid mixing content
    from different source documents in the same chunk.
    """
    doc_sections = re.split(r'\n(?=# )', context)

    if len(doc_sections) > 1:
        # Multi-document: chunk each section independently
        all_chunks = []
        for section in doc_sections:
            section = section.strip()
            if not section:
                continue
            section_chunks = split_into_chunks(section, sentences_per_chunk=3, overlap_sentences=1)
            if section_chunks:
                all_chunks.extend(section_chunks)
            elif len(section) >= 30:
                all_chunks.append(section)
        if all_chunks:
            return all_chunks

    # Single-document or fallback
    chunks = split_into_chunks(context, sentences_per_chunk=2, overlap_sentences=1)
    if not chunks:
        chunks = [p.strip() for p in context.split('\n\n') if p.strip()]
    if not chunks:
        chunks = [context]
    return chunks
