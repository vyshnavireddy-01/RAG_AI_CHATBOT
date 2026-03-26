import re


def chunk_text(text, max_sentences=6, overlap_sentences=2):
    """
    Split text into chunks by SENTENCE boundaries — never mid-word or mid-sentence.

    max_sentences  : how many sentences per chunk (tune up if context feels thin)
    overlap_sentences: how many sentences from previous chunk to carry into next
                       (keeps context continuous across chunk boundaries)
    """

    if not text or not text.strip():
        return []

    # Collapse whitespace
    text = " ".join(text.split())

    # Split into sentences on . ! ? followed by a space and capital letter
    # Also split on newlines that act as sentence separators
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)', text)

    # Clean and filter empty sentences
    sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 10]

    if not sentences:
        return [text.strip()]

    chunks = []
    i = 0

    while i < len(sentences):
        # Take up to max_sentences sentences for this chunk
        chunk_sentences = sentences[i: i + max_sentences]
        chunk = " ".join(chunk_sentences).strip()

        if chunk:
            chunks.append(chunk)

        # Move forward, but step back by overlap_sentences for continuity
        i += max_sentences - overlap_sentences

        # Safety: avoid infinite loop if overlap >= max
        if max_sentences <= overlap_sentences:
            i += 1

    return chunks