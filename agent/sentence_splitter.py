"""
Splits a stream of text chunks (e.g. agent.llm.respond_stream()'s output)
into complete sentences as soon as each one's terminating punctuation
appears, so a caller can dispatch each sentence to TTS immediately rather
than waiting for the whole reply.

Handles Latin sentence-enders (. ! ?) and the Devanagari danda/double danda
(। ॥), since agent/llm.py's system prompt allows Hindi/Hinglish replies and
Hindi doesn't use Latin punctuation to end a sentence.
"""
import re

# '.' only ends a sentence when not immediately followed by a digit, so
# decimals ("12.50") and similar don't get split mid-number.
SENTENCE_END_RE = re.compile(r"[!?।॥]|\.(?!\d)")

# a fragment with no word characters at all (e.g. a lone "." left over from
# "...") isn't a real sentence -- skip it rather than dispatch it to TTS.
HAS_WORD_RE = re.compile(r"\w", re.UNICODE)


def split_sentences(chunk_stream):
    """chunk_stream: an iterable of text chunks, any granularity (single
    string, word-by-word, character-by-character -- all work identically).
    Yields each complete sentence as soon as it's identified, then yields
    any trailing non-empty fragment once the stream ends (even without
    terminating punctuation), so no text is ever silently dropped.

    A boundary match sitting exactly at the end of the buffer received so
    far is deferred rather than acted on immediately: a real streaming API
    can split "12" / "." / "50" into separate chunks, so a '.' with
    nothing after it *yet* isn't distinguishable from a genuine sentence
    end until either more text arrives (resolving it) or the stream ends
    (also resolving it, via the final flush below)."""
    buffer = ""
    for chunk in chunk_stream:
        buffer += chunk
        while True:
            match = SENTENCE_END_RE.search(buffer)
            if not match or match.end() == len(buffer):
                break
            end = match.end()
            sentence = buffer[:end].strip()
            buffer = buffer[end:]
            if sentence and HAS_WORD_RE.search(sentence):
                yield sentence

    # final flush: no more input is coming, so any remaining match is no
    # longer ambiguous -- process the buffer fully.
    while True:
        match = SENTENCE_END_RE.search(buffer)
        if not match:
            break
        end = match.end()
        sentence = buffer[:end].strip()
        buffer = buffer[end:]
        if sentence and HAS_WORD_RE.search(sentence):
            yield sentence

    remainder = buffer.strip()
    if remainder and HAS_WORD_RE.search(remainder):
        yield remainder
