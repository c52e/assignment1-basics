import os
from typing import BinaryIO
import regex as re
from itertools import takewhile
from collections import Counter
import platform


def find_any(chunk, patterns):
    for pattern in patterns:
        found_at = chunk.find(pattern)
        if found_at != -1:
            return found_at
    return -1

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special tokens as a list of bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = find_any(mini_chunk, split_special_tokens)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge_tokens(tokens, pair, new_token):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == pair[0] and i + 1 < len(tokens) and tokens[i+1] == pair[1]:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, [token.encode("utf-8") for token in special_tokens])
        split_pattern = '|'.join(re.escape(token) for token in special_tokens)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        tokens_array = list(bytes([i]) for i in range(0, 256))
        is_windows = True#platform.system() == "Windows"
        
        words_count = {}
        chunk_with_boundaries = ''
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_with_boundaries += f.read(end - start).decode("utf-8", errors="ignore")
            if is_windows:
                if chunk_with_boundaries.endswith('\r'):
                    continue
                chunk_with_boundaries = chunk_with_boundaries.replace('\r\n', '\n')
            chunks = re.split(split_pattern, chunk_with_boundaries)
            for chunk in chunks:
                words = re.findall(PAT, chunk)
                local_words_count = Counter(words)
                for word, count in local_words_count.items():
                    words_count[word] = words_count.get(word, 0) + count
            chunk_with_boundaries = ''
        
        tokens_count = [(tuple(word.encode('utf-8')), count) for word, count in words_count.items() ]
        #print(tokens_count)

        merges = []
        while len(tokens_array) < vocab_size - len(special_tokens):
            #print(f"Vocabulary size: {len(tokens_array)} / {vocab_size}")
            token_pair_count = {}
            for tokens, count in tokens_count:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    token_pair_count[pair] = token_pair_count.get(pair, 0) + count
            sorted_pairs = sorted(token_pair_count.items(), key=lambda item: item[1], reverse=True)
            max_frequency = sorted_pairs[0][1]
            most_common_pairs = [x[0] for x in takewhile(lambda x: x[1] == max_frequency, sorted_pairs)]
            pair_to_merge = max(most_common_pairs, key=lambda pair: tuple(tokens_array[token] for token in pair)) # Use tuple instead of join with b''
            new_token_bytes = b''.join(tokens_array[token] for token in pair_to_merge)
            # if len(most_common_pairs) > 1:
            #     input(f'{[b''.join(tokens_array[token] for token in pair_to_merge) for pair_to_merge in most_common_pairs]} {new_token_bytes} {max_frequency}  {len(tokens_array)}')
            for i in range(len(tokens_count)):
                tokens_count[i] = (merge_tokens(tokens_count[i][0], pair_to_merge, len(tokens_array)), tokens_count[i][1])
            #print(tokens_count[:10])
            #print(tokens_map)
            merges.append((tokens_array[pair_to_merge[0]], tokens_array[pair_to_merge[1]]))
            tokens_array.append(new_token_bytes)
    tokens_array += [token.encode("utf-8") for token in special_tokens]
    vocab = {i: token for i, token in enumerate(tokens_array)}
    return vocab, merges

if __name__ == "__main__":
    vocab, merges = bpe_tokenizer('./tests/fixtures/corpus.en', 1000, ['<|endoftext|>'])
    print(vocab, merges[:32])

