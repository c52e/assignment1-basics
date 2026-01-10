import os
from typing import BinaryIO
import regex as re
from itertools import takewhile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import heapq

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

def detect_newline(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        f.readline()
        return f.newlines

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def process_file_chunk(file_path, start, end, split_pattern, is_windows):
    local_words_count = Counter()
    
    with open(file_path, 'rb') as f:
        f.seek(start)
        content = f.read(end - start).decode("utf-8", errors="ignore")
        
        if is_windows:
            content = content.replace('\r\n', '\n')
            
        chunks = split_pattern.split(content)
        for chunk in chunks:
            words = PAT.findall(chunk)
            local_words_count.update(words)
            
    return local_words_count

def main_parallel(file_path, boundaries, split_pattern, is_windows):
    final_words_count = Counter()
    
    with ProcessPoolExecutor() as executor:
        tasks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            tasks.append(executor.submit(process_file_chunk, file_path, start, end, split_pattern, is_windows))
        
        for future in tasks:
            local_result = future.result()
            final_words_count.update(local_result)

    return final_words_count

class ListNode:
    def __init__(self, value, head, pre=None, next=None):
        self.value = value
        self.head = head
        self.pre = pre
        self.next = next
        self.is_valid = True

class WordInfo:
    def __init__(self, next: ListNode, count: int):
        self.next = next
        self.count = count


def create_linked_list(token_bytes, head: WordInfo):
    front = ListNode(token_bytes[0], head, head)
    current = front
    for byte in token_bytes[1:]:
        new_node = ListNode(byte, head, front)
        current.next = new_node
        new_node.pre = current
        current = new_node
    return front

class PairInfo:
    def __init__(self):
        self.positions = set()
        self.count = 0

def bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    num_blocks = 32
    is_windows = detect_newline(input_path) == '\r\n'
    split_pattern = re.compile('|'.join(re.escape(token) for token in special_tokens))
    tokens_array = list(bytes([i]) for i in range(0, 256))
    class PairKey:
        def __init__(self, pair):
            self.value = tuple(tokens_array[token] for token in pair) # Use tuple instead of join with b''
        def __lt__(self, other):
            return self.value > other.value

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_blocks, [token.encode("utf-8") for token in special_tokens])

    words_count = main_parallel(input_path, boundaries, split_pattern, is_windows)
    tokens_count = [WordInfo(None, 0) for _ in range(len(words_count))]
    for i, (word, count) in enumerate(words_count.items()):
        tokens_count[i].next = create_linked_list(word.encode('utf-8'), tokens_count[i])
        tokens_count[i].count = count
    #print(tokens_count)

    merges = []
    token_pair_count = defaultdict(lambda: PairInfo())
    pair_heap = []
    for word_data in tokens_count:
        current = word_data.next
        while current and current.next:
            pair = (current.value, current.next.value)
            pair_info = token_pair_count[pair]
            pair_info.count += word_data.count
            pair_info.positions.add(current)
            current = current.next
    for pair, info in token_pair_count.items():
        heapq.heappush(pair_heap, (-info.count, PairKey(pair), pair))
    while len(tokens_array) < vocab_size - len(special_tokens):
        if len(tokens_array) % 100 == 0:
            print(f"Vocabulary size: {len(tokens_array)} / {vocab_size}")
            
        while True:
            neg_count, _, pair_to_merge = heapq.heappop(pair_heap)
            if neg_count == -token_pair_count[pair_to_merge].count:
                break
        new_token = len(tokens_array)
        merges.append((tokens_array[pair_to_merge[0]], tokens_array[pair_to_merge[1]]))
        new_token_bytes = b''.join(tokens_array[token] for token in pair_to_merge)
        tokens_array.append(new_token_bytes)
        # if len(most_common_pairs) > 1:
        #     input(f'{[b''.join(tokens_array[token] for token in pair_to_merge) for pair_to_merge in most_common_pairs]} {new_token_bytes} {max_frequency}  {len(tokens_array)}')
        pair_info = token_pair_count.pop(pair_to_merge)
        changed_pairs = set()
        for node_a in pair_info.positions:
            if not node_a.is_valid:
                continue

            node_b = node_a.next
            node_p = node_a.pre
            node_q = node_b.next

            if node_p != node_a.head:
                left_pair = (node_p.value, node_a.value)
                if left_pair != pair_to_merge:
                    left_pair = (node_p.value, node_a.value)
                    left_pair_info = token_pair_count[left_pair]
                    left_pair_info.count -= node_a.head.count
                    changed_pairs.add(left_pair)
                    left_pair_info.positions.remove(node_p)
                    if left_pair_info.count == 0:
                        token_pair_count.pop(left_pair)

                new_left_pair = (node_p.value, new_token)
                new_left_pair_info = token_pair_count[new_left_pair]
                new_left_pair_info.count += node_a.head.count
                changed_pairs.add(new_left_pair)
                new_left_pair_info.positions.add(node_p)
            
            if node_q:
                right_pair = (node_b.value, node_q.value)
                if right_pair != pair_to_merge:
                    right_pair = (node_b.value, node_q.value)
                    right_pair_info = token_pair_count[right_pair]
                    right_pair_info.count -= node_a.head.count
                    changed_pairs.add(right_pair)
                    right_pair_info.positions.remove(node_b)
                    if right_pair_info.count == 0:
                        token_pair_count.pop(right_pair)

                new_right_pair = (new_token, node_q.value)
                new_right_pair_info = token_pair_count[new_right_pair]
                new_right_pair_info.count += node_a.head.count
                changed_pairs.add(new_right_pair)
                new_right_pair_info.positions.add(node_a)

            node_a.value = new_token
            node_a.next = node_q
            if node_q:
                node_q.pre = node_a
            node_b.is_valid = False
        for pair in changed_pairs:
            heapq.heappush(pair_heap, (-token_pair_count[pair].count, PairKey(pair), pair))
        #print(tokens_count[:10])
        #print(tokens_map)
    tokens_array += [token.encode("utf-8") for token in special_tokens]
    vocab = {i: token for i, token in enumerate(tokens_array)}
    return vocab, merges

if __name__ == "__main__":
    vocab, merges = bpe_tokenizer('./data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
    print(vocab, merges[:32])

