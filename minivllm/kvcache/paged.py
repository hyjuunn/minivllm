"""
kvcache/paged.py

TODO:
This is where to implement the core idea of the vLLM paper. It's exactly
the same notion as virtual memory paging in an OS:
  - carve the whole GPU memory into a "block pool" of block_size (e.g. 16) tokens each
  - hand blocks to a sequence one at a time, as needed (no contiguity required!)
  - block_table[seq_id] = [3, 17, 5, ...]  ← logical position -> physical block mapping

Implementation order:
  1. BlockAllocator: free-block list, allocate() / free()
  2. append_token(seq_id): reserve a slot for one token (grab a new block when full)
  3. write(seq_id, layer, pos, k, v): follow block_table, write to the physical slot
  4. gather(seq_id, layer): follow block_table, collect K/V and hand them to attention
     (start with a slow Python gather -> swap in a Triton kernel later)
  5. prefix caching: same prompt hash -> share blocks + ref counting
  6. write tests where several Sequences of differing lengths coexist

Done when: tests/ and benchmarks/ show that, versus SimpleKVCache, (a) the output is
identical and (b) memory usage under batching goes down
"""


class PagedKVCache:
    def __init__(self, n_layers, n_kv_heads, head_dim, num_blocks, block_size, dtype, device):
        # physical storage shape = [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
        raise NotImplementedError("Not yet")
