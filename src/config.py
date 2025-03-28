GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 512  # Previously 1600. Maximum tokens per chunk for splitting
BATCH_SIZE = 1000  # Max batch size for embedding requests
CHUNK_OVERLAP = 0.1  # 10% overlap between chunks