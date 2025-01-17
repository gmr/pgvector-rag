pgvector-rag
============

A simple library for working with RAG documents using [pg_vector](https://github.com/pgvector/pgvector) in PostgreSQL.

Documents will pass through an optimization stage where the content is converted
to markdown, submitted to OpenAI's GPT-4o, and the response is used to a more
conside revision of the document for storage and tokenization.

OpenAI's `text-embedding-3-small` is used to generate the embeddings for the
for the document and used to generate the embeddings to compare against in
the database.

Schema is contained in the `postgres` directory.

Install with `pip install pgvector-rag`.
