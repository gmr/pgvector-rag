pgvector-rag
============

A simple library for working with RAG documents using [pg_vector](https://github.com/pgvector/pgvector) in PostgreSQL.

Documents will pass through an optimization stage where the content is converted
to markdown, submitted to Anthropic's Sonet 3.5 to optimize the content, and the
response is used to a store a conside revision of the document.

OpenAI's `text-embedding-3-small` is used to generate the embeddings for the
for the document and used to generate the embeddings to compare against in
the database.

Schema is contained in the `postgres` directory.

Install with `pip install pgvector-rag`.
