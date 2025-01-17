import datetime
import logging
import re
import typing

import anthropic
import markdown_it
import openai
import psycopg
import pydantic
import uuid_utils
from psycopg import rows

LOGGER = logging.getLogger(__name__)

INSERT_SQL = re.sub(r'\s+', ' ', """\
    INSERT INTO documents (document_id, title, content, url, labels)
         VALUES (%(document_id)s,  %(title)s,
                 %(content)s, %(url)s, %(labels)s)
    ON CONFLICT (url)
        DO UPDATE SET title = EXCLUDED.title,
                      content = EXCLUDED.content,
                      labels = EXCLUDED.labels,
                      modified_at = CURRENT_TIMESTAMP
    RETURNING document_id
""").encode('utf-8')

INSERT_CHUNK_SQL = re.sub(r'\s+', ' ', """\
    INSERT INTO chunks (document_id, chunk, embedding)
         VALUES (%(document_id)s,  %(chunk)s, %(embedding)s)
    ON CONFLICT (document_id, chunk)
      DO UPDATE SET embedding = EXCLUDED.embedding;
""").encode('utf-8')

DELETE_STALE_CHUNKS = re.sub(r'\s+', ' ', """\
    DELETE FROM chunks
          WHERE document_id = %(document_id)s
            AND chunk > %(chunk)s
""").encode('utf-8')

SEARCH_SQL = re.sub(r'\s+', ' ', """\
WITH matches AS (
    SELECT document_id,
           CAST(1 - (embedding <=> %(emb)s::vector) AS float) as similarity
      FROM chunks
     WHERE vector_dims(embedding) = vector_dims(%(emb)s::vector)
       AND 1 - (embedding <=> %(emb)s::vector) > 0.1
  ORDER BY embedding <=> %(emb)s::vector
     LIMIT %(limit)s)
    SELECT content,
           max(similarity) as similarity
      FROM documents
      JOIN matches USING (document_id)
     WHERE document_id IN (SELECT document_id FROM matches)
  GROUP BY content
""").encode('utf-8')

OPTIMIZE_PROMPT = re.sub(r'\s+', ' ', """\
Optimize this content, including its title, for use in a RAG system that will
be used with LLM models.  You can remove any unnecessary information, but do
not remove any substantive content. Do not remove or abbreviate any
imperatives, leave them as written. Do not include a preface that tells me
what the content is, just return the content.
""")

class Document(pydantic.BaseModel):
    title: str
    url: str
    labels: str
    last_modified_at: datetime.datetime
    content: str


class RAG:
    """Retrieval Augmented Generation system for LLM models."""

    def __init__(self,
                 anthropic_api_key: str,
                 openai_api_key: str,
                 postgres_uri: str):
        self._anthropic = anthropic.Client(api_key=anthropic_api_key)
        self._openai = openai.Client(api_key=openai_api_key)
        self._postgres =  psycopg.connect(postgres_uri)
        self._postgres.autocommit = True
        self._postgres_cursor = \
            self._postgres.cursor(row_factory=rows.dict_row)

    def add_document(self, document: Document) -> None:
            """Add document to vector store, optimizing it for LLM usage."""
            optimized = self._optimize_document(document)
            document_id = str(uuid_utils.uuid7())
            self._postgres_cursor.execute(
                INSERT_SQL,
                {
                    'document_id': document_id,
                    'title': document.title,
                    'url': document.url,
                    'labels': document.labels,
                    'content': optimized
                })
            new_document_id = self._postgres_cursor.fetchone()['document_id']
            updated = new_document_id != document_id
            if updated:
                LOGGER.info('Updated "%s"', document.title)
                document_id = new_document_id

            chunk = 9999
            for chunk, value in enumerate(self._chunk_document(optimized)):
                LOGGER.info('Processing chunk #%i (%i bytes)',
                            chunk, len(value))
                embedding = self._get_embedding(value)
                self._postgres_cursor.execute(
                    INSERT_CHUNK_SQL,
                    {
                        'document_id': document_id,
                        'chunk': chunk,
                        'embedding': embedding
                    })
            if updated and chunk != 9999:
                LOGGER.info('Deleting stale chunks')
                self._postgres_cursor.execute(
                    DELETE_STALE_CHUNKS,
                    {
                        'document_id': document_id,
                        'chunk': chunk
                    })

    def search(self, query: str, limit: int = 8) -> list[dict[str, float]]:
        """Search for documents to use with an LLM model for context.

        Returns a list of dictionaries with:

            - content: Document content
            - similarity: Similarity to query

        """
        vectors = self._get_embedding(query)
        self._postgres_cursor.execute(
            SEARCH_SQL, {'emb': vectors, 'limit': limit})
        results = self._postgres_cursor.fetchall()
        return [
            {
                'content': r['content'],
                'similarity': float(r['similarity'])
            }
            for r in results
        ]

    def _chunk_document(self,
                        text: str,
                        chunk_size: int = 500,
                        overlap: int = 50) \
            -> typing.Generator[str, None, None]:
        """Creates overlapping chunks from document text.

        Args:
            text: Document text to chunk
            chunk_size: Target size for each chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            List of chunk dicts with id and text fields

        """
        if overlap >= chunk_size:
            raise ValueError("Overlap cannot be >= to chunk_size.")
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            yield chunk

    def _chunk_markdown(self, value: str) -> typing.Generator[str, None, None]:
        """Chunk markdown content into smaller pieces."""
        md = markdown_it.MarkdownIt()
        tokens = md.parse(value)
        current_chunk = []
        for token in tokens:
            if token.type == 'heading_open':
                if current_chunk and len(''.join(current_chunk)) > 2048:
                    yield ''.join(current_chunk)
                    current_chunk = []
            current_chunk.append(token.markup + token.content)
        if current_chunk:
            yield ''.join(current_chunk)

    def _get_embedding(self, text: str) -> list[float]:
            """Get embeddings using OpenAI API."""
            response = self._openai.embeddings.create(
                input=text, model="text-embedding-3-small")
            return response.data[0].embedding

    def _optimize_document(self, document: Document) -> str:
            """Optimize document for RAG system."""
            LOGGER.info('Optimizing "%s"', document.title)
            output = [
                f'Title: {document.title}',
                f'URL: {document.url}',
                f'Last Modified: {document.last_modified_at}',
                f'Labels: {document.labels}' if document.labels else '',
                '<content>'
            ]
            for chunk in self._chunk_markdown(document.content):
                response = self._anthropic.messages.create(
                    model='claude-3-5-sonnet-20241022',
                    messages=[
                        {
                            'role': 'user',
                            'content': chunk
                        }
                    ],
                    max_tokens=1024,
                    system=OPTIMIZE_PROMPT)
                output.append(str(response.content[0].text))
            output.append('</content>')
            return '\n'.join(output)
