"""
pgvector_rag
============

A simple library for working with RAG documents using pg_vector in PostgreSQL.

"""
version = '0.1.0'

from .rag import Document, RAG

__all__ = ['Document', 'RAG', 'version']
