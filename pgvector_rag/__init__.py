"""
pgvector_rag
============

A simple library for working with RAG documents using pg_vector in PostgreSQL.

"""
from .rag import RAG, Document

version = '0.2.0'

__all__ = ['Document', 'RAG', 'version']
