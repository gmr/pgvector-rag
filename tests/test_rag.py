import unittest

import pgvector_rag
from tests import mixins


class TestRag(mixins.EnvVarMixin, unittest.TestCase):

    def test_rag_initialization(self):
        rag = pgvector_rag.RAG('api_key', self.os_environ['POSTGRES_URL'])
        self.assertIsInstance(rag, pgvector_rag.RAG)
