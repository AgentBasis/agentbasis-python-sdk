import unittest
from unittest.mock import MagicMock, patch
import sys
import types

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# --- MOCK SETUP START ---
# We must setup the mocks BEFORE importing the module under test
# This simulates 'gemini' being installed on the system

mock_google = types.ModuleType("google")
mock_genai = types.ModuleType("google.generativeai")

class MockGenerativeModel:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name

        def generate_content(self, *args, **kwargs):
            pass
                 
        