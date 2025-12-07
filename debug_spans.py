import sys
import time
import json
from dataclasses import dataclass
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Dummy Response object to satisfy OTel SDK
@dataclass
class MockResponse:
    ok: bool = True
    status_code: int = 200
    text: str = "OK"

# 1. Setup OTel with realistic Resource metadata
# This mimics what AgentBay.init(agent_id="agent-123") does internally
resource = Resource.create({
    "service.name": "agentbay-python-sdk",
    "service.version": "0.1.0",
    "deployment.environment": "production",
    # THIS IS THE NEW FIELD
    "service.instance.id": "agent-123-test-id",
    "agentbay.agent.id": "agent-123-test-id"
})

provider = TracerProvider(resource=resource)

# Exporter 2: OTLP JSON (Strict Backend Format)
class DebugOTLPExporter(OTLPSpanExporter):
    def _export(self, serialized_data, *args, **kwargs):
        try:
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
            from google.protobuf.json_format import MessageToJson
            
            request = ExportTraceServiceRequest()
            request.ParseFromString(serialized_data)
            
            json_output = MessageToJson(request)
            
            print("\n" + "="*50)
            print("  ACTUAL OTLP JSON PAYLOAD (What Backend Receives)  ")
            print("="*50)
            print(json_output)
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to decode protobuf: {e}")

        return MockResponse()

otlp_processor = SimpleSpanProcessor(DebugOTLPExporter(endpoint="http://dummy"))
provider.add_span_processor(otlp_processor)

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agentbay.debug")

# 2. Simulate a Complex Agent Workflow
def simulate_openai_call(prompt):
    """Simulates a call to OpenAI with token usage."""
    with tracer.start_as_current_span("openai.chat.completions.create") as span:
        span.set_attribute("llm.system", "openai")
        span.set_attribute("llm.request.model", "gpt-4")
        span.set_attribute("llm.request.messages", f"[{{'role': 'user', 'content': '{prompt}'}}]")
        
        time.sleep(0.2)
        
        response_content = "Here is a summary of your data..."
        span.set_attribute("llm.response.content", response_content)
        span.set_attribute("llm.usage.prompt_tokens", 50)
        span.set_attribute("llm.usage.completion_tokens", 20)
        span.set_attribute("llm.usage.total_tokens", 70)
        return response_content

def agent_workflow(user_query):
    """Top level agent function."""
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("agent.name", "FinancialAnalystAgent")
        span.set_attribute("input.query", user_query)
        
        print(f"Agent processing: {user_query}")
        
        with tracer.start_as_current_span("tool.search") as tool_span:
            tool_span.set_attribute("tool.name", "google_search")
            time.sleep(0.1)
            tool_span.set_attribute("tool.result", "Found 5 financial reports.")
        
        result = simulate_openai_call(user_query)
        span.set_attribute("output.result", result)

# 3. Run the Simulation
print("--- SIMULATING REALISTIC AGENT TRACE ---")
agent_workflow("Analyze AAPL stock performance")
print("--- DONE ---")
