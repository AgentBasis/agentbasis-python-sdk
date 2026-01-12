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
# This mimics what AgentBasis.init(agent_id="agent-123") does internally
resource = Resource.create({
    "service.name": "agentbasis-python-sdk",
    "service.version": "0.1.0",
    "deployment.environment": "production",
    "service.instance.id": "agent-123-test-id",
    "agentbasis.agent.id": "1234-5678-9012-3456"
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
tracer = trace.get_tracer("agentbasis.debug")

# 2. Simulate a Complex Agent Workflow
def simulate_openai_call(prompt, model="gpt-4"):
    """Simulates a call to OpenAI with token usage."""
    with tracer.start_as_current_span("openai.chat.completions.create") as span:
        # Standard OTel Attributes for GenAI
        span.set_attribute("llm.system", "openai")
        span.set_attribute("llm.request.model", model)
        span.set_attribute("llm.request.temperature", 0.7)
        span.set_attribute("llm.request.messages", f"[{{'role': 'user', 'content': '{prompt}'}}]")
        
        time.sleep(0.3)
        
        response_content = f"Analysis for: {prompt[:20]}..."
        span.set_attribute("llm.response.content", response_content)
        
        # Token Usage Tracking
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_content) // 4
        span.set_attribute("llm.usage.prompt_tokens", prompt_tokens)
        span.set_attribute("llm.usage.completion_tokens", completion_tokens)
        span.set_attribute("llm.usage.total_tokens", prompt_tokens + completion_tokens)
        
        return response_content

def agent_tool_call(tool_name, tool_input):
    """Simulates a tool execution."""
    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.input", str(tool_input))
        
        time.sleep(0.1)
        
        result = f"Result from {tool_name}: Success"
        span.set_attribute("tool.output", result)
        return result

def agent_workflow(user_query):
    """Top level agent function."""
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("agent.name", "FinancialAnalystAgent")
        span.set_attribute("input.query", user_query)
        
        print(f"Agent processing: {user_query}")
        
        # Step 1: Initial Thought
        simulate_openai_call("I need to search for stock data first.")
        
        # Step 2: Parallel Tool Calls
        agent_tool_call("google_search", {"query": "AAPL quarterly report 2024"})
        agent_tool_call("calculator", {"expression": "150 * 1.05"})
        
        # Step 3: Final Answer
        final_answer = simulate_openai_call("Here is the summary based on the search results.")
        
        span.set_attribute("output.result", final_answer)

# 3. Run the Simulation
print("--- SIMULATING REALISTIC AGENT TRACE ---")
agent_workflow("Analyze AAPL stock performance for Q3 2024")
print("--- DONE ---")
