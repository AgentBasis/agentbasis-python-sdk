from typing import Any
import functools
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Get our tracer
tracer = trace.get_tracer("agentbasis.llms.gemini")

def instrument_chat(gemini_module: Any):
    """
    Instruments the Google Gemini Chat API with OpenTelemetry.
    Also instruments underlying gRPC calls if available.
    """
    # 1. Instrument gRPC client (Gemini uses gRPC under the hood)
    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
        grpc_instrumentor = GrpcInstrumentorClient()
        grpc_instrumentor.instrument()
    except Exception:
        # Catch all exceptions to ensure gRPC instrumentation failures don't break
        # the main Gemini instrumentation. gRPC instrumentation is optional.
        pass

    # 2. Instrument the Gemini chat API
    try:
        from google.generativeai import GenerativeModel
    except ImportError:
        return

    original_generate_content = GenerativeModel.generate_content

    @functools.wraps(original_generate_content)
    def wrapped_generate_content(self, *args, **kwargs):
        model = getattr(self, "model_name", "unknown")
        contents = args[0] if args else kwargs.get('contents', [])
        tools = kwargs.get('tools', [])

        # Semantic Convention: "gemini.chat.generate_content"
        span_name = f"gemini.chat.generate_content {model}"

        with tracer.start_as_current_span(span_name) as span:
            # 1. Record Input Attributes (Semantic Conventions)
            span.set_attribute("llm.system", "gemini")
            span.set_attribute("llm.request.model", model)
            
            # We can serialize complex objects (contents) to string for now
            # In future, we might map them to specific OTel semantic events
            span.set_attribute("llm.request.messages", str(contents))
            
            # Track tools if provided to agent
            if tools:
                try:
                    tools_str = json.dumps(tools) if isinstance(tools, (list, dict)) else str(tools)
                except (TypeError, ValueError):
                    # Fallback to string representation if JSON serialization fails
                    tools_str = str(tools)
                span.set_attribute("llm.request.tools", tools_str)
                if isinstance(tools, list):
                    span.set_attribute("llm.request.tool_count", len(tools))

            try:
                response = original_generate_content(self, *args, **kwargs)

                # 2. Record Response Attributes
                if hasattr(response, 'text') and response.text:
                    span.set_attribute("llm.response.content", str(response.text))

                # Track function/tool calls if used in the agent
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'function_calls') and candidate.function_calls:
                        # Serialize all function calls as JSON for complete tracking
                        function_calls_data = []
                        for func_call in candidate.function_calls:
                            func_data = {
                                'name': getattr(func_call, 'name', None),
                            }
                            # Handle arguments - serialize as JSON if it's a dict/object
                            arguments = getattr(func_call, 'arguments', None)
                            if arguments is not None:
                                if isinstance(arguments, (dict, list)):
                                    try:
                                        func_data['arguments'] = json.dumps(arguments)
                                    except (TypeError, ValueError):
                                        # Fallback to string representation if JSON serialization fails
                                        func_data['arguments'] = str(arguments)
                                else:
                                    func_data['arguments'] = str(arguments)
                            else:
                                func_data['arguments'] = None
                            
                            if hasattr(func_call, 'response') and func_call.response:
                                func_data['response'] = str(func_call.response)
                            if hasattr(func_call, 'error') and func_call.error:
                                func_data['error'] = str(func_call.error)
                            function_calls_data.append(func_data)
                        
                        try:
                            function_calls_str = json.dumps(function_calls_data)
                        except (TypeError, ValueError):
                            # Fallback to string representation if JSON serialization fails
                            function_calls_str = str(function_calls_data)
                        span.set_attribute("llm.response.function_calls", function_calls_str)
                        span.set_attribute("llm.response.function_call_count", len(candidate.function_calls))

                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    if hasattr(usage, 'prompt_token_count'):
                        span.set_attribute("llm.usage.prompt_tokens", usage.prompt_token_count)
                    if hasattr(usage, 'candidates_token_count'):
                        span.set_attribute("llm.usage.completion_tokens", usage.candidates_token_count)
                    if hasattr(usage, 'total_token_count'):
                        span.set_attribute("llm.usage.total_tokens", usage.total_token_count)

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise e

    GenerativeModel.generate_content = wrapped_generate_content