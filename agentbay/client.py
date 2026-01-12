from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from .config import Config

class AgentBasis:
    """
    The main AgentBasis client.
    Manages OpenTelemetry configuration and data transmission.
    """
    _instance: Optional['AgentBasis'] = None

    def __init__(self, config: Config):
        self.config = config
        
        # 1. Create Resource (Metadata about who is sending data)
        attributes = {
            "service.name": "agentbasis-python-sdk",
            # We can add more metadata here like environment
        }
        
        # Add agent_id if present (it should be, as Config validates it)
        if config.agent_id:
            attributes["service.instance.id"] = config.agent_id
            # Also adding a custom attribute just in case we want to query by it explicitly later
            attributes["agentbasis.agent.id"] = config.agent_id

        resource = Resource.create(attributes=attributes)

        # 2. Initialize Tracer Provider
        self.tracer_provider = TracerProvider(resource=resource)

        # 3. Configure Exporter
        # We send data to <api_url>/api/v1/traces via HTTP/Protobuf, and also pass the API Key as a header
        endpoint = f"{config.api_url}/api/v1/traces" 
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )

        # 4. Add Batch Processor (Background thread for sending)
        processor = BatchSpanProcessor(exporter)
        self.tracer_provider.add_span_processor(processor)

        # 5. Register as Global Tracer
        # This allows trace.get_tracer(__name__) to work anywhere in the user's code
        trace.set_tracer_provider(self.tracer_provider)

    @classmethod
    def initialize(cls, api_key: Optional[str] = None, api_url: Optional[str] = None, agent_id: Optional[str] = None) -> 'AgentBasis':
        """
        Initializes the global AgentBasis client.
        """
        config = Config(api_key=api_key, api_url=api_url, agent_id=agent_id)
        config.validate()
        
        cls._instance = cls(config)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'AgentBasis':
        """
        Returns the global AgentBasis client instance.
        """
        if cls._instance is None:
            raise RuntimeError(
                "AgentBasis is not initialized. "
                "Please call `agentbasis.init(api_key='...', agent_id='...')` first."
            )
        return cls._instance

    def shutdown(self):
        """
        Flushes remaining spans and shuts down the provider.
        """
        if self.tracer_provider:
            self.tracer_provider.shutdown()
