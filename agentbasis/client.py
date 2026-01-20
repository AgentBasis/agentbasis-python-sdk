from typing import Optional
import atexit
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
    _shutdown_registered: bool = False

    def __init__(self, config: Config):
        self.config = config
        self._is_shutdown = False
        
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
        endpoint = f"{config.api_url}/api/v1/traces" 
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )

        # 4. Add Batch Processor (Background thread for sending)
        self._processor = BatchSpanProcessor(exporter)
        self.tracer_provider.add_span_processor(self._processor)

        # 5. Register as Global Tracer
        # This allows trace.get_tracer(__name__) to work anywhere in the user's code
        trace.set_tracer_provider(self.tracer_provider)
        
        # 6. Register atexit handler for graceful shutdown
        self._register_atexit()

    def _register_atexit(self):
        """
        Register the shutdown handler to run when Python exits.
        Only registers once to avoid duplicate handlers.
        """
        if not AgentBasis._shutdown_registered:
            atexit.register(self._atexit_handler)
            AgentBasis._shutdown_registered = True

    def _atexit_handler(self):
        """
        Handler called when Python exits. Flushes and shuts down gracefully.
        """
        self.shutdown()

    @classmethod
    def initialize(cls, api_key: Optional[str] = None, agent_id: Optional[str] = None) -> 'AgentBasis':
        """
        Initializes the global AgentBasis client.
        """
        config = Config(api_key=api_key, agent_id=agent_id)
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

    def flush(self, timeout_millis: int = 30000) -> bool:
        """
        Forces a flush of all pending spans.
        
        This is useful when you want to ensure all telemetry is sent before
        a critical operation, or at specific checkpoints in your application.
        
        Args:
            timeout_millis: Maximum time to wait for flush to complete (default 30 seconds).
            
        Returns:
            True if flush completed successfully, False if timed out.
        """
        if self._is_shutdown:
            return False
            
        if self.tracer_provider and hasattr(self.tracer_provider, 'force_flush'):
            return self.tracer_provider.force_flush(timeout_millis)
        return True

    def shutdown(self):
        """
        Flushes remaining spans and shuts down the provider.
        
        This method is idempotent - calling it multiple times is safe.
        It's automatically called when Python exits via atexit.
        """
        if self._is_shutdown:
            return
            
        self._is_shutdown = True
        
        if self.tracer_provider:
            try:
                self.tracer_provider.shutdown()
            except Exception:
                # Silently ignore shutdown errors to avoid noise during exit
                pass
