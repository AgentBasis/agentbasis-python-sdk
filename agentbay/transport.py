# IMPORTANT NOTE: This transport layer is deprecated and will be removed in the future.
# We are using the OpenTelemetry SDK to send data to the AgentBasis backend.
# The OpenTelemetry SDK is a more robust and feature-rich SDK that is more suited for this purpose.

import threading
import time
import queue
from typing import Dict, Any, List, Optional
from .config import Config

class Transport:
    """
    Handles asynchronous data transmission to the AgentBasis backend.
    Uses a background thread to batch and send events.
    """
    def __init__(self, config: Config, batch_size: int = 10, flush_interval: float = 2.0):
        self.config = config
        self.queue: queue.Queue = queue.Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self):
        """
        Starts the background worker thread.
        """
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._worker_thread.start()

    def send(self, event: Dict[str, Any]):
        """
        Adds an event (Session or Span dict) to the queue.
        Non-blocking.
        """
        self.queue.put(event)

    def stop(self):
        """
        Stops the worker thread and flushes remaining events.
        """
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def _flush_loop(self):
        """
        The main loop for the background thread.
        """
        while not self._stop_event.is_set():
            batch = []
            start_time = time.time()
            
            # Collect a batch
            while len(batch) < self.batch_size:
                # Check if time is up for this batch
                if (time.time() - start_time) > self.flush_interval:
                    break
                
                try:
                    # Wait briefly for new items
                    item = self.queue.get(timeout=0.5)
                    batch.append(item)
                except queue.Empty:
                    # Queue is empty, continue checking time or stop event
                    continue

            # If we have data, send it
            if batch:
                self._send_batch(batch)

    def _send_batch(self, batch: List[Dict[str, Any]]):
        """
        Internal method to send a batch of data via HTTP.
        """
        import requests
        
        url = f"{self.config.api_url}/v1/telemetry" # API URL for the AgentBasis backend, needs to be configured 
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        try:
            response = requests.post(url, json=batch, headers=headers, timeout=10)
            # In a real SDK, we might log errors if response.status_code != 200
        except Exception as e:
            # In production, we would log this error or retry
            print(f"AgentBasis SDK Error: Failed to send telemetry: {e}")
