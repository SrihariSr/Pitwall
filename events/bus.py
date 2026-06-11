import asyncio
from typing import AsyncIterator
from events.types import Event

class EventBus:
    def __init__(self) -> None:
        # Each subscriber gets their own queue
        self._subscribers: list[tuple[asyncio.Queue, set[str] | None]] = []
        self._max_queue_size = 1000
    
    async def publish(self, event: Event) -> None:
        """
        Broadcast one event to every interested subscriber.
        """
        for queue, type_filter in self._subscribers:
            if type_filter is not None and event.event_type not in type_filter:
                continue

            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                raise RuntimeError(f"Event bus subscrber queue is full.\nCurrently has {queue.qsize()} items.")
    
    async def subscribe(self, event_types: list[str] | None = None,) -> AsyncIterator[Event]:
        """
        Subsribe to the bus and recieve events as they occur.
        """

        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        type_filter = set(event_types) if event_types is not None else None
        self._subscribers.append((queue, type_filter))

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.remove((queue, type_filter))
            