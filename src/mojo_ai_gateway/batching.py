import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class _Item(Generic[T, R]):
    value: T
    future: asyncio.Future[R]


class AsyncBatcher(Generic[T, R]):
    def __init__(
        self,
        handler: Callable[[list[T]], Awaitable[list[R]]],
        max_size: int,
        max_delay_ms: int,
    ) -> None:
        self.handler = handler
        self.max_size = max_size
        self.max_delay = max_delay_ms / 1000
        self.queue: list[_Item[T, R]] = []
        self.lock = asyncio.Lock()
        self.flush_task: asyncio.Task[None] | None = None

    async def submit(self, value: T) -> R:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()
        async with self.lock:
            self.queue.append(_Item(value=value, future=future))
            if len(self.queue) >= self.max_size:
                await self._flush_locked()
            elif self.flush_task is None or self.flush_task.done():
                self.flush_task = asyncio.create_task(self._delayed_flush())
        return await future

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.max_delay)
        async with self.lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self.queue:
            return
        batch = self.queue
        self.queue = []
        try:
            results = await self.handler([item.value for item in batch])
            if len(results) != len(batch):
                raise RuntimeError("batch handler returned wrong result count")
            for item, result in zip(batch, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
