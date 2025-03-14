import asyncio
import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from asyncio import Semaphore
from ragEmbed import async_update_db

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.current_words = []
        self.chunk_size = 200
        self.overlap_size = 70
        self.previous_chunk_end = []
        self.last_transcript = ""
        self.upsert_sem = Semaphore(5)
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                new_text = alt.transcript
                
                # Prevent repeating previous words
                if new_text.startswith(self.last_transcript):
                    new_text = new_text[len(self.last_transcript):].strip()
                
                new_words = new_text.split()
                if new_words:
                    self.current_words.extend(new_words)
                    print("[Streaming]:", new_text)  # Print streaming text
                    self.last_transcript = alt.transcript  # Update last seen transcript
                
                if len(self.current_words) >= self.chunk_size:
                    await self.store_chunk()  # <-- Fix: Added `await`
    
    async def store_chunk(self):
        if len(self.current_words) < self.chunk_size:
            return  # Wait until we have a full chunk

        # Get the current chunk and overlap from previous
        chunk_start = max(0, len(self.previous_chunk_end) - self.overlap_size)
        overlap = self.previous_chunk_end[chunk_start:]
        new_chunk_words = overlap + self.current_words[:self.chunk_size]
        
        # Update previous chunk end for next overlap
        self.previous_chunk_end = new_chunk_words[-self.overlap_size:]
        
        # Remove processed words (keep overlap for next chunk)
        self.current_words = self.current_words[self.chunk_size:]
        
        # Upsert the new chunk with overlap
        chunk_text = " ".join(new_chunk_words)
        asyncio.create_task(self.upsert_to_vector_db(chunk_text))
    
    async def final_flush(self):
        if self.current_words:
            # Combine remaining words with overlap
            chunk_start = max(0, len(self.previous_chunk_end) - self.overlap_size)
            chunk_text = " ".join(self.previous_chunk_end[chunk_start:] + self.current_words)
            await self.upsert_to_vector_db(chunk_text)

    async def upsert_to_vector_db(self, chunk):
        async with self.upsert_sem:
            try:
                # Use async version of update_db
                await async_update_db(chunk)  # Requires implementing async_update_db
                print(f"[Upserted] {chunk[:50]}...")
            except Exception as e:
                print(f"Failed to upsert: {str(e)}")

async def mic_stream():
    loop = asyncio.get_running_loop()  # Updated to avoid errors
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status

async def write_chunks(stream):
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()

async def basic_transcribe():
    client = TranscribeStreamingClient(region="us-east-1")
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )
    handler = MyEventHandler(stream.output_stream)
    
    try:
        await asyncio.gather(
            write_chunks(stream),
            handler.handle_events(),
        )
    finally:
        await handler.final_flush()
        await stream.input_stream.end_stream()

# Updated event loop handling for Python 3.11+
if __name__ == "__main__":
    try:
        asyncio.run(basic_transcribe())
    except KeyboardInterrupt:
        print("Streaming stopped by user.")