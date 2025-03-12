import asyncio
import sounddevice
import streamlit as st
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from ragEmbed import async_update_db, startup  # Import startup


class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, text_area):
        super().__init__(output_stream)
        self.current_words = []
        self.chunk_size = 200
        self.overlap_size = 70
        self.previous_chunk_end = []
        self.last_transcript = ""
        self.text_area = text_area  # Store the text area for updating
        self.full_transcript = ""  # Keep track of the full transcript

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
                    self.full_transcript += " " + new_text  # Append new text to full transcript
                    self.text_area.write(self.full_transcript)  # Update the text area
                    self.last_transcript = alt.transcript  # Update last seen transcript

                if len(self.current_words) >= self.chunk_size:
                    await self.store_chunk()

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
            # Update the text area one last time, if there are any more words
            self.full_transcript += " " + chunk_text
            self.text_area.write(self.full_transcript)

    async def upsert_to_vector_db(self, chunk):
        try:
            # Use async version of update_db
            await async_update_db(chunk)  # Requires implementing async_update_db
            print(f"[Upserted] {chunk[:50]}...")
        except Exception as e:
            print(f"Failed to upsert: {str(e)}")


async def mic_stream():
    loop = asyncio.get_running_loop()
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


async def basic_transcribe(text_area):
    client = TranscribeStreamingClient(region="us-east-1")
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )
    handler = MyEventHandler(stream.output_stream, text_area)

    try:
        await asyncio.gather(
            write_chunks(stream),
            handler.handle_events(),
        )
    finally:
        await handler.final_flush()
        await stream.input_stream.end_stream()


# def main():
#     st.title("Live Transcription and VectorDB Upsert")

#     # Create a placeholder for the text area
#     text_area = st.empty()
#     text_area.write("Click 'Start' to begin transcription.") # Initial instructions

#     # Use an expander for running the transcription
#     with st.expander("Run Transcription"):
#         # Use session_state to store the running state
#         if 'running' not in st.session_state:
#             st.session_state.running = False

#         if "task" not in st.session_state:
#             st.session_state.task = None

#         if 'stop_requested' not in st.session_state: # Add stop_requested
#              st.session_state.stop_requested = False

#         if not st.session_state.running:
#             if st.button("Start"):
#                 st.session_state.running = True
#                 st.session_state.stop_requested = False
#                 # Use rerun to immediately update the UI and start the process
#                 st.rerun()
#         else:
#             if st.button("Stop"):
#                 st.session_state.running = False
#                 st.session_state.stop_requested = True # Request a stop
#                 if st.session_state.task:
#                     st.session_state.task.cancel()  # Cancel the current task

#         if st.session_state.running:
#             loop = asyncio.new_event_loop()  # Create a *new* event loop
#             asyncio.set_event_loop(loop)
#             try:
#                 loop.run_until_complete(startup())  # Initialize clients
#                 loop.run_until_complete(basic_transcribe(text_area))  # Run transcription

#             finally:
#                 loop.close()
#                 # Reset running state when transcription completes.  Very important!
#                 st.session_state.running = False

# if __name__ == "__main__":
#     main()