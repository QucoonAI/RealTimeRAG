import streamlit as st
import asyncio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import pyaudio  # Import PyAudio

# Placeholder for your async_update_db function
async def async_update_db(chunk):
    """Simulates upserting to a vector database."""
    await asyncio.sleep(0.5)  # Simulate some network latency
    print(f"Upserted chunk: {chunk[:50]}...")

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


async def write_chunks(stream, p, audio_stream):
    """Sends audio chunks from PyAudio to the transcription service."""
    try:
        while True:
            data = audio_stream.read(1024, exception_on_overflow=False)  # Read audio data
            if st.session_state.get('stop_requested', False):
                 break
            await stream.input_stream.send_audio_event(audio_chunk=data)  # Send audio data
            await asyncio.sleep(0)  # Yield control

    except asyncio.CancelledError:
        print("write_chunks cancelled")
    finally:
        print("write_chunks finished")


async def basic_transcribe(text_area):
    client = TranscribeStreamingClient(region="us-east-1")
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )
    handler = MyEventHandler(stream.output_stream, text_area)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                    frames_per_buffer=1024)

    try:
        await asyncio.gather(
            write_chunks(stream, p, audio_stream),  # Pass PyAudio objects
            handler.handle_events(),
        )

    except asyncio.CancelledError:
        print("basic_transcribe was cancelled.")
        # No need to explicitly stop the stream here; the finally block will handle it

    finally:
        print("basic_transcribe cleaning up")
        await handler.final_flush()
        await stream.input_stream.end_stream()
        audio_stream.stop_stream()  # Stop the PyAudio stream
        audio_stream.close()
        p.terminate()  # Terminate PyAudio
        print("basic_transcribe cleanup complete")

# def main():
#     st.title("Live Transcription and VectorDB Upsert")

#     # Create a placeholder for the text area
#     text_area = st.empty()
#     text_area.write("Click 'Start' to begin transcription.")

#     with st.expander("Run Transcription"):
#         if 'running' not in st.session_state:
#             st.session_state.running = False
#         if "task" not in st.session_state:
#             st.session_state.task = None
#         if 'stop_requested' not in st.session_state:
#              st.session_state.stop_requested = False


#         if not st.session_state.running:
#             if st.button("Start"):
#                 st.session_state.running = True
#                 st.session_state.stop_requested = False
#                 st.rerun()
#         else:
#             if st.button("Stop"):
#                 st.session_state.running = False
#                 st.session_state.stop_requested = True
#                 if st.session_state.task:
#                     st.session_state.task.cancel()


#         if st.session_state.running:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             try:
#                 loop.run_until_complete(startup()) # Assuming you still have startup
#                 st.session_state.task = loop.create_task(basic_transcribe(text_area))
#                 loop.run_until_complete(st.session_state.task)

#             except asyncio.CancelledError:
#                 print("Outer loop was cancelled.")
#                 # text_area.write("Transcription cancelled (outer).") # May not need

#             finally:
#                 loop.close()
#                 st.session_state.running = False
#                 st.session_state.stop_requested = False

# if __name__ == "__main__":
#     main()

# async def startup():
#     """Simulates initialization."""
#     await asyncio.sleep(0.1)  # Simulate some setup time
#     print("Startup complete")
#     return