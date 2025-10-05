import argparse
import asyncio
import os
import sys
import logging
import json
import time
import multiprocessing
import threading
import queue
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import signal

import openai
import aiohttp
from dotenv import load_dotenv

# Add the parent directory to sys.path to import prompts and utils
sys.path.append(str(Path(__file__).parent / "data"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from prompts import PROMPTS
from inference_utils import load_vllm_model, batch_conversation_chat, close_vllm_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class ConversationState:
    """Represents the state of an ongoing conversation."""
    persona_data: Dict[str, Any]
    target_prompt: Optional[Dict[str, Any]]
    auditor_messages: List[Dict[str, str]]
    target_messages: List[Dict[str, str]]
    full_conversation: List[Dict[str, str]]
    current_turn: int
    max_turns: int
    conversation_id: str

    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        # Check turn limit
        if self.current_turn >= self.max_turns:
            return True

        # Check for end signal
        if self.full_conversation and self.is_end_conversation_signal(
            self.full_conversation[-1]["content"]
        ):
            return True

        return False

    def is_end_conversation_signal(self, message: str) -> bool:
        """Check if the message contains an end conversation signal."""
        message_clean = message.strip()
        return message_clean == 'END_CONVERSATION' or message_clean == '<END_CONVERSATION>'

    def to_transcript(self, target_model_name: str, auditor_model_name: str) -> Dict[str, Any]:
        """Convert to transcript format."""
        transcript = {
            "model": target_model_name,
            "auditor_model": auditor_model_name,
            "domain": self.persona_data["domain"],
            "persona_id": self.persona_data["persona_id"],
            "persona": self.persona_data["persona"],
            "topic_id": self.persona_data["topic_id"],
            "topic": self.persona_data["topic"],
            "turns": len(self.full_conversation),
            "conversation": self.full_conversation,
            "conversation_id": self.conversation_id
        }

        # Add target prompt metadata if available
        if self.target_prompt:
            transcript["target_role"] = self.target_prompt["role"]
            transcript["target_prompt_index"] = self.target_prompt["prompt_index"]
            transcript["target_prompt"] = self.target_prompt["system_prompt"]

        # Check if conversation ended early due to end signal
        if (self.full_conversation and
            self.is_end_conversation_signal(self.full_conversation[-1]["content"])):
            transcript["ended_early"] = True

        return transcript


class VLLMWorker(multiprocessing.Process):
    """Worker process for handling vLLM target model inference in batches."""

    def __init__(self, target_model_name: str, max_model_len: int, vllm_queue: multiprocessing.Queue,
                 openrouter_queue: multiprocessing.Queue, completed_queue: multiprocessing.Queue,
                 progress_counter: multiprocessing.Value, batch_size: int = 100, timeout: float = 0.1):
        super().__init__()
        self.target_model_name = target_model_name
        self.max_model_len = max_model_len
        self.vllm_queue = vllm_queue
        self.openrouter_queue = openrouter_queue
        self.completed_queue = completed_queue
        self.progress_counter = progress_counter
        self.batch_size = batch_size
        self.timeout = timeout
        self.target_model = None
        self.should_stop = False

    def setup_model(self):
        """Initialize the vLLM model."""
        logger.info(f"VLLMWorker: Loading target model: {self.target_model_name}")
        self.target_model = load_vllm_model(self.target_model_name, max_model_len=self.max_model_len)
        logger.info("VLLMWorker: Model loaded successfully")

    def cleanup_model(self):
        """Clean up the vLLM model."""
        if self.target_model:
            close_vllm_model(self.target_model)
            self.target_model = None
            logger.info("VLLMWorker: Model cleanup completed")

    def calculate_adaptive_batch_size(self, conversation_states: List[ConversationState]) -> int:
        """Calculate optimal batch size based on conversation lengths and turn progress."""
        if not conversation_states:
            return self.batch_size

        # Calculate average turn progress and conversation length
        total_turn_progress = 0
        total_length = 0

        for state in conversation_states:
            # Turn progress as ratio (0.0 = start, 1.0 = max_turns)
            turn_progress = state.current_turn / state.max_turns
            total_turn_progress += turn_progress

            # Conversation length (number of messages)
            conversation_length = len(state.full_conversation)
            total_length += conversation_length

        avg_turn_progress = total_turn_progress / len(conversation_states)
        avg_length = total_length / len(conversation_states)

        # Adaptive batch sizing based on turn progress
        # Early turns (0-20% progress): batch_size = 100
        # Mid turns (20-60% progress): batch_size = 50-20
        # Late turns (60-80% progress): batch_size = 20-5
        # Final turns (80-100% progress): batch_size = 5-1

        if avg_turn_progress <= 0.2:  # Early conversations
            adaptive_size = self.batch_size  # Use full batch size (100)
        elif avg_turn_progress <= 0.6:  # Mid conversations
            # Linear interpolation from 100 to 20
            progress_in_range = (avg_turn_progress - 0.2) / 0.4
            adaptive_size = int(100 - (80 * progress_in_range))
        elif avg_turn_progress <= 0.8:  # Late conversations
            # Linear interpolation from 20 to 5
            progress_in_range = (avg_turn_progress - 0.6) / 0.2
            adaptive_size = int(20 - (15 * progress_in_range))
        else:  # Final turns
            # Linear interpolation from 5 to 1
            progress_in_range = (avg_turn_progress - 0.8) / 0.2
            adaptive_size = max(1, int(5 - (4 * progress_in_range)))

        # Also consider average conversation length for fine-tuning
        if avg_length > 20:  # Very long conversations
            adaptive_size = max(1, adaptive_size // 2)
        elif avg_length > 15:  # Long conversations
            adaptive_size = max(1, int(adaptive_size * 0.7))

        logger.debug(f"VLLMWorker: Adaptive batch size: {adaptive_size} "
                    f"(avg_turn_progress: {avg_turn_progress:.2f}, avg_length: {avg_length:.1f})")

        return max(1, adaptive_size)

    def collect_batch(self) -> List[ConversationState]:
        """Collect a batch of conversations from the queue with adaptive sizing."""
        initial_batch = []
        deadline = time.time() + self.timeout

        # First, collect a small sample to assess conversation states
        sample_size = min(10, self.batch_size)
        while len(initial_batch) < sample_size and time.time() < deadline and not self.should_stop:
            try:
                remaining_time = max(0.001, deadline - time.time())
                conversation_state = self.vllm_queue.get(timeout=remaining_time)
                initial_batch.append(conversation_state)
            except queue.Empty:
                break

        if not initial_batch:
            return []

        # Calculate adaptive batch size based on sample
        adaptive_batch_size = self.calculate_adaptive_batch_size(initial_batch)

        # If we already have enough, return the sample
        if len(initial_batch) >= adaptive_batch_size:
            # Put back excess conversations
            for excess in initial_batch[adaptive_batch_size:]:
                self.vllm_queue.put(excess)
            return initial_batch[:adaptive_batch_size]

        # Otherwise, collect more up to the adaptive size
        batch = initial_batch
        while len(batch) < adaptive_batch_size and time.time() < deadline and not self.should_stop:
            try:
                remaining_time = max(0.001, deadline - time.time())
                conversation_state = self.vllm_queue.get(timeout=remaining_time)
                batch.append(conversation_state)
            except queue.Empty:
                break

        logger.debug(f"VLLMWorker: Collected batch of {len(batch)} conversations "
                    f"(target: {adaptive_batch_size})")

        return batch

    def get_target_responses(self, conversation_states: List[ConversationState]) -> List[str]:
        """Get responses from target model for a batch of conversations."""
        try:
            # Prepare conversations for batch processing
            conversations = []
            for state in conversation_states:
                # Handle Gemma models specially - they don't support system prompts
                is_gemma = self.target_model_name.startswith("google/gemma-2")
                target_system_prompt = state.target_prompt["system_prompt"] if state.target_prompt else None

                if is_gemma and target_system_prompt:
                    # For Gemma models, modify first user message
                    modified_messages = []
                    first_user_found = False

                    for msg in state.target_messages:
                        if msg["role"] == "user" and not first_user_found:
                            modified_content = f"{target_system_prompt}\n\n{msg['content']}"
                            modified_messages.append({"role": "user", "content": modified_content})
                            first_user_found = True
                        elif msg["role"] != "system":
                            modified_messages.append(msg)

                    conversations.append(modified_messages)
                else:
                    # For non-Gemma models, filter system messages and add system prompt
                    filtered_messages = [msg for msg in state.target_messages if msg["role"] != "system"]
                    if target_system_prompt:
                        filtered_messages.insert(0, {"role": "system", "content": target_system_prompt})
                    conversations.append(filtered_messages)

            # Use batch processing from inference_utils
            responses = batch_conversation_chat(
                self.target_model,
                conversations,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.8,
                progress=False
            )

            return responses

        except Exception as e:
            logger.error(f"VLLMWorker: Error getting target responses: {e}")
            # Return empty responses for failed batch
            return [""] * len(conversation_states)

    def process_batch(self, conversation_states: List[ConversationState]):
        """Process a batch of conversations through the target model."""
        if not conversation_states:
            return

        if conversation_states:
            avg_turns = sum(state.current_turn for state in conversation_states) / len(conversation_states)
            logger.info(f"VLLMWorker: Processing batch of {len(conversation_states)} conversations "
                       f"(avg turn: {avg_turns:.1f})")

        try:
            responses = self.get_target_responses(conversation_states)

            for state, response in zip(conversation_states, responses):
                if not response:
                    logger.warning(f"Empty response for conversation {state.conversation_id}")
                    continue

                # Update conversation state
                state.target_messages.append({"role": "assistant", "content": response})
                state.auditor_messages.append({"role": "user", "content": response})
                state.full_conversation.append({"role": "assistant", "content": response})
                state.current_turn += 1

                # Check if conversation is complete
                if state.is_complete():
                    logger.info(f"Conversation {state.conversation_id} completed after {state.current_turn} turns")
                    self.completed_queue.put(state)

                    with self.progress_counter.get_lock():
                        self.progress_counter.value += 1
                else:
                    # Send back to OpenRouter queue for next turn
                    self.openrouter_queue.put(state)

        except Exception as e:
            logger.error(f"VLLMWorker: Error processing batch: {e}")
            # Put conversations back for retry
            for state in conversation_states:
                self.openrouter_queue.put(state)

    def run(self):
        """Main worker loop."""
        try:
            self.setup_model()

            while not self.should_stop:
                batch = self.collect_batch()
                if batch:
                    self.process_batch(batch)
                elif self.should_stop:
                    break
                else:
                    # Small sleep when no work available
                    time.sleep(0.01)

        except Exception as e:
            logger.error(f"VLLMWorker: Fatal error in run loop: {e}")
        finally:
            self.cleanup_model()
            logger.info("VLLMWorker: Shutting down")

    def stop(self):
        """Signal worker to stop."""
        self.should_stop = True


class OpenRouterWorker(multiprocessing.Process):
    """Async worker process for handling OpenRouter API calls."""

    def __init__(self, worker_id: int, auditor_model_name: str, openrouter_queue: multiprocessing.Queue,
                 vllm_queue: multiprocessing.Queue, completed_queue: multiprocessing.Queue,
                 progress_counter: multiprocessing.Value, batch_size: int = 40, timeout: float = 0.2):
        super().__init__()
        self.worker_id = worker_id
        self.auditor_model_name = auditor_model_name
        self.openrouter_queue = openrouter_queue
        self.vllm_queue = vllm_queue
        self.completed_queue = completed_queue
        self.progress_counter = progress_counter
        self.batch_size = batch_size
        self.timeout = timeout
        self.should_stop = False

    def collect_batch(self) -> List[ConversationState]:
        """Collect a batch of conversations from the queue with timeout."""
        batch = []
        deadline = time.time() + self.timeout

        while len(batch) < self.batch_size and time.time() < deadline and not self.should_stop:
            try:
                remaining_time = max(0.001, deadline - time.time())
                conversation_state = self.openrouter_queue.get(timeout=remaining_time)
                batch.append(conversation_state)
            except queue.Empty:
                break

        return batch

    async def get_auditor_response(self, session: aiohttp.ClientSession, messages: List[Dict[str, str]],
                                  max_retries: int = 3) -> Tuple[str, bool]:
        """
        Get response from auditor model via OpenRouter API with retry logic.

        Returns:
            Tuple of (response_text, success_flag)
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.auditor_model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512
        }

        last_exception = None

        for attempt in range(max_retries):
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        return content, True
                    elif response.status == 429:  # Rate limit
                        logger.warning(f"OpenRouterWorker {self.worker_id}: Rate limited on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    elif response.status >= 500:  # Server error
                        logger.warning(f"OpenRouterWorker {self.worker_id}: Server error {response.status} on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                    else:
                        logger.error(f"OpenRouterWorker {self.worker_id}: API error {response.status}: {await response.text()}")
                        return "", False

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"OpenRouterWorker {self.worker_id}: Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(f"OpenRouterWorker {self.worker_id}: Client error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

            except Exception as e:
                last_exception = e
                logger.warning(f"OpenRouterWorker {self.worker_id}: Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        # All retries failed
        logger.error(f"OpenRouterWorker {self.worker_id}: All {max_retries} attempts failed. Last error: {last_exception}")
        return "", False

    async def process_batch_async(self, conversation_states: List[ConversationState]):
        """Process a batch of conversations through the auditor model asynchronously."""
        if not conversation_states:
            return

        logger.info(f"OpenRouterWorker {self.worker_id}: Processing batch of {len(conversation_states)} conversations")

        try:
            async with aiohttp.ClientSession() as session:
                # Create tasks for all API calls
                tasks = []
                for state in conversation_states:
                    task = self.get_auditor_response(session, state.auditor_messages)
                    tasks.append((state, task))

                # Execute all API calls concurrently
                completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                # Process responses with retry handling
                for (state, _), result in zip(tasks, completed_tasks):
                    if isinstance(result, Exception):
                        logger.error(f"OpenRouterWorker {self.worker_id}: Exception for conversation {state.conversation_id}: {result}")
                        # Put back to OpenRouter queue for retry
                        self.openrouter_queue.put(state)
                        continue

                    response_text, success = result

                    if not success:
                        logger.warning(f"API call failed for conversation {state.conversation_id}, requeueing")
                        # Put back to OpenRouter queue for retry
                        self.openrouter_queue.put(state)
                        continue

                    if not response_text:
                        logger.warning(f"Empty response for conversation {state.conversation_id}, requeueing")
                        # Put back to OpenRouter queue for retry
                        self.openrouter_queue.put(state)
                        continue

                    # Check for end conversation signal
                    if state.is_end_conversation_signal(response_text):
                        logger.info(f"Conversation {state.conversation_id} ended early by auditor")
                        state.full_conversation.append({"role": "user", "content": response_text})
                        self.completed_queue.put(state)

                        with self.progress_counter.get_lock():
                            self.progress_counter.value += 1
                    else:
                        # Update conversation state
                        state.auditor_messages.append({"role": "assistant", "content": response_text})
                        state.target_messages.append({"role": "user", "content": response_text})
                        state.full_conversation.append({"role": "user", "content": response_text})

                        # Send to vLLM queue for target response
                        self.vllm_queue.put(state)

        except Exception as e:
            logger.error(f"OpenRouterWorker {self.worker_id}: Error processing batch: {e}")
            # Put conversations back for retry
            for state in conversation_states:
                self.openrouter_queue.put(state)

    def process_batch(self, conversation_states: List[ConversationState]):
        """Synchronous wrapper for async batch processing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_batch_async(conversation_states))
        finally:
            loop.close()

    def run(self):
        """Main worker loop."""
        logger.info(f"OpenRouterWorker {self.worker_id}: Starting")

        try:
            while not self.should_stop:
                batch = self.collect_batch()
                if batch:
                    self.process_batch(batch)
                elif self.should_stop:
                    break
                else:
                    # Small sleep when no work available
                    time.sleep(0.01)

        except Exception as e:
            logger.error(f"OpenRouterWorker {self.worker_id}: Fatal error in run loop: {e}")
        finally:
            logger.info(f"OpenRouterWorker {self.worker_id}: Shutting down")

    def stop(self):
        """Signal worker to stop."""
        self.should_stop = True


class TranscriptSaver(multiprocessing.Process):
    """Worker process for saving completed conversation transcripts."""

    def __init__(self, completed_queue: multiprocessing.Queue, output_dir: str,
                 target_model_name: str, auditor_model_name: str):
        super().__init__()
        self.completed_queue = completed_queue
        self.output_dir = Path(output_dir)
        self.target_model_name = target_model_name
        self.auditor_model_name = auditor_model_name
        self.should_stop = False

    def generate_transcript_filename(self, data: Dict[str, Any]) -> str:
        """Generate transcript filename for given conversation data."""
        return f"{data['domain']}_persona{data['persona_id']}_topic{data['topic_id']}.json"

    def save_transcript(self, transcript: Dict[str, Any]):
        """Save conversation transcript to file."""
        try:
            # Create output directory structure
            output_path = self.output_dir

            # If using target prompts, organize by role in subdirectories
            if "target_role" in transcript:
                role_dir = output_path / transcript["target_role"]
                role_dir.mkdir(parents=True, exist_ok=True)
                output_path = role_dir
            else:
                output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename
            filename = self.generate_transcript_filename(transcript)
            file_path = output_path / filename

            # Save transcript
            with open(file_path, "w") as f:
                json.dump(transcript, f, indent=2)

            logger.info(f"TranscriptSaver: Saved transcript to {file_path}")

        except Exception as e:
            logger.error(f"TranscriptSaver: Error saving transcript: {e}")

    def run(self):
        """Main saver loop."""
        logger.info("TranscriptSaver: Starting")

        try:
            while not self.should_stop:
                try:
                    # Wait for completed conversations
                    conversation_state = self.completed_queue.get(timeout=1.0)
                    transcript = conversation_state.to_transcript(self.target_model_name, self.auditor_model_name)
                    self.save_transcript(transcript)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"TranscriptSaver: Error processing completed conversation: {e}")

        except Exception as e:
            logger.error(f"TranscriptSaver: Fatal error in run loop: {e}")
        finally:
            logger.info("TranscriptSaver: Shutting down")

    def stop(self):
        """Signal saver to stop."""
        self.should_stop = True


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Continue existing conversations between a target model (vLLM) and auditor model (OpenRouter)"
    )

    # Required arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing existing transcript JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for continued conversation transcripts"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="The target model that is treated as the Assistant; it will be spun up as a vLLM instance"
    )
    parser.add_argument(
        "--turns",
        type=int,
        required=True,
        help="Target number of conversation turn pairs to reach (each turn = user + assistant message pair)"
    )

    # Optional arguments
    parser.add_argument(
        "--auditor-model",
        type=str,
        default="moonshotai/kimi-k2-0905",
        help="The auditor model that will role-play as a user; it will be accessed via OpenRouter API"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the target model (default: 8192)"
    )
    parser.add_argument(
        "--num-openrouter-workers",
        type=int,
        default=10,
        help="Number of OpenRouter worker processes (default: 10)"
    )
    parser.add_argument(
        "--openrouter-batch-size",
        type=int,
        default=10,
        help="Batch size for OpenRouter workers (default: 10)"
    )
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=100,
        help="Maximum batch size for vLLM worker - will adapt down based on conversation length (default: 100)"
    )
    parser.add_argument(
        "--vllm-timeout",
        type=float,
        default=0.1,
        help="Timeout in seconds for vLLM batch collection (default: 0.1)"
    )
    parser.add_argument(
        "--openrouter-timeout",
        type=float,
        default=0.2,
        help="Timeout in seconds for OpenRouter batch collection (default: 0.2)"
    )

    return parser.parse_args()


def load_transcripts(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all transcript JSON files from input directory recursively.

    Args:
        input_dir: Directory containing transcript files

    Returns:
        List of transcript dictionaries with added 'file_path' field
    """
    transcripts = []

    # Find all JSON files recursively
    for json_file in input_dir.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                transcript = json.load(f)
                # Add file path for later reference
                transcript["file_path"] = str(json_file)
                transcripts.append(transcript)
        except Exception as e:
            logger.warning(f"Error loading transcript {json_file}: {e}")

    logger.info(f"Loaded {len(transcripts)} transcripts from {input_dir}")
    return transcripts


def filter_transcripts(transcripts: List[Dict[str, Any]], target_turns: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter transcripts into those needing continuation and those already complete.

    Args:
        transcripts: List of transcript dictionaries
        target_turns: Target number of turn pairs (each turn = user + assistant message pair)

    Returns:
        Tuple of (transcripts_to_continue, transcripts_already_complete)
    """
    to_continue = []
    already_complete = []

    # target_turns refers to turn pairs, so target message count is target_turns * 2
    target_message_count = target_turns * 2

    for transcript in transcripts:
        conversation = transcript.get("conversation", [])
        # Use actual conversation length instead of relying on turns field
        message_count = len(conversation)

        # Check if already at target message count
        if message_count >= target_message_count:
            already_complete.append(transcript)
            continue

        # Check if last message is END_CONVERSATION
        if conversation:
            last_message = conversation[-1].get("content", "").strip()
            if last_message == "END_CONVERSATION" or last_message == "<END_CONVERSATION>":
                to_continue.append(transcript)
            else:
                # Ended early but not with END_CONVERSATION signal
                already_complete.append(transcript)
        else:
            # Empty conversation, mark as complete
            already_complete.append(transcript)

    logger.info(f"Found {len(to_continue)} transcripts to continue, {len(already_complete)} already complete")
    return to_continue, already_complete


def reconstruct_conversation_state(transcript: Dict[str, Any], target_turns: int) -> ConversationState:
    """
    Reconstruct a ConversationState from a transcript.

    Args:
        transcript: Transcript dictionary
        target_turns: Target number of turn pairs (each turn = user + assistant message pair)

    Returns:
        ConversationState object ready to continue
    """
    # Extract persona data
    persona_data = {
        "domain": transcript["domain"],
        "persona_id": transcript["persona_id"],
        "persona": transcript["persona"],
        "topic_id": transcript["topic_id"],
        "topic": transcript["topic"]
    }

    # Generate auditor system prompt from template
    domain = transcript["domain"]
    if domain in ["coding", "writing"]:
        prompt_template = PROMPTS["generate_coding_writing_task"]
    elif domain in ["therapy", "philosophy"]:
        prompt_template = PROMPTS["generate_therapy_philosophy_task"]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    auditor_system_prompt = prompt_template.format(
        domain=domain,
        persona=transcript["persona"],
        topic=transcript["topic"]
    )

    # Extract target prompt if present
    target_prompt = None
    if "target_role" in transcript:
        target_prompt = {
            "role": transcript["target_role"],
            "prompt_index": transcript.get("target_prompt_index", 0),
            "system_prompt": transcript.get("target_prompt", "")
        }

    # Reconstruct message lists
    # Remove last message if it's END_CONVERSATION
    conversation = transcript["conversation"][:]
    if conversation and conversation[-1].get("content", "").strip() in ["END_CONVERSATION", "<END_CONVERSATION>"]:
        conversation = conversation[:-1]

    # Auditor sees conversation with roles flipped (as assistant)
    auditor_messages = [{"role": "system", "content": auditor_system_prompt}]
    for msg in conversation:
        if msg["role"] == "user":
            auditor_messages.append({"role": "assistant", "content": msg["content"]})
        else:  # assistant
            auditor_messages.append({"role": "user", "content": msg["content"]})

    # Target sees conversation with normal roles (as assistant)
    target_messages = []
    for msg in conversation:
        target_messages.append({"role": msg["role"], "content": msg["content"]})

    # Full conversation (no role flipping)
    full_conversation = conversation[:]

    # Create conversation ID
    conversation_id = f"{domain}_p{persona_data['persona_id']}_t{persona_data['topic_id']}_continue"

    # max_turns in ConversationState expects message count, not turn pairs
    max_message_count = target_turns * 2

    return ConversationState(
        persona_data=persona_data,
        target_prompt=target_prompt,
        auditor_messages=auditor_messages,
        target_messages=target_messages,
        full_conversation=full_conversation,
        current_turn=len(full_conversation),
        max_turns=max_message_count,
        conversation_id=conversation_id
    )


def copy_complete_transcripts(transcripts: List[Dict[str, Any]], output_dir: Path):
    """
    Copy already-complete transcripts to output directory.

    Args:
        transcripts: List of complete transcript dictionaries
        output_dir: Output directory
    """
    for transcript in transcripts:
        try:
            source_path = Path(transcript["file_path"])

            # Determine output path
            if "target_role" in transcript:
                role_dir = output_dir / transcript["target_role"]
                role_dir.mkdir(parents=True, exist_ok=True)
                output_path = role_dir
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir

            # Generate filename
            filename = f"{transcript['domain']}_persona{transcript['persona_id']}_topic{transcript['topic_id']}.json"
            dest_path = output_path / filename

            # Copy file
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied complete transcript to {dest_path}")

        except Exception as e:
            logger.error(f"Error copying transcript {transcript.get('file_path')}: {e}")


def signal_handler(signum, frame, workers):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")

    # Signal all workers to stop
    for worker in workers:
        if hasattr(worker, 'stop'):
            worker.stop()

    # Give workers time to finish current work
    time.sleep(2)

    # Force terminate if needed
    for worker in workers:
        if worker.is_alive():
            worker.terminate()

    sys.exit(0)


def main():
    args = parse_arguments()

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Load transcripts
    transcripts = load_transcripts(input_dir)
    if not transcripts:
        logger.error("No transcripts found in input directory")
        sys.exit(1)

    # Filter transcripts
    to_continue, already_complete = filter_transcripts(transcripts, args.turns)

    if len(to_continue) == 0:
        logger.info("No transcripts need continuation")
        if len(already_complete) > 0:
            logger.info(f"Copying {len(already_complete)} already-complete transcripts to output directory")
            output_dir = Path(args.output_dir)
            copy_complete_transcripts(already_complete, output_dir)
        return

    logger.info(f"Will continue {len(to_continue)} conversations to {args.turns} turns")

    # Copy already-complete transcripts to output directory
    if already_complete:
        output_dir = Path(args.output_dir)
        logger.info(f"Copying {len(already_complete)} already-complete transcripts")
        copy_complete_transcripts(already_complete, output_dir)

    # Reconstruct conversation states
    conversation_states = []
    for transcript in to_continue:
        try:
            state = reconstruct_conversation_state(transcript, args.turns)
            conversation_states.append(state)
        except Exception as e:
            logger.error(f"Error reconstructing conversation from {transcript.get('file_path')}: {e}")

    if not conversation_states:
        logger.error("No valid conversation states to continue")
        sys.exit(1)

    logger.info(f"Successfully reconstructed {len(conversation_states)} conversation states")

    # Create multiprocessing queues
    openrouter_queue = multiprocessing.Queue()
    vllm_queue = multiprocessing.Queue()
    completed_queue = multiprocessing.Queue()

    # Progress tracking
    total_conversations = len(conversation_states)
    progress_counter = multiprocessing.Value('i', 0)

    # Populate initial OpenRouter queue (auditor continues conversations)
    for state in conversation_states:
        openrouter_queue.put(state)

    logger.info(f"Added {len(conversation_states)} conversations to OpenRouter queue")

    # Create and start workers
    workers = []

    # Set up signal handling for graceful shutdown
    def signal_handler_wrapper(signum, frame):
        signal_handler(signum, frame, workers)

    signal.signal(signal.SIGINT, signal_handler_wrapper)
    signal.signal(signal.SIGTERM, signal_handler_wrapper)

    try:
        # Start vLLM worker
        vllm_worker = VLLMWorker(
            target_model_name=args.target_model,
            max_model_len=args.max_model_len,
            vllm_queue=vllm_queue,
            openrouter_queue=openrouter_queue,
            completed_queue=completed_queue,
            progress_counter=progress_counter,
            batch_size=args.vllm_batch_size,
            timeout=args.vllm_timeout
        )
        vllm_worker.start()
        workers.append(vllm_worker)
        logger.info("Started vLLM worker")

        # Start OpenRouter workers
        for i in range(args.num_openrouter_workers):
            worker = OpenRouterWorker(
                worker_id=i,
                auditor_model_name=args.auditor_model,
                openrouter_queue=openrouter_queue,
                vllm_queue=vllm_queue,
                completed_queue=completed_queue,
                progress_counter=progress_counter,
                batch_size=args.openrouter_batch_size,
                timeout=args.openrouter_timeout
            )
            worker.start()
            workers.append(worker)
        logger.info(f"Started {args.num_openrouter_workers} OpenRouter workers")

        # Start transcript saver
        saver = TranscriptSaver(
            completed_queue=completed_queue,
            output_dir=args.output_dir,
            target_model_name=args.target_model,
            auditor_model_name=args.auditor_model
        )
        saver.start()
        workers.append(saver)
        logger.info("Started transcript saver")

        # Monitor progress
        start_time = time.time()
        last_progress = 0

        while True:
            time.sleep(5)  # Check progress every 5 seconds

            current_progress = progress_counter.value
            if current_progress != last_progress:
                elapsed_time = time.time() - start_time
                rate = current_progress / elapsed_time if elapsed_time > 0 else 0
                eta = (total_conversations - current_progress) / rate if rate > 0 else 0

                logger.info(f"Progress: {current_progress}/{total_conversations} conversations completed "
                           f"({current_progress/total_conversations*100:.1f}%) - "
                           f"Rate: {rate:.2f} conversations/sec - "
                           f"ETA: {eta:.0f} seconds")
                last_progress = current_progress

            # Check if all conversations are complete
            if current_progress >= total_conversations:
                logger.info("All conversations completed!")
                break

            # Check if workers are still alive and detect dead workers
            alive_workers = [w for w in workers if w.is_alive()]
            if len(alive_workers) < len(workers):
                dead_workers = [w for w in workers if not w.is_alive()]
                logger.warning(f"Some workers have died: {len(alive_workers)}/{len(workers)} still alive")

                # Log details of dead workers
                for dead_worker in dead_workers:
                    logger.error(f"Dead worker: {dead_worker.__class__.__name__} - "
                               f"PID: {dead_worker.pid} - "
                               f"Exit code: {dead_worker.exitcode}")

                # Check if we have too many dead workers to continue
                if len(alive_workers) < len(workers) * 0.5:  # Less than 50% alive
                    logger.error("Too many workers have died. Shutting down...")
                    break

        # Give some time for final transcript saves
        time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        # Shutdown all workers
        logger.info("Shutting down workers...")
        for worker in workers:
            if hasattr(worker, 'stop'):
                worker.stop()

        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker}")
                worker.terminate()

        logger.info("All workers shut down successfully")

        # Final statistics
        final_progress = progress_counter.value
        total_time = time.time() - start_time
        logger.info(f"Final statistics: {final_progress}/{total_conversations} conversations completed "
                   f"in {total_time:.2f} seconds "
                   f"(average: {final_progress/total_time:.2f} conversations/sec)")


if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
