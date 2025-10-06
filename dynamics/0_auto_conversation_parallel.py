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

DEFAULT_PERSONAS_FILE = "/root/git/persona-subspace/dynamics/data/personas/personas.jsonl"
DEFAULT_OUTPUT_DIR = "/root/git/persona-subspace/dynamics/results"


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
                    logger.warning(f"Empty response for conversation {state.conversation_id}, assuming max model length reached")
                    # Mark conversation as complete (will set ended_early in transcript)
                    self.completed_queue.put(state)
                    with self.progress_counter.get_lock():
                        self.progress_counter.value += 1
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
        description="Run parallel conversations between a target model (vLLM) and auditor model (OpenRouter)"
    )

    # Original arguments from 0_auto_conversation.py
    parser.add_argument(
        "--personas-file",
        type=str,
        default=DEFAULT_PERSONAS_FILE,
        help="Input JSONL file containing personas and conversation topics"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="The target model that is treated as the Assistant; it will be spun up as a vLLM instance"
    )
    parser.add_argument(
        "--auditor-model",
        type=str,
        default="moonshotai/kimi-k2-0905",
        help="The auditor model that will role-play as a user; it will be accessed via OpenRouter API"
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=15,
        help="Number of conversation turns (default: 15)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run only the first conversation topic for testing"
    )
    parser.add_argument(
        "--interval",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Run conversations for persona_data in range [START:END] (start inclusive, end exclusive)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript files instead of skipping them"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for conversation transcripts"
    )
    parser.add_argument(
        "--target-prompts",
        type=str,
        help="JSONL file containing target prompts with role and system content to initialize target model"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the target model (default: 8192)"
    )

    # New parallel processing arguments
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
        default=200,
        help="Maximum batch size for vLLM worker - will adapt down based on conversation length (default: 200)"
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


def load_personas_and_topics(personas_file: str) -> List[Dict[str, Any]]:
    """
    Load personas and topics from file.
    """
    personas_data = []
    prompt_template_coding_writing = PROMPTS["generate_coding_writing_task"]
    prompt_template_therapy_philosophy = PROMPTS["generate_therapy_philosophy_task"]

    with open(personas_file, "r") as f:
        for line in f:
            persona_data = json.loads(line)
            domain = persona_data["domain"]
            persona = persona_data["persona"]
            topics = persona_data["topics"]
            persona_id = persona_data["id"]

            for i, topic in enumerate(topics):
                if domain == "coding" or domain == "writing":
                    prompt_template = prompt_template_coding_writing
                elif domain == "therapy" or domain == "philosophy":
                    prompt_template = prompt_template_therapy_philosophy
                else:
                    raise ValueError(f"Invalid domain: {domain}")

                system_prompt = prompt_template.format(domain=domain, persona=persona, topic=topic)
                personas_data.append({
                    "persona_id": persona_id,
                    "topic_id": i,
                    "domain": domain,
                    "persona": persona,
                    "topic": topic,
                    "system_prompt": system_prompt
                })

    return personas_data


def load_target_prompts(target_prompts_file: str) -> List[Dict[str, Any]]:
    """
    Load target prompts from JSONL file.

    Args:
        target_prompts_file: Path to JSONL file containing target prompts

    Returns:
        List of target prompt dictionaries with role, prompt_index, and system content
    """
    target_prompts = []

    with open(target_prompts_file, "r") as f:
        for line in f:
            prompt_data = json.loads(line)

            # Extract system prompt from conversation
            system_prompt = None
            for message in prompt_data["conversation"]:
                if message["role"] == "system":
                    system_prompt = message["content"]
                    break

            if system_prompt:
                target_prompts.append({
                    "role": prompt_data["role"],
                    "prompt_index": prompt_data["prompt_index"],
                    "system_prompt": system_prompt
                })

    return target_prompts


def generate_transcript_filename(data: Dict[str, Any]) -> str:
    """Generate transcript filename for given persona data."""
    return f"{data['domain']}_persona{data['persona_id']}_topic{data['topic_id']}.json"


def check_existing_transcripts(personas_data: List[Dict[str, Any]],
                             target_prompts: Optional[List[Dict[str, Any]]],
                             output_dir: Path):
    """Filter out persona data that already have existing transcripts."""
    missing_combinations = []
    existing_count = 0

    if target_prompts:
        # When using target prompts, check each persona + target_prompt combination
        total_combinations = len(personas_data) * len(target_prompts)

        for persona_data in personas_data:
            for target_prompt in target_prompts:
                # Build path with role subdirectory
                role_dir = output_dir / target_prompt["role"]
                filename = generate_transcript_filename(persona_data)
                file_path = role_dir / filename

                if file_path.exists():
                    existing_count += 1
                    logger.info(f"Skipping existing transcript: {target_prompt['role']}/{filename}")
                else:
                    missing_combinations.append((persona_data, target_prompt))

        logger.info(f"Found {existing_count} existing transcripts, {len(missing_combinations)} combinations to process (out of {total_combinations} total)")
    else:
        # Original behavior when no target prompts provided
        for persona_data in personas_data:
            filename = generate_transcript_filename(persona_data)
            file_path = output_dir / filename

            if file_path.exists():
                existing_count += 1
                logger.info(f"Skipping existing transcript: {filename}")
            else:
                missing_combinations.append((persona_data, None))

        logger.info(f"Found {existing_count} existing transcripts, {len(missing_combinations)} to process")

    return missing_combinations


def create_initial_conversation_states(missing_combinations: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
                                     num_turns: int) -> List[ConversationState]:
    """Create initial conversation states from missing combinations."""
    conversation_states = []

    for i, (persona_data, target_prompt) in enumerate(missing_combinations):
        conversation_id = f"{persona_data['domain']}_p{persona_data['persona_id']}_t{persona_data['topic_id']}_{i}"

        # Initialize conversation state
        auditor_messages = [{"role": "system", "content": persona_data["system_prompt"]}]
        target_messages = []
        full_conversation = []

        conversation_state = ConversationState(
            persona_data=persona_data,
            target_prompt=target_prompt,
            auditor_messages=auditor_messages,
            target_messages=target_messages,
            full_conversation=full_conversation,
            current_turn=0,
            max_turns=num_turns,
            conversation_id=conversation_id
        )

        conversation_states.append(conversation_state)

    return conversation_states


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

    # Load personas and topics
    personas_data = load_personas_and_topics(args.personas_file)
    logger.info(f"Loaded {len(personas_data)} conversation topics")

    # Load target prompts if provided
    target_prompts = None
    if args.target_prompts:
        target_prompts = load_target_prompts(args.target_prompts)
        logger.info(f"Loaded {len(target_prompts)} target prompts from {args.target_prompts}")

    # Filter for interval if specified
    if args.interval:
        start, end = args.interval
        if start < 0 or end < 0:
            logger.error("Interval indices must be non-negative")
            sys.exit(1)
        if start >= end:
            logger.error("Interval start must be less than end")
            sys.exit(1)
        if start >= len(personas_data):
            logger.error(f"Interval start ({start}) is beyond available data ({len(personas_data)} topics)")
            sys.exit(1)

        # Apply interval filtering
        end = min(end, len(personas_data))  # Cap end at available data length
        personas_data = personas_data[start:end]
        logger.info(f"Interval mode: running conversations {start} to {end-1} ({len(personas_data)} topics)")

    # Filter for test mode
    if args.test_mode:
        personas_data = personas_data[:1]
        logger.info("Test mode: running only first conversation topic")

    # Check for existing transcripts unless overwrite is specified
    output_dir = Path(args.output_dir)
    if not args.overwrite:
        logger.info("Checking for existing transcripts...")
        missing_combinations = check_existing_transcripts(personas_data, target_prompts, output_dir)
        if len(missing_combinations) == 0:
            logger.info("All conversations already completed. Use --overwrite to re-run.")
            return
    else:
        logger.info("Overwrite mode: will re-run all conversations")
        # Build all combinations for overwrite mode
        if target_prompts:
            missing_combinations = [(persona_data, target_prompt)
                                  for persona_data in personas_data
                                  for target_prompt in target_prompts]
        else:
            missing_combinations = [(persona_data, None) for persona_data in personas_data]

    logger.info(f"Will process {len(missing_combinations)} conversations")

    # Create initial conversation states
    conversation_states = create_initial_conversation_states(missing_combinations, args.num_turns)

    # Create multiprocessing queues
    openrouter_queue = multiprocessing.Queue()
    vllm_queue = multiprocessing.Queue()
    completed_queue = multiprocessing.Queue()

    # Progress tracking
    total_conversations = len(conversation_states)
    progress_counter = multiprocessing.Value('i', 0)

    # Populate initial OpenRouter queue (auditor starts conversations)
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