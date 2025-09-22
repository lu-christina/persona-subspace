import argparse
import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import openai

from dotenv import load_dotenv

# Add the parent directory to sys.path to import prompts and utils
sys.path.append(str(Path(__file__).parent / "data"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from prompts import PROMPTS
from inference_utils import load_vllm_model, chat_with_model, close_vllm_model

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_PERSONAS_FILE = "/root/git/persona-subspace/dynamics/data/personas/personas.jsonl"
DEFAULT_OUTPUT_DIR = "/root/git/persona-subspace/dynamics/results"

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run conversations between a target model (vLLM) and auditor model (OpenRouter)"
    )
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
        help="Number of conversation turns (default: 30)"
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

class ConversationRunner:
    """Manages conversations between target and auditor models."""

    def __init__(self, target_model_name: str, auditor_model_name: str, output_dir: str):
        self.target_model_name = target_model_name
        self.auditor_model_name = auditor_model_name
        self.output_dir = Path(output_dir)
        self.target_model = None
        self.auditor_client = None

    def is_end_conversation_signal(self, message: str) -> bool:
        """Check if the message contains an end conversation signal."""
        message_clean = message.strip()
        return message_clean == 'END_CONVERSATION' or message_clean == '<END_CONVERSATION>'

    def generate_transcript_filename(self, data: Dict[str, Any]) -> str:
        """Generate transcript filename for given persona data or transcript."""
        return f"{data['domain']}_persona{data['persona_id']}_topic{data['topic_id']}.json"

    def check_existing_transcripts(self, personas_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out persona data that already have existing transcripts."""
        missing_personas = []
        existing_count = 0

        for persona_data in personas_data:
            filename = self.generate_transcript_filename(persona_data)
            file_path = self.output_dir / filename

            if file_path.exists():
                existing_count += 1
                logger.info(f"Skipping existing transcript: {filename}")
            else:
                missing_personas.append(persona_data)

        logger.info(f"Found {existing_count} existing transcripts, {len(missing_personas)} to process")
        return missing_personas

    def setup_models(self):
        """Initialize both target and auditor models."""
        logger.info(f"Loading target model: {self.target_model_name}")
        self.target_model = load_vllm_model(self.target_model_name, max_model_len=8192)

        logger.info(f"Setting up auditor model: {self.auditor_model_name}")
        self.auditor_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def cleanup_models(self):
        """Clean up model resources."""
        if self.target_model:
            close_vllm_model(self.target_model)
            self.target_model = None

    def get_auditor_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from auditor model via OpenRouter."""
        try:
            response = self.auditor_client.chat.completions.create(
                model=self.auditor_model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting auditor response: {e}")
            raise

    def get_target_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from target model via vLLM (no system prompt, no thinking)."""
        try:
            # Filter out system messages for target model
            filtered_messages = [msg for msg in messages if msg["role"] != "system"]

            if not filtered_messages:
                raise ValueError("No messages to respond to")

            # Get the latest user message
            latest_message = filtered_messages[-1]["content"]

            # Build conversation history without system prompts
            history = filtered_messages[:-1] if len(filtered_messages) > 1 else None

            return chat_with_model(
                self.target_model,
                latest_message,
                conversation_history=history,
                enable_thinking=False,  # Explicitly disable thinking
                system_prompt=None,     # No system prompt
                temperature=0.7,
                max_tokens=2048,
                top_p=0.8,
                top_k=20,
                min_p=0
            )
        except Exception as e:
            logger.error(f"Error getting target response: {e}")
            raise

    def run_conversation(self, persona_data: Dict[str, Any], num_turns: int) -> Dict[str, Any]:
        """Run a single conversation between auditor and target."""
        logger.info(f"Starting conversation for persona {persona_data['persona_id']}, topic {persona_data['topic_id']}")

        # Initialize conversation contexts
        auditor_messages = [{"role": "system", "content": persona_data["system_prompt"]}]
        target_messages = []
        full_conversation = []

        # Auditor starts the conversation
        try:
            # Get initial message from auditor
            auditor_response = self.get_auditor_response(auditor_messages)

            # Check if auditor wants to end conversation immediately
            if self.is_end_conversation_signal(auditor_response):
                logger.info("Auditor sent end conversation signal on first turn, ending conversation")
                full_conversation.append({"role": "user", "content": auditor_response})
                # Prepare early exit transcript
                transcript = {
                    "model": self.target_model_name,
                    "auditor_model": self.auditor_model_name,
                    "domain": persona_data["domain"],
                    "persona_id": persona_data["persona_id"],
                    "persona": persona_data["persona"],
                    "topic_id": persona_data["topic_id"],
                    "topic": persona_data["topic"],
                    "turns": len(full_conversation),
                    "conversation": full_conversation,
                    "ended_early": True
                }
                return transcript

            # Add to conversation contexts
            auditor_messages.append({"role": "assistant", "content": auditor_response})
            target_messages.append({"role": "user", "content": auditor_response})
            full_conversation.append({"role": "user", "content": auditor_response})

            # Alternate between target and auditor for remaining turns
            for turn in range(num_turns - 1):
                logger.info(f"Turn {turn + 2}/{num_turns}")

                # Target responds
                target_response = self.get_target_response(target_messages)
                target_messages.append({"role": "assistant", "content": target_response})
                auditor_messages.append({"role": "user", "content": target_response})
                full_conversation.append({"role": "assistant", "content": target_response})

                # If this is the last turn, stop here
                if turn + 2 >= num_turns:
                    break

                # Auditor responds
                auditor_response = self.get_auditor_response(auditor_messages)
                auditor_messages.append({"role": "assistant", "content": auditor_response})
                target_messages.append({"role": "user", "content": auditor_response})
                full_conversation.append({"role": "user", "content": auditor_response})

                # Check if auditor wants to end conversation
                if self.is_end_conversation_signal(auditor_response):
                    logger.info(f"Auditor sent end conversation signal on turn {turn + 3}, ending conversation early")
                    break

        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            # Return partial conversation if something went wrong

        # Prepare transcript
        transcript = {
            "model": self.target_model_name,
            "auditor_model": self.auditor_model_name,
            "domain": persona_data["domain"],
            "persona_id": persona_data["persona_id"],
            "persona": persona_data["persona"],
            "topic_id": persona_data["topic_id"],
            "topic": persona_data["topic"],
            "turns": len(full_conversation),
            "conversation": full_conversation
        }

        # Check if conversation ended early due to end signal
        if full_conversation and self.is_end_conversation_signal(full_conversation[-1]["content"]):
            transcript["ended_early"] = True

        return transcript

    def save_transcript(self, transcript: Dict[str, Any]):
        """Save conversation transcript to file."""
        # Create output directory structure
        output_path = self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename using the helper method
        filename = self.generate_transcript_filename(transcript)
        file_path = output_path / filename

        # Save transcript
        with open(file_path, "w") as f:
            json.dump(transcript, f, indent=2)

        logger.info(f"Saved transcript to {file_path}")

def main():
    args = parse_arguments()

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)

    # Load personas and topics
    personas_data = load_personas_and_topics(args.personas_file)
    logger.info(f"Loaded {len(personas_data)} conversation topics")

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

    # Initialize conversation runner
    runner = ConversationRunner(args.target_model, args.auditor_model, args.output_dir)

    # Check for existing transcripts unless overwrite is specified
    if not args.overwrite:
        logger.info("Checking for existing transcripts...")
        personas_data = runner.check_existing_transcripts(personas_data)
        if len(personas_data) == 0:
            logger.info("All conversations already completed. Use --overwrite to re-run.")
            return
    else:
        logger.info("Overwrite mode: will re-run all conversations")

    try:
        # Setup models
        runner.setup_models()

        # Run conversations
        for i, persona_data in enumerate(personas_data):
            logger.info(f"Running conversation {i+1}/{len(personas_data)}")

            try:
                transcript = runner.run_conversation(persona_data, args.num_turns)
                runner.save_transcript(transcript)
            except Exception as e:
                logger.error(f"Failed to run conversation {i+1}: {e}")
                continue

        logger.info("All conversations completed")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        runner.cleanup_models()


if __name__ == "__main__":
    main()