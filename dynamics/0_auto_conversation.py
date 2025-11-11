import argparse
import os
import sys
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
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


@dataclass
class PersonaInput:
    """Input configuration for persona-based conversations"""
    input_type: str
    domain: str
    persona_id: int
    persona: str
    topic_id: int
    topic: str
    system_prompt: str


@dataclass
class InstructionInput:
    """Input configuration for instruction-file-based conversations"""
    input_type: str
    system_prompt: str        # auditor system prompt
    first_message: str        # content from instruction file
    instruction: str          # copy for output metadata
    source_file: str          # basename like "isolation" for filename


def strip_internal_state(message: str) -> str:
    """
    Remove <INTERNAL_STATE>...</INTERNAL_STATE> tags from a message.

    Args:
        message: The message potentially containing internal state tags

    Returns:
        The message with internal state tags and their content removed
    """
    # Remove internal state tags and everything between them
    pattern = r'<INTERNAL_STATE>.*?</INTERNAL_STATE>'
    cleaned = re.sub(pattern, '', message, flags=re.DOTALL)
    # Clean up any extra whitespace that might be left
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()


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
    parser.add_argument(
        "--target-prompts",
        type=str,
        help="JSONL file containing target prompts with role and system content to initialize target model"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="Maximum sequence length for the target model (default: 8192)"
    )
    parser.add_argument(
        "--instruction-file",
        type=str,
        help="Text file containing instruction for the first user message (mutually exclusive with --personas-file)"
    )
    parser.add_argument(
        "--auditor-system-prompt",
        type=str,
        default="red_team_system_prompt",
        help="Name of the system prompt from prompts.py to use for auditor model (default: red_team_system_prompt)"
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

def load_instruction_file_inputs(instruction_file: str, system_prompt_name: str = "red_team_system_prompt") -> List[InstructionInput]:
    """
    Load instruction file and create input configuration.

    Args:
        instruction_file: Path to text file containing the first user message
        system_prompt_name: Name of the system prompt in PROMPTS dict

    Returns:
        List containing a single InstructionInput (list for future extensibility)
    """
    # Read the instruction file
    with open(instruction_file, "r") as f:
        first_message = f.read()

    # Get system prompt from PROMPTS
    if system_prompt_name not in PROMPTS:
        raise ValueError(f"System prompt '{system_prompt_name}' not found in PROMPTS")
    system_prompt = PROMPTS[system_prompt_name]

    # Extract source file basename (without extension)
    source_file = Path(instruction_file).stem

    # Create InstructionInput
    instruction_input = InstructionInput(
        input_type="instruction",
        system_prompt=system_prompt,
        first_message=first_message,
        instruction=first_message,  # Copy for metadata
        source_file=source_file
    )

    return [instruction_input]

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

class ConversationRunner:
    """Manages conversations between target and auditor models."""

    def __init__(self, target_model_name: str, auditor_model_name: str, output_dir: str, target_prompts: Optional[List[Dict[str, Any]]] = None, max_model_len: int = 8192):
        self.target_model_name = target_model_name
        self.auditor_model_name = auditor_model_name
        self.output_dir = Path(output_dir)
        self.target_prompts = target_prompts
        self.max_model_len = max_model_len
        self.target_model = None
        self.auditor_client = None

    def is_end_conversation_signal(self, message: str) -> bool:
        """Check if the message contains an end conversation signal."""
        return 'END_CONVERSATION' in message or '<END_CONVERSATION>' in message

    def generate_transcript_filename(self, data: Union[Dict[str, Any], PersonaInput, InstructionInput]) -> str:
        """Generate transcript filename for given input data or transcript."""
        # Handle dataclass inputs
        if isinstance(data, InstructionInput):
            return f"{data.source_file}.json"
        elif isinstance(data, PersonaInput):
            return f"{data.domain}_persona{data.persona_id}_topic{data.topic_id}.json"
        # Handle dict (for transcript dicts)
        elif isinstance(data, dict):
            if 'domain' in data:
                # Persona-based transcript
                return f"{data['domain']}_persona{data['persona_id']}_topic{data['topic_id']}.json"
            elif 'source_file' in data:
                # Instruction-based transcript
                return f"{data['source_file']}.json"
        raise ValueError(f"Unknown data type for filename generation: {type(data)}")

    def check_existing_transcripts(self, inputs_data: List[Union[PersonaInput, InstructionInput]], target_prompts: Optional[List[Dict[str, Any]]] = None):
        """Filter out input data (and target prompts) that already have existing transcripts."""
        missing_combinations = []
        existing_count = 0

        # Note: target_prompts are only used with PersonaInput, not InstructionInput
        if target_prompts:
            # When using target prompts, check each persona + target_prompt combination
            total_combinations = len(inputs_data) * len(target_prompts)

            for input_data in inputs_data:
                if isinstance(input_data, InstructionInput):
                    # InstructionInput doesn't support target_prompts pathway
                    logger.warning("InstructionInput does not support target_prompts, skipping")
                    continue

                for target_prompt in target_prompts:
                    # Build path with role subdirectory
                    role_dir = self.output_dir / target_prompt["role"]
                    filename = self.generate_transcript_filename(input_data)
                    file_path = role_dir / filename

                    if file_path.exists():
                        existing_count += 1
                        logger.info(f"Skipping existing transcript: {target_prompt['role']}/{filename}")
                    else:
                        missing_combinations.append((input_data, target_prompt))

            logger.info(f"Found {existing_count} existing transcripts, {len(missing_combinations)} combinations to process (out of {total_combinations} total)")
        else:
            # Original behavior when no target prompts provided
            for input_data in inputs_data:
                filename = self.generate_transcript_filename(input_data)
                file_path = self.output_dir / filename

                if file_path.exists():
                    existing_count += 1
                    logger.info(f"Skipping existing transcript: {filename}")
                else:
                    missing_combinations.append((input_data, None))

            logger.info(f"Found {existing_count} existing transcripts, {len(missing_combinations)} to process")

        return missing_combinations

    def setup_models(self):
        """Initialize both target and auditor models."""
        logger.info(f"Loading target model: {self.target_model_name}")
        self.target_model = load_vllm_model(self.target_model_name, max_model_len=self.max_model_len)

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

    def get_target_response(self, messages: List[Dict[str, str]], target_system_prompt: Optional[str] = None) -> str:
        """Get response from target model via vLLM with optional target system prompt."""
        try:
            # Handle Gemma models specially - they don't support system prompts
            is_gemma = self.target_model_name.startswith("google/gemma-2")

            if is_gemma and target_system_prompt:
                # For Gemma models, append system prompt to first user message
                modified_messages = []
                first_user_found = False

                for msg in messages:
                    if msg["role"] == "user" and not first_user_found:
                        # Prepend system prompt to first user message
                        modified_content = f"{target_system_prompt}\n\n{msg['content']}"
                        modified_messages.append({"role": "user", "content": modified_content})
                        first_user_found = True
                    elif msg["role"] != "system":  # Skip system messages for Gemma
                        modified_messages.append(msg)

                filtered_messages = modified_messages
                system_prompt_for_model = None
            else:
                # For non-Gemma models, use system prompt normally but filter out existing system messages
                filtered_messages = [msg for msg in messages if msg["role"] != "system"]
                system_prompt_for_model = target_system_prompt

            if not filtered_messages:
                raise ValueError("No messages to respond to")

            # Get the latest user message
            latest_message = filtered_messages[-1]["content"]

            # Build conversation history
            history = filtered_messages[:-1] if len(filtered_messages) > 1 else None

            return chat_with_model(
                self.target_model,
                latest_message,
                conversation_history=history,
                enable_thinking=False,  # Explicitly disable thinking
                system_prompt=system_prompt_for_model,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.8,
                top_k=20,
                min_p=0
            )
        except Exception as e:
            logger.error(f"Error getting target response: {e}")
            raise

    def run_conversation(self, input_data: Union[PersonaInput, InstructionInput], num_turns: int, target_prompt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a single conversation between auditor and target."""
        # Log based on input type
        if isinstance(input_data, PersonaInput):
            logger.info(f"Starting conversation for persona {input_data.persona_id}, topic {input_data.topic_id}")
        elif isinstance(input_data, InstructionInput):
            logger.info(f"Starting conversation for instruction file: {input_data.source_file}")

        if target_prompt:
            logger.info(f"Using target prompt: {target_prompt['role']} (index {target_prompt['prompt_index']})")

        # Initialize conversation contexts
        auditor_messages = [{"role": "system", "content": input_data.system_prompt}]
        target_messages = []
        full_conversation = []

        # Extract target system prompt if available
        target_system_prompt = target_prompt["system_prompt"] if target_prompt else None

        # For InstructionInput, add the first message from the file to auditor context
        if isinstance(input_data, InstructionInput):
            auditor_messages.append({"role": "user", "content": input_data.first_message})

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
                    "turns": len(full_conversation),
                    "conversation": full_conversation,
                    "ended_early": True
                }

                # Add input-specific fields
                if isinstance(input_data, PersonaInput):
                    transcript.update({
                        "domain": input_data.domain,
                        "persona_id": input_data.persona_id,
                        "persona": input_data.persona,
                        "topic_id": input_data.topic_id,
                        "topic": input_data.topic
                    })
                elif isinstance(input_data, InstructionInput):
                    transcript.update({
                        "instruction": input_data.instruction,
                        "source_file": input_data.source_file
                    })

                # Add target prompt metadata if available
                if target_prompt:
                    transcript["target_role"] = target_prompt["role"]
                    transcript["target_prompt_index"] = target_prompt["prompt_index"]
                    transcript["target_prompt"] = target_prompt["system_prompt"]

                return transcript

            # Add to conversation contexts
            auditor_messages.append({"role": "assistant", "content": auditor_response})
            # Strip internal state tags before sending to target
            target_messages.append({"role": "user", "content": strip_internal_state(auditor_response)})
            full_conversation.append({"role": "user", "content": auditor_response})

            # Alternate between target and auditor for remaining turns
            for turn in range(num_turns - 1):
                logger.info(f"Turn {turn + 2}/{num_turns}")

                # Target responds (with optional target system prompt)
                target_response = self.get_target_response(target_messages, target_system_prompt)
                target_messages.append({"role": "assistant", "content": target_response})
                auditor_messages.append({"role": "user", "content": target_response})
                full_conversation.append({"role": "assistant", "content": target_response})

                # If this is the last turn, stop here
                if turn + 2 >= num_turns:
                    break

                # Auditor responds
                auditor_response = self.get_auditor_response(auditor_messages)
                auditor_messages.append({"role": "assistant", "content": auditor_response})
                # Strip internal state tags before sending to target
                target_messages.append({"role": "user", "content": strip_internal_state(auditor_response)})
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
            "turns": len(full_conversation),
            "conversation": full_conversation
        }

        # Add input-specific fields
        if isinstance(input_data, PersonaInput):
            transcript.update({
                "domain": input_data.domain,
                "persona_id": input_data.persona_id,
                "persona": input_data.persona,
                "topic_id": input_data.topic_id,
                "topic": input_data.topic
            })
        elif isinstance(input_data, InstructionInput):
            transcript.update({
                "instruction": input_data.instruction,
                "source_file": input_data.source_file
            })

        # Add target prompt metadata if available
        if target_prompt:
            transcript["target_role"] = target_prompt["role"]
            transcript["target_prompt_index"] = target_prompt["prompt_index"]
            transcript["target_prompt"] = target_prompt["system_prompt"]

        # Check if conversation ended early due to end signal
        if full_conversation and self.is_end_conversation_signal(full_conversation[-1]["content"]):
            transcript["ended_early"] = True

        return transcript

    def save_transcript(self, transcript: Dict[str, Any]):
        """Save conversation transcript to file."""
        # Create output directory structure
        output_path = self.output_dir

        # If using target prompts, organize by role in subdirectories
        if "target_role" in transcript:
            role_dir = output_path / transcript["target_role"]
            role_dir.mkdir(parents=True, exist_ok=True)
            output_path = role_dir
        else:
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

    # Check for mutual exclusivity between --personas-file and --instruction-file
    if args.instruction_file and args.personas_file != DEFAULT_PERSONAS_FILE:
        logger.error("Cannot specify both --instruction-file and --personas-file")
        sys.exit(1)

    # Load inputs based on mode
    inputs_data: List[Union[PersonaInput, InstructionInput]]
    if args.instruction_file:
        # Instruction-file mode
        inputs_data = load_instruction_file_inputs(args.instruction_file, args.auditor_system_prompt)
        logger.info(f"Loaded instruction file: {args.instruction_file}")

        # Warn if target_prompts is specified (not supported for instruction mode)
        if args.target_prompts:
            logger.warning("--target-prompts is not supported with --instruction-file, ignoring")
            args.target_prompts = None
    else:
        # Persona mode (default)
        personas_dict_list = load_personas_and_topics(args.personas_file)
        logger.info(f"Loaded {len(personas_dict_list)} conversation topics")

        # Convert dict list to PersonaInput list
        inputs_data = [
            PersonaInput(
                input_type="persona",
                domain=p["domain"],
                persona_id=p["persona_id"],
                persona=p["persona"],
                topic_id=p["topic_id"],
                topic=p["topic"],
                system_prompt=p["system_prompt"]
            )
            for p in personas_dict_list
        ]

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
        if start >= len(inputs_data):
            logger.error(f"Interval start ({start}) is beyond available data ({len(inputs_data)} inputs)")
            sys.exit(1)

        # Apply interval filtering
        end = min(end, len(inputs_data))  # Cap end at available data length
        inputs_data = inputs_data[start:end]
        logger.info(f"Interval mode: running conversations {start} to {end-1} ({len(inputs_data)} inputs)")

    # Filter for test mode
    if args.test_mode:
        inputs_data = inputs_data[:1]
        logger.info("Test mode: running only first conversation input")

    # Initialize conversation runner
    runner = ConversationRunner(args.target_model, args.auditor_model, args.output_dir, target_prompts, args.max_model_len)

    # Check for existing transcripts unless overwrite is specified
    if not args.overwrite:
        logger.info("Checking for existing transcripts...")
        missing_combinations = runner.check_existing_transcripts(inputs_data, target_prompts)
        if len(missing_combinations) == 0:
            logger.info("All conversations already completed. Use --overwrite to re-run.")
            return
    else:
        logger.info("Overwrite mode: will re-run all conversations")
        # Build all combinations for overwrite mode
        if target_prompts:
            missing_combinations = [(input_data, target_prompt)
                                  for input_data in inputs_data
                                  for target_prompt in target_prompts]
        else:
            missing_combinations = [(input_data, None) for input_data in inputs_data]

    try:
        # Setup models
        runner.setup_models()

        # Run conversations from missing combinations
        total_conversations = len(missing_combinations)
        conversation_count = 0

        for input_data, target_prompt in missing_combinations:
            conversation_count += 1
            logger.info(f"Running conversation {conversation_count}/{total_conversations}")

            try:
                transcript = runner.run_conversation(input_data, args.num_turns, target_prompt)
                runner.save_transcript(transcript)
            except Exception as e:
                logger.error(f"Failed to run conversation {conversation_count}: {e}")
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