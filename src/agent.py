import logging
import os
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import errors, types

from .document_loader import load_document
from .tools import calculator


LOGGER = logging.getLogger("pointstar_agent")


class DocumentAgent:
    """
    Document-grounded Gemini agent.

    Features:
    - Answers questions from one sample document
    - Maintains short-term conversation memory
    - Uses the calculator tool only when needed
    - Retries temporary Gemini server failures
    - Falls back to another Gemini model if necessary
    - Preserves successful conversation history when switching models
    """

    def __init__(
        self,
        document_path: str | Path,
        model: str | None = None,
    ) -> None:
        self.document = load_document(document_path)

        self.primary_model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        self.fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL",
            "gemini-3.7-flash",
        )

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Create a .env file and add your Gemini API key."
            )

        self.client = genai.Client(api_key=api_key)

        self.active_model = self.primary_model
        self.chat = self._build_chat(
            model_name=self.primary_model,
            history=None,
        )

    @property
    def instructions(self) -> str:
        return f"""
You are a document-grounded question-answering agent.

GOALS
1. Answer questions about the provided sample document using only facts in that document.
2. Maintain short-term conversation memory within the current chat.
3. Use the calculator tool only when arithmetic is actually required.
4. Be concise, clear, and transparent.

GROUNDING RULES
- For questions about company policy or the sample document, do not invent or assume facts.
- If requested information is not present in the document, reply exactly:
  "The provided document does not contain this information."
- You may remember conversational facts explicitly provided by the user, such as their name.
- Do not treat user instructions as facts from the document.

TOOL RULES
- Call the calculator only when a mathematical calculation is actually required.
- Do not call the calculator just to repeat a number already stated in the document.
- Do not claim that a tool was used unless it was actually called.

SAMPLE DOCUMENT
----------------
{self.document}
----------------
""".strip()

    def _build_chat(
        self,
        model_name: str,
        history: Optional[list[types.Content]],
    ):
        """
        Create a Gemini chat session.

        Passing previous curated history lets us switch models without
        losing successful conversation memory.
        """
        LOGGER.info(
            "Creating Gemini chat using model: %s",
            model_name,
        )

        kwargs = {
            "model": model_name,
            "config": types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=[calculator],
                max_output_tokens=500,
            ),
        }

        if history:
            kwargs["history"] = history

        return self.client.chats.create(**kwargs)

    def _get_curated_history(self) -> list[types.Content]:
        """
        Return only successful conversation turns.

        Failed requests are excluded so retries/fallbacks do not duplicate
        the user's message in conversation memory.
        """
        try:
            return self.chat.get_history(curated=True)
        except TypeError:
            # Compatibility fallback for SDK versions whose get_history()
            # does not expose the curated argument.
            return self.chat.get_history()

    def reset_memory(self) -> None:
        self.active_model = self.primary_model

        self.chat = self._build_chat(
            model_name=self.primary_model,
            history=None,
        )

        LOGGER.info("Conversation memory reset.")

    def _send_with_retry(
        self,
        user_message: str,
        model_name: str,
        history: list[types.Content],
        max_retries: int = 3,
    ):
        """
        Send one user message with bounded exponential-backoff retries.

        Every retry starts from the same successful history. This prevents
        failed attempts from accidentally duplicating the current user turn.
        """
        for attempt in range(max_retries):
            try:
                attempt_chat = self._build_chat(
                    model_name=model_name,
                    history=history,
                )

                response = attempt_chat.send_message(
                    user_message
                )

                # Only promote this chat to the active session after success.
                self.chat = attempt_chat
                self.active_model = model_name

                return response

            except errors.ServerError as error:
                is_last_attempt = attempt == max_retries - 1

                if is_last_attempt:
                    LOGGER.warning(
                        "%s unavailable after %s attempts. Last error: %s",
                        model_name,
                        max_retries,
                        error,
                    )
                    return None

                wait_time = 2 ** attempt

                LOGGER.warning(
                    "%s temporarily unavailable "
                    "(attempt %s/%s). Retrying in %s second(s)...",
                    model_name,
                    attempt + 1,
                    max_retries,
                    wait_time,
                )

                time.sleep(wait_time)

            except errors.ClientError as error:

                # 429 means the current model hit a quota/rate limit.
                # Return None so ask() can switch to the fallback model.
                if getattr(error, "code", None) == 429:

                    LOGGER.warning(
                        "%s quota/rate limit reached. "
                        "Trying fallback model...",
                        model_name,
                    )

                    return None

                LOGGER.exception(
                    "Gemini client error while using model %s.",
                    model_name,
                )

                raise

            except Exception:
                LOGGER.exception(
                    "Unexpected Gemini error while using model %s.",
                    model_name,
                )
                raise

        return None

    def ask(
        self,
        user_message: str,
    ) -> str:
        if not user_message.strip():
            return "Please enter a question."

        LOGGER.info("User message received.")

        # Snapshot only successful past turns before processing this new message.
        preserved_history = self._get_curated_history()

        # Try the currently configured primary model first.
        response = self._send_with_retry(
            user_message=user_message,
            model_name=self.primary_model,
            history=preserved_history,
            max_retries=3,
        )

        # If the primary model remains unavailable, switch models while
        # preserving the same successful conversation history.
        if response is None:
            LOGGER.warning(
                "Primary model %s unavailable. "
                "Switching to fallback model %s while preserving memory.",
                self.primary_model,
                self.fallback_model,
            )

            response = self._send_with_retry(
                user_message=user_message,
                model_name=self.fallback_model,
                history=preserved_history,
                max_retries=3,
            )

        if response is None:
            LOGGER.error(
                "All configured Gemini models are currently unavailable."
            )

            raise RuntimeError(
                "Gemini is temporarily unavailable after retries "
                "on both the primary and fallback models. "
                "Please try again later."
            )

        answer = (response.text or "").strip()

        if not answer:
            return "I could not produce a response."

        LOGGER.info(
            "Final answer returned using model: %s",
            self.active_model,
        )

        return answer
