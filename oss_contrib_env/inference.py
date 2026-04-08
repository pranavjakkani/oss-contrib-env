import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from baseline_agent import build_candidate_preview, choose_route_action


API_BASE_URL = os.environ.get("API_BASE_URL", "https://huggingface.co/api/inference-proxy/together")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASKS = ["triage", "duplicate", "patch_loc"]
MAX_STEPS = 5
HTTP_TIMEOUT_SECONDS = 30
MODEL_TIMEOUT_SECONDS = 90
MAX_COMPLETION_TOKENS = 700
TEMPERATURE = 0.2

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving an offline open-source benchmark.

    Follow the task instruction exactly.
    Return only the action payload, with no markdown fences.
    Prefer concise outputs:
    - triage: one issue id
    - duplicate: JSON list of duplicate ids
    - patch_loc: JSON list of ranked file paths
    """
).strip()


def sanitize_log_value(value: Optional[Any]) -> str:
    if value is None:
        return "null"
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def get_openai_base_url() -> str:
    base_url = API_BASE_URL.rstrip("/")
    if base_url == "https://huggingface.co/api/inference-proxy/together":
        return "https://router.huggingface.co/v1"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def log_start(task: str, env_url: str, model: str) -> None:
    print(
        f"[START] task={sanitize_log_value(task)} env={sanitize_log_value(env_url)} model={sanitize_log_value(model)}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error_message: Optional[str],
    progress: float,
) -> None:
    print(
        f"[STEP] step={step} action={sanitize_log_value(action)} reward={reward:.4f} "
        f"progress={progress:.4f} done={str(done).lower()} error={sanitize_log_value(error_message)}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float) -> None:
    print(
        f"[END] task={sanitize_log_value(task)} success={str(success).lower()} steps={steps} score={score:.4f}",
        flush=True,
    )


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
        data = response.read().decode("utf-8")
    return json.loads(data) if data else {}


def split_result(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
    observation = payload.get("observation")
    if isinstance(observation, dict):
        reward = float(payload.get("reward", observation.get("reward", 0.0)) or 0.0)
        done = bool(payload.get("done", observation.get("done", False)))
        return observation, reward, done

    reward = float(payload.get("reward", 0.0) or 0.0)
    done = bool(payload.get("done", payload.get("observation", {}).get("done", False)))
    return payload, reward, done


def build_user_prompt(observation: Dict[str, Any], step: int, history: List[str], heuristic_action: str) -> str:
    history_block = "\n".join(history) if history else "None"
    info = observation.get("info", {})
    return textwrap.dedent(
        f"""
        Task type: {observation.get("task_type", "triage")}
        Difficulty: {observation.get("difficulty", "unknown")}
        Step: {step}
        Task ID: {observation.get("task_id", "unknown")}
        Attempts remaining: {observation.get("attempts_remaining", "unknown")}
        Current progress: {info.get("progress", 0.0)}
        Available actions: {info.get("available_actions", [])}

        Task:
        {observation.get("issue", "")}

        Candidates:
        {build_candidate_preview(observation, limit=8)}

        Latest feedback:
        {observation.get("test_output") or "None"}

        Previous submissions:
        {history_block}

        Heuristic baseline suggestion:
        {heuristic_action or "None"}

        Choose the best next action for an interactive trajectory.
        You may inspect first and submit later.
        Improve on the heuristic if you can, otherwise return it exactly.
        """
    ).strip()


def create_openai_client():
    if not HF_TOKEN:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(base_url=get_openai_base_url(), api_key=HF_TOKEN)


def get_model_response(client, observation: Dict[str, Any], step: int, history: List[str], heuristic_action: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(observation, step, history, heuristic_action),
            },
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    message = completion.choices[0].message.content or ""
    return message.strip()


def normalize_action(observation: Dict[str, Any], action: str, heuristic_action: str) -> str:
    cleaned = (action or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned or heuristic_action


def choose_hybrid_action(client, observation: Dict[str, Any], step: int, history: List[str]) -> str:
    heuristic_action = choose_route_action(observation)
    if client is None:
        return heuristic_action
    model_action = get_model_response(client, observation, step, history, heuristic_action)
    return normalize_action(observation, model_action, heuristic_action)


def run_task(client, task_name: str) -> None:
    history: List[str] = []
    score = 0.0
    steps_taken = 0
    success = False
    progress = 0.0
    current_observation: Dict[str, Any] = {}

    log_start(task_name, ENV_URL, MODEL_NAME if client is not None else "heuristic-baseline")

    try:
        reset_payload = post_json(f"{ENV_URL.rstrip('/')}/reset", {"task_id": task_name})
        current_observation, score, done = split_result(reset_payload)
        progress = float(current_observation.get("info", {}).get("progress", 0.0) or 0.0)
        success = done and progress >= 1.0

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = ""
            error_message: Optional[str] = None

            try:
                action = choose_hybrid_action(client, current_observation, step, history)
                if not action:
                    raise ValueError("agent returned empty content")

                step_payload = post_json(f"{ENV_URL.rstrip('/')}/step", {"action": {"response": action}})
                current_observation, score, done = split_result(step_payload)
                progress = float(current_observation.get("info", {}).get("progress", 0.0) or 0.0)
                success = done and progress >= 1.0
            except error.HTTPError as exc:
                error_message = f"HTTP {exc.code}"
                done = True
                score = 0.0
                progress = 0.0
            except error.URLError as exc:
                error_message = str(exc.reason)
                done = True
                score = 0.0
                progress = 0.0
            except Exception as exc:
                error_message = str(exc)
                done = True
                score = 0.0
                progress = 0.0

            steps_taken = step
            if error_message is None:
                history.append(
                    f"Step {step}: submission={sanitize_log_value(action)} | "
                    f"reward={score:.4f} | progress={progress:.4f} | "
                    f"feedback={sanitize_log_value(current_observation.get('test_output'))}"
                )
            log_step(step, action, score, done, error_message, progress)

            if done:
                break
    except error.HTTPError as exc:
        log_step(0, "", 0.0, True, f"HTTP {exc.code} during reset", 0.0)
    except error.URLError as exc:
        log_step(0, "", 0.0, True, f"{exc.reason} during reset", 0.0)
    except Exception as exc:
        log_step(0, "", 0.0, True, f"{exc} during reset", 0.0)
    finally:
        log_end(task_name, success, steps_taken, score)


def main() -> None:
    client = create_openai_client()
    for task_name in TASKS:
        run_task(client, task_name)


if __name__ == "__main__":
    main()
