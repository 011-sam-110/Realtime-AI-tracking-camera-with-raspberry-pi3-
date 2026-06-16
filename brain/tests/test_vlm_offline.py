"""Offline unit tests for the VLM backends (INTERFACES.md §9).

Covers `kitchenvision.vlm.cloud_vlm.CloudVlm` and `kitchenvision.vlm.local_vlm.LocalVlm`.

NO network, NO model download, NO torch/CUDA required:
  * CloudVlm is tested with a fully MOCKED openai SDK injected into `sys.modules`, asserting that
    generate() returns a VlmResult (text + X-Routed-Via provider) and that 422 -> NoVisionError,
    429 -> RateLimitError, other -> VlmError.
  * LocalVlm: __init__ + `available` must not raise whether or not deps are present; generate() is
    exercised by monkeypatching `_load()` to install a STUB model + processor (no real download).

Plain-python runnable (pytest may be absent):  python tests/test_vlm_offline.py
Run from the brain root (C:/Users/sampo/pi/brain) so `import kitchenvision...` resolves.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

# Make `import kitchenvision...` work when run as a bare script from the brain root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from kitchenvision.vlm.base import (  # noqa: E402
    NoVisionError,
    RateLimitError,
    VisionModel,
    VlmError,
    VlmResult,
)
from kitchenvision.vlm.cloud_vlm import CloudVlm  # noqa: E402
from kitchenvision.vlm.local_vlm import LocalVlm  # noqa: E402


# --------------------------------------------------------------------------- a mock openai SDK
class _MockHeaders:
    def __init__(self, mapping):
        self._m = {k.lower(): v for k, v in mapping.items()}

    def get(self, key, default=None):
        return self._m.get(key.lower(), default)


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockChoice:
    def __init__(self, content):
        self.message = _MockMessage(content)


class _MockCompletion:
    def __init__(self, content):
        self.choices = [_MockChoice(content)]


class _MockRawResponse:
    def __init__(self, content, headers):
        self._content = content
        self.headers = _MockHeaders(headers)

    def parse(self):
        return _MockCompletion(self._content)


def _make_mock_openai(*, behaviour="ok", text="hello", headers=None):
    """Return a stand-in `openai` module object configured for a given behaviour.

    behaviour: "ok" | "422" | "429" | "boom" — controls what `.create(...)` does.
    """
    mod = types.ModuleType("openai")

    # Error classes the SDK exposes (CloudVlm catches these by attribute).
    class APIStatusError(Exception):
        def __init__(self, message="", body=None):
            super().__init__(message)
            self.message = message
            self.body = body

    class UnprocessableEntityError(APIStatusError):
        pass

    class RateLimitError(APIStatusError):
        pass

    mod.APIStatusError = APIStatusError
    mod.UnprocessableEntityError = UnprocessableEntityError
    mod.RateLimitError = RateLimitError

    captured = {}

    class _Create:
        def create(self, *, model, max_tokens, messages):
            captured["model"] = model
            captured["max_tokens"] = max_tokens
            captured["messages"] = messages
            if behaviour == "422":
                raise UnprocessableEntityError("no vision model", body={"message": "no vision"})
            if behaviour == "429":
                raise RateLimitError("rate limited", body={"message": "slow down"})
            if behaviour == "boom":
                raise RuntimeError("network exploded")
            hdrs = {"X-Routed-Via": "groq/llava"} if headers is None else headers
            return _MockRawResponse(text, hdrs)

    class _WithRaw:
        with_raw_response = _Create()

    class _Chat:
        completions = _WithRaw()

    class _Client:
        def __init__(self, base_url, api_key):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = _Chat()

    mod.OpenAI = _Client
    mod._captured = captured  # for assertions
    return mod


def _install_mock_openai(monkeypatch_dict, **kw):
    """Put a freshly-built mock `openai` into sys.modules; return it. Caller restores."""
    mod = _make_mock_openai(**kw)
    sys.modules["openai"] = mod
    return mod


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- CloudVlm tests
def _cloud_config():
    return {"vlm": {"backend": "cloud", "base_url": "https://x/v1", "api_key": "k", "model": "auto"}}


def test_cloud_available() -> None:
    """available is True only with both creds; construction never imports openai / never raises."""
    assert CloudVlm(_cloud_config()).available is True
    assert CloudVlm({"vlm": {"base_url": "", "api_key": "k"}}).available is False
    assert CloudVlm({"vlm": {"base_url": "u", "api_key": ""}}).available is False
    assert CloudVlm({}).available is False  # no vlm block at all
    print("  cloud available: needs both base_url + api_key")


def test_cloud_is_visionmodel() -> None:
    """CloudVlm satisfies the runtime-checkable VisionModel protocol."""
    assert isinstance(CloudVlm(_cloud_config()), VisionModel)
    print("  cloud isinstance VisionModel: ok")


def test_cloud_generate_ok() -> None:
    """generate() returns VlmResult(text, provider=X-Routed-Via) and ships a text+image block."""
    saved = sys.modules.get("openai")
    try:
        mod = _install_mock_openai(None, behaviour="ok", text="making tea")
        c = CloudVlm(_cloud_config())
        res = c.generate(_frame(), "what is happening?", max_tokens=123)
        assert isinstance(res, VlmResult)
        assert res.text == "making tea"
        assert res.provider == "groq/llava"  # from X-Routed-Via header
        cap = mod._captured
        assert cap["model"] == "auto"
        assert cap["max_tokens"] == 123
        assert cap["base_url"] == "https://x/v1" and cap["api_key"] == "k"
        # The message carries a text block and an image_url data: URL.
        content = cap["messages"][0]["content"]
        kinds = {b["type"] for b in content}
        assert kinds == {"text", "image_url"}
        img = next(b for b in content if b["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
        print("  cloud generate ok: VlmResult + data-URL image block + header provider")
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)


def test_cloud_provider_falls_back_to_model() -> None:
    """With no X-Routed-Via header, provider falls back to the model name."""
    saved = sys.modules.get("openai")
    try:
        _install_mock_openai(None, behaviour="ok", text="x", headers={})
        c = CloudVlm({"vlm": {"base_url": "u", "api_key": "k", "model": "gpt-vision"}})
        res = c.generate(_frame(), "p")
        assert res.provider == "gpt-vision"
        print("  cloud provider fallback: -> model name when header absent")
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)


def _assert_cloud_raises(behaviour, exc_type):
    saved = sys.modules.get("openai")
    try:
        _install_mock_openai(None, behaviour=behaviour)
        c = CloudVlm(_cloud_config())
        raised = None
        try:
            c.generate(_frame(), "p")
        except Exception as e:  # noqa: BLE001
            raised = e
        assert isinstance(raised, exc_type), (
            f"behaviour={behaviour} expected {exc_type.__name__}, got {type(raised).__name__}"
        )
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)


def test_cloud_error_mapping() -> None:
    """422 -> NoVisionError, 429 -> RateLimitError, other -> VlmError (all subclasses of VlmError)."""
    _assert_cloud_raises("422", NoVisionError)
    _assert_cloud_raises("429", RateLimitError)
    _assert_cloud_raises("boom", VlmError)
    # And the typed errors are all VlmError subclasses (worker catches the base).
    assert issubclass(NoVisionError, VlmError) and issubclass(RateLimitError, VlmError)
    print("  cloud error mapping: 422->NoVision, 429->RateLimit, other->VlmError")


# --------------------------------------------------------------------------- LocalVlm tests
def test_local_construct_and_available() -> None:
    """__init__ + available never raise (regardless of torch/transformers presence); id mapping ok."""
    lv = LocalVlm({"vlm": {"local_model": "moondream2", "device": "cuda"}})
    assert lv.model_id == "vikhyatk/moondream2"
    assert lv.device == "cuda"
    # available returns a plain bool without importing torch.
    assert isinstance(lv.available, bool)

    # Default / unknown alias falls back to qwen2-vl-2b.
    assert LocalVlm({}).model_id == "Qwen/Qwen2-VL-2B-Instruct"
    assert LocalVlm({"vlm": {"local_model": "florence2"}}).model_id == "microsoft/Florence-2-base"
    assert LocalVlm({"vlm": {"local_model": "bogus"}}).model_id == "Qwen/Qwen2-VL-2B-Instruct"
    print(f"  local construct: id-map ok, available={lv.available} (no raise)")


def test_local_is_visionmodel() -> None:
    """LocalVlm satisfies the VisionModel protocol without loading anything."""
    assert isinstance(LocalVlm({}), VisionModel)
    print("  local isinstance VisionModel: ok")


def test_local_generate_via_stub_load() -> None:
    """generate() works when _load() is monkeypatched to install stub model + processor.

    No real torch / transformers / download — we replace _load and the inference helper so the
    code path returns a VlmResult tagged with the model id.
    """
    lv = LocalVlm({"vlm": {"local_model": "qwen2-vl-2b"}})

    loaded_flag = {"called": False}

    def fake_load():
        loaded_flag["called"] = True
        lv._model = object()
        lv._processor = object()
        lv._loaded = True

    # Replace the lazy loader and the torch-dependent inference with offline stubs.
    lv._load = fake_load  # type: ignore[assignment]
    lv._run_inference = lambda image, prompt, max_tokens: f"caption:{prompt}"  # type: ignore[assignment]

    res = lv.generate(_frame(), "doing dishes", max_tokens=50)
    assert loaded_flag["called"] is True
    assert isinstance(res, VlmResult)
    assert res.text == "caption:doing dishes"
    assert res.provider == "Qwen/Qwen2-VL-2B-Instruct"
    print("  local generate (stubbed _load): VlmResult + provider = model id")


def test_local_generate_load_failure_raises_vlmerror() -> None:
    """If _load() raises VlmError (deps/model missing), generate() propagates VlmError."""
    lv = LocalVlm({})

    def boom_load():
        raise VlmError("no torch on this box")

    lv._load = boom_load  # type: ignore[assignment]
    raised = None
    try:
        lv.generate(_frame(), "p")
    except Exception as e:  # noqa: BLE001
        raised = e
    assert isinstance(raised, VlmError), f"expected VlmError, got {type(raised).__name__}"
    print("  local load failure: -> VlmError")


# --------------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [
        test_cloud_available,
        test_cloud_is_visionmodel,
        test_cloud_generate_ok,
        test_cloud_provider_falls_back_to_model,
        test_cloud_error_mapping,
        test_local_construct_and_available,
        test_local_is_visionmodel,
        test_local_generate_via_stub_load,
        test_local_generate_load_failure_raises_vlmerror,
    ]
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__} ...")
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} tests FAILED")
    else:
        print(f"RESULT: all {len(tests)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
