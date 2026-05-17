from pathlib import Path
from typing import Any, Literal, TypeAlias

from PIL import Image

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
ConfigDict: TypeAlias = dict[str, Any]

PathLike: TypeAlias = str | Path

ModelFamily: TypeAlias = Literal["medgemma", "gemma4", "chexone"]
BackendName: TypeAlias = Literal["vllm", "transformers"]

CaseData: TypeAlias = dict[str, Any]
CaseBatch: TypeAlias = list[tuple[str, CaseData]]
ResultPayload: TypeAlias = dict[str, Any]
ResultsById: TypeAlias = dict[str, ResultPayload]
EvaluationStats: TypeAlias = dict[str, Any]

ImageInput: TypeAlias = Image.Image
MessageContent: TypeAlias = str | list[dict[str, Any]]
ChatMessage: TypeAlias = dict[str, Any]
Conversation: TypeAlias = list[ChatMessage]
PreparedCase: TypeAlias = dict[str, Any]
ModelResponse: TypeAlias = str | dict[str, Any]
