from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


ShortText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=800,
    ),
]

SourceNumber = Annotated[
    int,
    Field(
        ge=1,
        le=20,
    ),
]


class StrictOutputModel(BaseModel):
    """
    禁止 LLM 輸出 Schema 沒有定義的欄位。
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class CitedDefinition(StrictOutputModel):
    risk_label: Literal[
        "harassment",
        "cyberbullying",
    ] = Field(
        description=(
            "此說明對應的 BERT 風險類別"
        )
    )

    explanation: ShortText = Field(
        description=(
            "保守說明可能相關的定義或風險，"
            "不得作出法律定罪或事實認定"
        )
    )

    source_numbers: list[SourceNumber] = Field(
        min_length=1,
        max_length=3,
        description=(
            "支持此說明的知識庫來源編號"
        ),
    )


class CitedAction(StrictOutputModel):
    action: ShortText = Field(
        description=(
            "由知識庫支持的具體處置建議"
        )
    )

    source_numbers: list[SourceNumber] = Field(
        min_length=1,
        max_length=3,
        description=(
            "支持此建議的知識庫來源編號"
        ),
    )


class RagGenerationOutput(StrictOutputModel):
    definitions: list[CitedDefinition] = Field(
        min_length=1,
        max_length=4,
        description=(
            "只說明本次 BERT 實際偵測到的"
            "風險類別"
        ),
    )

    actions: list[CitedAction] = Field(
        min_length=2,
        max_length=4,
        description=(
            "由知識庫支持的處置建議"
        ),
    )

"""
Schema 的功能
這會要求 TAIDE 只能輸出：
{
  "definitions": [
    {
      "risk_label": "harassment",
      "explanation": "可能涉及言語攻擊……",
      "source_numbers": [2]
    }
  ],
  "actions": [
    {
      "action": "保存完整留言與上下文。",
      "source_numbers": [1]
    }
  ]
}
"""