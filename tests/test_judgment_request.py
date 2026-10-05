import pytest
from pydantic import ValidationError

from backend.models.schemas import ProcessThreadRequest


BASE_REQUEST = {
    "url": "https://www.threads.com/@example/post/1234567890",
}


def test_judgment_limit_defaults_to_five() -> None:
    request = ProcessThreadRequest(**BASE_REQUEST)
    assert request.judgment_limit == 5


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_judgment_limit_accepts_supported_range(limit: int) -> None:
    request = ProcessThreadRequest(**BASE_REQUEST, judgment_limit=limit)
    assert request.judgment_limit == limit


@pytest.mark.parametrize("limit", [0, 11])
def test_judgment_limit_rejects_values_outside_supported_range(limit: int) -> None:
    with pytest.raises(ValidationError):
        ProcessThreadRequest(**BASE_REQUEST, judgment_limit=limit)
