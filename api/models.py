from pydantic import BaseModel


class SummaryDict(BaseModel):
    title: str
    bullets: list[dict]
