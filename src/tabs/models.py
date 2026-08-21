from dataclasses import dataclass


@dataclass
class Source:
    name: str
    feed_url: str
    category: str
    institutional_tier: int
