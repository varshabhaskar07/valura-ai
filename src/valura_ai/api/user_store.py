"""UserStore — looks up a User by user_id.

Loads from ``fixtures/users/*.json`` at startup. Production would replace
this with a database adapter; the interface is just ``get(user_id) -> User | None``.
"""

from __future__ import annotations

from pathlib import Path

from ..portfolio.models import User


class UserStore:
    def __init__(self, users_dir: Path) -> None:
        self._users: dict[str, User] = {}
        if users_dir.exists():
            for path in sorted(users_dir.glob("*.json")):
                user = User.from_fixture(path)
                self._users[user.user_id] = user

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._users

    @property
    def known_ids(self) -> tuple[str, ...]:
        return tuple(self._users)
