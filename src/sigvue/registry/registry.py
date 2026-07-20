from __future__ import annotations

from dataclasses import dataclass, field

from sigvue.core.contracts import WorkspaceProtocol
from sigvue.core.errors import DuplicateWorkspaceError


@dataclass
class WorkspaceRegistry:
    _workspaces: dict[str, WorkspaceProtocol] = field(default_factory=dict)

    def register(self, workspace: WorkspaceProtocol) -> None:
        identifier = workspace.metadata.identifier
        if identifier in self._workspaces:
            raise DuplicateWorkspaceError(f"Workspace '{identifier}' is already registered")
        self._workspaces[identifier] = workspace

    def get(self, workspace_id: str) -> WorkspaceProtocol:
        return self._workspaces[workspace_id]

    def list(self) -> list[WorkspaceProtocol]:
        return list(self._workspaces.values())
