# Core models module for ultimate_mcp_server
# This module contains all Pydantic models used for tournaments, requests, and synthesis

from ultimate_mcp_server.core.models.requests import (
    CompletionRequest,
)
from ultimate_mcp_server.core.models.tournament import (
    CancelTournamentInput,
    CancelTournamentOutput,
    CreateTournamentInput,
    CreateTournamentOutput,
    EvaluatorConfig,
    GetTournamentResultsInput,
    GetTournamentStatusInput,
    GetTournamentStatusOutput,
    ModelConfig,
    ModelResponseData,
    SingleShotGeneratorModelConfig,
    SingleShotIndividualResponse,
    SingleShotSynthesisInput,
    SingleShotSynthesisOutput,
    TournamentBasicInfo,
    TournamentConfig,
    TournamentData,
    TournamentRoundResult,
    TournamentStatus,
)

__all__ = [
    # Tournament models
    "TournamentStatus",
    "ModelConfig",
    "EvaluatorConfig",
    "TournamentConfig",
    "TournamentRoundResult",
    "ModelResponseData",
    "TournamentData",
    "CreateTournamentInput",
    "CreateTournamentOutput",
    "GetTournamentStatusInput",
    "GetTournamentStatusOutput",
    "GetTournamentResultsInput",
    "CancelTournamentInput",
    "CancelTournamentOutput",
    "TournamentBasicInfo",
    # Single-shot synthesis models
    "SingleShotGeneratorModelConfig",
    "SingleShotIndividualResponse",
    "SingleShotSynthesisInput",
    "SingleShotSynthesisOutput",
    # Request models
    "CompletionRequest",
]
