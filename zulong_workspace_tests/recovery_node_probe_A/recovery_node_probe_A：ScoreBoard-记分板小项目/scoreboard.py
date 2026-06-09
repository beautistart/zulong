"""
ScoreBoard - core module
Manages players and scores.
"""

from collections import defaultdict


class ScoreBoard:
    """A simple scoreboard that manages players and their scores."""

    def __init__(self):
        """Initialize an empty scoreboard."""
        self._scores = defaultdict(list)

    def add_player(self, name: str) -> bool:
        """Add a new player.

        Args:
            name: Player name (non-empty string).

        Returns:
            True if the player is newly added, False if already exists.

        Raises:
            ValueError: If name is empty.
        """
        if not name or not name.strip():
            raise ValueError("Player name cannot be empty")
        name = name.strip()
        if name in self._scores:
            return False
        self._scores[name] = []
        return True

    def record(self, name: str, score: int) -> bool:
        """Record a score for a player.

        Args:
            name: Player name.
            score: Score (integer).

        Returns:
            True if recorded successfully.

        Raises:
            ValueError: If player does not exist.
        """
        if name not in self._scores:
            raise ValueError(f"Player '{name}' does not exist. Call add_player first.")
        self._scores[name].append(score)
        return True

    def leader(self, n: int = 3) -> list:
        """Return top n players by average score descending.

        Args:
            n: Number of players to return (default 3).

        Returns:
            List of (player_name, average_score) tuples.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        averages = []
        for name, scores in self._scores.items():
            if scores:
                avg = sum(scores) / len(scores)
            else:
                avg = 0.0
            averages.append((name, avg))

        averages.sort(key=lambda x: (-x[1], x[0]))
        return averages[:n]

    def player_count(self) -> int:
        """Return the current number of players.
        
        Returns:
            Integer count of registered players.
        """
        return len(self._scores)

    def summary(self) -> dict:
        """Return aggregate statistics for the scoreboard.

        Returns:
            Dict with keys:
            - total_players: Number of players
            - total_scores: Number of score records
            - top_player: Name of player with highest avg (or None)
            - top_avg: Highest average score (or None)
        """
        total_players = len(self._scores)
        total_scores = sum(len(scores) for scores in self._scores.values())

        top_player = None
        top_avg = None
        if total_players > 0:
            leaders = self.leader(1)
            if leaders:
                top_player, top_avg = leaders[0]

        return {
            "total_players": total_players,
            "total_scores": total_scores,
            "top_player": top_player,
            "top_avg": top_avg,
        }