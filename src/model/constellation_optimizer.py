import numpy as np


def score_network(snapshots, satellites):
    """
    Returns a normalized constellation score between 0 and 1.

    Each active link contributes:

        actual_distance / maximum_connection_distance

    Disconnected satellite pairs contribute 0.

    The score is averaged over every possible pair and every frame.
    """

    n = len(satellites)

    # Number of possible links in a fully connected network
    max_possible_links = n * (n - 1) // 2

    frame_scores = []

    for snapshot in snapshots:

        frame_score = 0.0

        for i, j in snapshot.connection_indices:

            # Current physical distance between connected satellites
            distance = np.linalg.norm(
                snapshot.positions[i] - snapshot.positions[j]
            )

            # Your link rule uses the smaller range of the two satellites
            max_distance = min(
                satellites[i].connection_range(),
                satellites[j].connection_range(),
            )

            # 0 -> 1 contribution from this link
            normalized_distance = distance / max_distance

            frame_score += normalized_distance

        # Convert total for this frame into 0 -> 1
        frame_score /= max_possible_links

        frame_scores.append(frame_score)

    # Average quality of the network across the whole simulation
    return float(np.mean(frame_scores))