"""
File - packet.py
Data packet generation, store-and-forward routing, and delivery simulation.

This module models packets originating at satellites and being routed hop-by-hop
toward Earth ground receivers across the active inter-satellite link graph.

At each simulation frame:
  - Each satellite independently spawns a packet with a fixed arrival probability.
  - Every in-flight packet attempts one hop toward the nearest reachable receiver
    using a breadth-first search over the current frame's link topology.
  - Packets that cannot find a route are held on their current satellite and
    retried on the next frame (store-and-forward).

Key components:
  Packet                    — immutable-style dataclass tracking a packet's
                              origin, route, hop count, and delivery state.
  generate_packets()        — Poisson packet spawning per frame.
  build_adjacency_graph()   — converts a NetworkSnapshot into a neighbour dict.
  can_see_any_ground_point()— checks whether a satellite has line-of-sight to
                              at least one Earth receiver.
  bfs_to_closest_receiver() — BFS over the link graph to find the shortest-hop
                              path to any receiver-visible satellite.
  simulate_packet_routing() — runs the full store-and-forward loop over all
                              simulation frames and returns delivery statistics.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict, deque
import ground_receivers
import numpy as np

# Packet Object
@dataclass
class Packet:

    # Each packet is born at a specific satellite index at a specific frame.
    # That's all it knows about its origin.
    # _________________________________________________________________________
    # Packet ID
    packet_id: int

    # Packet Creation
    original_satellite: int
    created_at_frame: int

    # Packet Delivery
    delivered: bool = False
    delivered_at_frame: Optional[int] = None

    # Packet Traversal Route
    hops: int = 0
    path: list[int] = field(default_factory=list)

    # Bandwidth
    bytes_size: int = 1024
    # _________________________________________________________________________


# Packet Generation
# ________________________________________________________________
# At each frame, satellites independently generates a packet with
# probability p per frame
# ________________________________________________________________
def generate_packets(random_number, frame, satellite_count):
    
    """ This function will return a list of new Packet Object born at the passed in frame """

    # Satellite has a 10% chance of spawning a packet per frame.
    arrival_rate = 0.1

    # Generating packets per frame with a 10% probability
    packets_generated = []
    for satellite_idx in range(satellite_count):
        if random_number.random() < arrival_rate:
            packets_generated.append(Packet(
                packet_id=frame * satellite_count + satellite_idx,
                original_satellite=satellite_idx,
                created_at_frame=frame)
            )

    return packets_generated


# NetworkSnapshot Graph Construction
# _______________________________________________________
# For every connection_indices for each NetworkSnapshot, 
#   connection_indices: tuple[tuple[int, int], ...]
# turn the active links at those instant into a adjacency
# dictionary
# _______________________________________________________
def build_adjacency_graph(networkSnapshot):
    adj_graph = defaultdict(set)
    for i, j in networkSnapshot.connection_indices:
        adj_graph[i].add(j)
        adj_graph[j].add(i)
    return adj_graph


# Ground Visibility Check
# _______________________________________________________
# Check if a single satellite has line of sight to at
# least one ground receiving point
# _______________________________________________________
def can_see_any_ground_point(satellite, ground_points, earth_radius):

    """ Return True if this satellite has line-of-sight to at least one ground point. """

    # Retreiving any uncovered ground receivers based off satellite position
    position = np.atleast_2d(satellite.pos())  # Shape (1, 3)
    _, covered = ground_receivers.nearest_visible_satellite_distances(
        position, ground_points, earth_radius=earth_radius
    )

    # Return if there is any receivers uncovered
    return bool(np.any(covered))


# Simulating Routing Loop
# _______________________________________________________
# Simulate the packet routing in a loop
# _______________________________________________________
def simulate_packet_routing(networkSnapshots, satellites, ground_points, earth_radius, route_strategy="bfs", receiver_capacity=10, seed=0):

    # Initializing starting variables
    random_number = np.random.default_rng(seed)
    satellite_queues: dict[int, deque] = defaultdict(deque)
    satellites_dropped: list[Packet] = []
    satellites_delivered: list[Packet] = []

    # Satellite has a 10% chance of spawning a packet per frame.
    arrival_rate = 0.1
    for networkSnapshot in networkSnapshots: 

        # Retrieving the current frame of the network snapshot
        # and building its adjacency graph
        frame = networkSnapshot.frame
        adj_graph = build_adjacency_graph(networkSnapshot)

        # Resets the amount of receivers delivered per frame
        receivers_delivered = 0

        # Spawning new packets
        for sat_idx in range(len(satellites)):

            if random_number.random() < arrival_rate:

                # Drop the new packet if this satellite's queue is already full else queue it
                if len(satellite_queues[sat_idx]) >= satellites[sat_idx].storage_capacity():
                    satellites_dropped.append(Packet(
                        packet_id=frame * len(satellites) + sat_idx,
                        original_satellite=sat_idx,
                        created_at_frame=frame))
                else:
                    satellite_queues[sat_idx].append(Packet(
                        packet_id=frame * len(satellites) + sat_idx,
                        original_satellite=sat_idx,
                        created_at_frame=frame))

        # Attempting one hop per packet across all satellite queues
        satellites_next_queues: dict[int, deque] = defaultdict(deque)
        for sat_idx in range(len(satellites)):
            
            # Initialize the amount of bytes sent to 0
            bytes_sent = 0
            
            while (satellite_queues[sat_idx]):

                # Check for bandwidth limits
                packet = satellite_queues[sat_idx][0]
                if bytes_sent + packet.bytes_size > satellites[sat_idx].link_bandwidth():
                    break

                # Retireve the current packet and carrier satellite
                packet = satellite_queues[sat_idx].popleft()
                carrierSatellite = packet.path[-1] if packet.path else packet.original_satellite
                bytes_sent += packet.bytes_size


                # Retrieving the next hop satellite depending on the routing strategy
                next_hop_satellite = None
                if route_strategy == "greedy":
                    next_hop_satellite = greedy_distance_to_receiver(
                        carrierSatellite, adj_graph, satellites, ground_points, earth_radius)
                elif route_strategy == "bfs":
                    next_hop_satellite = bfs_to_closest_receiver(
                        carrierSatellite, adj_graph, satellites, ground_points, earth_radius)
                elif route_strategy == "least_congested":
                    next_hop_satellite = least_congested_to_receiver(
                        carrierSatellite, adj_graph, satellites, ground_points, earth_radius, satellite_queues)
                    

                # No route this frame, store packet and retry next frame
                if next_hop_satellite is None:
                    satellites_next_queues[sat_idx].append(packet)

                # Already at a receiver-visible satellite, deliver packet
                elif next_hop_satellite == carrierSatellite:

                    # Deliver only if the receivers has capacity in this current frame else
                    # if it is full, store it and retry the next frame
                    if receivers_delivered < len(ground_points) * receiver_capacity:
                        packet.delivered = True
                        packet.delivered_at_frame = frame
                        satellites_delivered.append(packet)
                        receivers_delivered += 1
                    else:
                        satellites_next_queues[sat_idx].append(packet)
                    

                # Move one hop closer to a receiver
                else:
                    packet.path.append(next_hop_satellite)
                    packet.hops += 1
                    satellites_next_queues[next_hop_satellite].append(packet)

        satellite_queues = satellites_next_queues

    return satellites_delivered, satellites_dropped, satellite_queues


# Packet Routing (Three Different Search Approach
# _________________________________________________________________________


# Find the shortest path to a neigboring satellite that is close
# to a receiver
# _______________________________________________________
def bfs_to_closest_receiver(carrierSatellite, adj_graph, satellites, ground_points, earth_radius):

    """ Return the next hop satellite index, or None if no packet path exist in this frame """

    # Retrieving the destinations - Satellites that directly covers a receiver
    destinations = {
        idx for idx, satellite in enumerate(satellites)
        if can_see_any_ground_point(satellite, ground_points, earth_radius)
    }

    # Return if destination is already in the delivery node
    if carrierSatellite in destinations:
        return carrierSatellite

    # Queue and set to store satellites either in queue or has been visited
    visited_satellites = {carrierSatellite}
    queue = deque([(carrierSatellite, carrierSatellite)])
    
    # Breath First Search to find the shortest path to any satellite
    # that has a receiver
    while (queue):
        (node, first_hop) = queue.popleft()
        for neighborSatellite in adj_graph[node]:
            if neighborSatellite not in visited_satellites:
                visited_satellites.add(neighborSatellite)
                queue.append((neighborSatellite, first_hop))
                if neighborSatellite in destinations:
                    return first_hop
    
    # Return if no route to a receiver was found
    # Store this and wait
    return None


# Using greedy algorithmn, per each hop, retrieve the
# neighboring satellite that is the closest to a receiver
# _______________________________________________________
def greedy_distance_to_receiver(carrierSatellite, adj_graph, satellites, ground_points, earth_radius):
    """ Return the next hop satellite index, or if no route exists in this frame, return None """

    # If the carrierSatellite is already visible to the receiever, deliver immediately
    if can_see_any_ground_point(satellites[carrierSatellite], ground_points, earth_radius):
        return carrierSatellite

    # Accessing its neighboring satellites
    neighboring_satellites = adj_graph[carrierSatellite]
    if not neighboring_satellites:
        return None

    # Pick the neighbour whose position is closest to any ground point
    def min_distance_to_ground(sat_idx):
        pos = satellites[sat_idx].pos()
        return min(
            float(np.linalg.norm(pos - gp))
            for gp in ground_points
        )

    return min(neighboring_satellites, key=min_distance_to_ground)


# Extended version of BFS search meaning finding the shortest path
# to a satellite connected to a ground receiver, but also accounting for which
# satellite has the least amount of packets congested
# _______________________________________________________
def least_congested_to_receiver(carrierSatellite, adj_graph, satellites, ground_points, earth_radius, satellite_queues):
    """ Return the next hop satellite index, or None if no route exists this frame """

    # If the carrierSatellite is already visible to the receiever, deliver immediately
    if can_see_any_ground_point(satellites[carrierSatellite], ground_points, earth_radius):
        return carrierSatellite

    # Accessing its neighboring satellites
    neighbors = set()
    for index, satellite in enumerate(satellites):
        if can_see_any_ground_point(satellite, ground_points, earth_radius):
            neighbors.add(index)

    visited = {carrierSatellite}
    queue: deque[tuple[Any, Any]] = deque([(carrierSatellite, carrierSatellite)])

    while queue:
        node, first_hop = queue.popleft()

        # Sort neighbours by how many packets they currently hold
        neighbours_by_load = sorted(
            adj_graph[node],
            key=lambda idx: len(satellite_queues[idx])
        )

        for neighbour in neighbours_by_load:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, first_hop))
                if neighbour in neighbors:
                    return first_hop

    return None
# _________________________________________________________________________