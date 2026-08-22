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
from typing import Optional
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


# Packet Routing
# _______________________________________________________
# For every packet in each frame, route one hop
# toward one ground receiver and find the closest
# receiver to it
# (Using Greedy and BFS Algorithm)
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


# Simulating Routing Loop
# _______________________________________________________
# Simulate the packer routing in a loop
# _______________________________________________________
def simulate_packet_routing(networkSnapshots, satellites, ground_points, earth_radius, seed=0):

    # Initializing starting variables
    random_number = np.random.default_rng(seed)
    satellites_in_flight: list[Packet] = []
    satellites_delivered: list[Packet] = []

    # Satellite has a 10% chance of spawning a packet per frame.
    arrival_rate = 0.1

    for networkSnapshot in networkSnapshots:

        # Retrieving the current frame of the network snapshot
        # and building its adjacency graph
        frame = networkSnapshot.frame
        adj_graph = build_adjacency_graph(networkSnapshot)

        # Count how many satellites_in_flight packets are currently sitting on each satellite
        packets_queued_per_satellite = defaultdict(int)
        for packet in satellites_in_flight:
            current_carrier_satellite = packet.path[-1] if packet.path else packet.original_satellite
            packets_queued_per_satellite[current_carrier_satellite] += 1

        # Spawning new packets
        for sat_idx in range(len(satellites)):

            # Drop the new packet if this satellite's queue is already full
            if packets_queued_per_satellite[sat_idx] >= satellites[sat_idx].storage_capacity():
                continue

            if random_number.random() < arrival_rate:
                satellites_in_flight.append(Packet(
                    packet_id=frame * len(satellites) + sat_idx,
                    original_satellite=sat_idx,
                    created_at_frame=frame))

        # Attempting one hop per packet
        satellites_flying = []
        for packet in satellites_in_flight:

            # Retrieve the current carrier satellite index
            carrierSatellite = packet.path[-1] if packet.path else packet.original_satellite
            next_hop_satellite = bfs_to_closest_receiver(carrierSatellite, adj_graph, satellites, ground_points, earth_radius)

            if next_hop_satellite is None:
                # No route this frame, store packet and wait
                satellites_flying.append(packet)
            elif next_hop_satellite == carrierSatellite:
                # Already at a receiver-visible satellite, deliver packet
                packet.delivered = True
                packet.delivered_at_frame = frame
                satellites_delivered.append(packet)
            else:
                # Move one packet hop closer to a receiver
                packet.path.append(next_hop_satellite)
                packet.hops += 1
                satellites_flying.append(packet)

        satellites_in_flight = satellites_flying

    return satellites_delivered, satellites_in_flight

            

