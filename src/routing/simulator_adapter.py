"""
Bridge between the continuous orbital Network simulator (src/model)
and the Temporal / RL Routing ContactPlan (src/models/contact.py).
"""
import numpy as np
from src.model.satellite_generator import create_random_satellites, load_constellation
from src.model.network import Network
from src.models.contact import Contact, ContactPlan

def extract_contact_plan_from_simulation(
    constellation_file: str = "coverage_optimized_constellation.json",
    duration_s: float = 1800.0,
    dt_s: float = 10.0,
    data_rate_bps: float = 100_000_000.0,  # 100 Mbps
) -> ContactPlan:
    """
    Simulates the orbital constellation and extracts dynamic contact windows.
    """
    try:
        satellites = load_constellation(constellation_file)
    except Exception:
        satellites = create_random_satellites(num_satellites=10, seed=42)

    net = Network(satellites)
    n_steps = int(duration_s / dt_s)
    
    active_contacts = {}  # (src, dst) -> start_time
    completed_contacts = []
    
    current_time = 0.0
    for step in range(n_steps):
        net.update_network()
        current_links = set(net.connection_indices())
        
        # Check ended links
        ended = set(active_contacts.keys()) - current_links
        for pair in ended:
            start_t = active_contacts.pop(pair)
            if current_time - start_t >= dt_s:  # Valid window
                completed_contacts.append(
                    Contact(
                        source_id=pair[0],
                        destination_id=pair[1],
                        start_s=start_t,
                        end_s=current_time,
                        data_rate_bps=data_rate_bps
                    )
                )
                # Bi-directional link
                completed_contacts.append(
                    Contact(
                        source_id=pair[1],
                        destination_id=pair[0],
                        start_s=start_t,
                        end_s=current_time,
                        data_rate_bps=data_rate_bps
                    )
                )
        
        # Check newly started links
        for pair in current_links:
            if pair not in active_contacts:
                active_contacts[pair] = current_time
        
        # Advance simulation
        current_time += dt_s
        for sat in net.satellites():
            sat.step(dt_s)
            
    # Flush remaining active contacts at horizon
    for pair, start_t in active_contacts.items():
        if current_time - start_t >= dt_s:
            completed_contacts.append(
                Contact(
                    source_id=pair[0],
                    destination_id=pair[1],
                    start_s=start_t,
                    end_s=current_time,
                    data_rate_bps=data_rate_bps
                )
            )
            completed_contacts.append(
                Contact(
                    source_id=pair[1],
                    destination_id=pair[0],
                    start_s=start_t,
                    end_s=current_time,
                    data_rate_bps=data_rate_bps
                )
            )
            
    return ContactPlan(completed_contacts)