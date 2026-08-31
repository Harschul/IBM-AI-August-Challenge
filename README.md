

#AI Routing & Validation for Space Research Networks

**IBM AI Builders Challenge — August: Advance Space Exploration with AI**

> **AI is already learning which observations from space matter most. AstraRoute explores the next step: intelligently getting that science back to Earth.**

---

## Problem Statement

Space research missions are generating more data than ever.

NASA's Nancy Grace Roman Space Telescope is designed to return around **1.4 TB of science data every day**, more than any previous NASA astrophysics mission. ESA's Euclid must return up to **100 GB per day** through only a few hours of ground-station contact.

At the same time, satellites cannot simply transmit whenever they want.

Ground stations move in and out of view. Relay opportunities appear and disappear. Bandwidth is limited. Weather can degrade ground links. Transfers can fail. And scientific data is not equally valuable: an observation of a transient astronomical event may be far more urgent than routine calibration or housekeeping data.

This creates a growing problem for astronomy and space research:

> **When communication opportunities are limited, how should scientific data move through a satellite network so that the most valuable information reaches researchers quickly and reliably?**

---

## Solution Description

**AstraRoute is an AI routing development and validation platform for scientific satellite networks.**

The prototype combines five components in one software package:

| Component                 | Purpose                                                                          |
| ------------------------- | -------------------------------------------------------------------------------- |
| **Orbital Digital Twin**  | Simulates spacecraft, relays, ground stations and changing communication windows |
| **AI Router**             | Uses reinforcement learning to decide where scientific data should move next     |
| **Deterministic Router**  | Provides a trusted earliest-arrival baseline and safety fallback                 |
| **Benchmark Environment** | Tests routing policies under identical traffic, failures and network conditions  |
| **Mission Dashboard**     | Visualises packets, contacts, routes, failures and performance                   |

The simulator is not just developer scaffolding.

It is the environment in which a mission team can **train routing policies, stress-test them, compare them against deterministic routing and understand when AI should or should not be trusted**.

The submitted prototype uses a reproducible 14-node constellation consisting of research spacecraft, LEO relays, GEO relays and ground receivers.

A future version could accept mission-specific constellations and operational data.

---

## AI Approach and Architecture

AstraRoute uses **MaskablePPO reinforcement learning** to make next-hop routing decisions.

Rather than calculating one fixed route from spacecraft to Earth, the AI repeatedly observes the current network and decides:

> **Where should this scientific data go next?**

Its inputs include:

* scientific priority and deadline;
* packet size;
* available communication windows;
* link capacity and congestion;
* spacecraft health and battery state;
* weather risk;
* link reliability.

Physically impossible actions are masked before inference.

Alongside the AI, AstraRoute includes a deterministic **Temporal earliest-arrival router**.

This gives the platform a hybrid architecture:

```text
Scientific observation
        |
        v
Priority · Size · Deadline · Type
        |
        v
+--------------------------------+
|      ROUTING INTELLIGENCE      |
|                                |
|       MaskablePPO AI           |
|              +                 |
|    Temporal Safety Router      |
+---------------+----------------+
                |
                v
      Physical Contact Network
  geometry · capacity · weather
      health · reliability
                |
                v
       LEO / GEO Relay Network
                |
                v
             Earth
```

The final AI policy was trained with:

* **14 routing actions**
* **158 observation features**
* **2,000,000 training timesteps**
* **4 parallel environments**
* **32 science bundles per training episode**

The product philosophy is simple:

> **Use AI where it creates measurable value. Keep a deterministic route home when it does not.**

---

## Selected Challenge Theme

### Advance Space Exploration with AI

Our focus is **AI assistance for astronomy and scientific research**.

Future research missions will increasingly combine:

* larger scientific instruments;
* more onboard computing;
* autonomous science analysis;
* multiple spacecraft;
* relay infrastructure;
* intermittent communication with Earth.

AstraRoute explores how AI can assist one of the next decisions in that chain: **how important scientific information should travel through the network after it has been collected.**

This turns AI from something that only analyses science data into part of the infrastructure that helps deliver that science.

---

## How IBM Bob Was Used

IBM Bob was used as an AI-assisted engineering partner throughout development.

The project began as several independent components: orbital simulation, reinforcement learning, Temporal routing, stochastic transmission modelling, benchmarking and frontend visualisation.

Bob helped us integrate these systems into a single reproducible application by assisting with:

* implementation and debugging;
* interface integration;
* regression testing;
* benchmark tooling;
* frontend development;
* release verification.

Its most valuable role was helping the different components agree on the same spacecraft state, packet state, routing actions, communication windows and transfer behaviour.

IBM Bob therefore helped turn several research components into a **working end-to-end AI prototype**.

---

# Why Now?

The communications problem AstraRoute addresses is becoming increasingly relevant from both directions:

### Scientific missions are producing dramatically more data.

NASA's Roman Space Telescope is expected to send approximately **1.4 TB of science data to Earth every day** — around 500 times Hubble's historical daily rate.

ESA's Euclid generates up to **100 GB per day** while communicating with Earth through a limited daily window.

More capable detectors create more science, but also put greater pressure on the communications system carrying that science.

At the same time, the network itself is changing.

---

# Space Relay Networks Are Already Emerging

The relay network represented in AstraRoute is not science fiction.

## ESA's SpaceDataHighway

ESA's **European Data Relay System (EDRS)** already uses geostationary satellites to relay data from lower-orbit spacecraft.

Instead of forcing an Earth-observation satellite to wait until it passes directly over its own ground station, EDRS can receive its data through an optical inter-satellite link and relay it toward Earth.

ESA designed the system specifically to reduce delays for time-sensitive applications and lists a transmission capacity of at least **50 TB per day**.

```text
Traditional downlink

Satellite
   |
   | wait for ground pass
   |
   v
Ground Station


Relay network

Satellite
   |
   v
GEO Relay
   |
   v
Ground
```

That changes routing from:

> "When can I see Earth?"

to:

> **"Which available path through the network should my data take?"**

---

## NASA's Delay-Tolerant Networking

NASA is moving in the same direction.

In 2026, **Delay/Disruption Tolerant Networking (DTN)** became an operational service across NASA's Near Space Network and Deep Space Network.

DTN is based on store-and-forward communication: data can remain at one node until another communication opportunity becomes available.

NASA is also expanding networking toward the Moon through LunaNet and commercial lunar relay services.

The network therefore increasingly looks like this:

```text
Research Spacecraft
        |
        v
      Relay
        |
      store
        |
        v
   another relay
        |
        v
   Ground Network
```

As those networks become larger, the number of possible routing decisions increases.

That is the environment in which intelligent routing becomes interesting.

---

# AI Is Already Deciding Which Science Is Worth Sending

The other half of this idea is already happening onboard spacecraft.

NASA JPL's **OASIS — Onboard Autonomous Science Investigation System** analyses spacecraft science data and assigns priority so that the most valuable information can be transmitted first.

ESA has taken this idea directly into orbit with **Φsat-2**.

Its onboard AI can analyse Earth-observation imagery before downlink.

One application assigns anomaly scores to maritime images. Images containing potentially important anomalies can be prioritised for transmission before routine imagery.

Other Φsat-2 AI applications can detect cloud-covered images and avoid wasting bandwidth sending imagery that is not useful.

The progression is natural:

```text
PAST

Collect data
     |
     v
Send everything
     |
     v
Analyse on Earth


TODAY

Collect data
     |
     v
AI analyses onboard
     |
     v
Which data matters?
     |
     v
Prioritise downlink


ASTRAROUTE

Collect data
     |
     v
AI analyses / prioritises
     |
     v
Which data matters?
     |
     v
AI-assisted routing
     |
     v
How should it reach Earth?
```

If onboard AI can already determine **what is worth transmitting**, the next natural question is:

> **Can AI also help determine the best way to transmit it?**

That is the idea AstraRoute explores.

---

# What Does the User Get?

AstraRoute is both the **routing intelligence** and the **environment used to develop and validate that intelligence**.

The two would serve different roles in a future operational system.

## Ground Side — AstraRoute Platform

Mission and AI teams use the full software environment to:

```text
Define mission
     |
     v
Simulate network
     |
     v
Generate scientific traffic
     |
     v
Train AI router
     |
     v
Stress-test AI
     |
     v
Compare against Temporal
     |
     v
Inspect behaviour
     |
     v
Approve / improve model
```

This is the **digital twin, training, validation and monitoring environment**.

## Spacecraft Side — Routing Intelligence

A future spacecraft would not need to run the entire simulator.

It would carry the lightweight part required to make decisions:

```text
Science Data Queue
       |
       v
Current Network State
       |
       v
AI Routing Policy
       |
       +------ safe decision ------> Route packet
       |
       +------ unsafe / failure
                         |
                         v
                  Temporal fallback
```

The submitted repository contains both sides in one reproducible prototype.

---

# Simulation Environment

The environment is designed to make routing realistically difficult.

It models:

### Changing orbital contacts

Communication links appear and disappear as spacecraft move.

A transfer must completely fit inside its communication window:

```text
departure_time + transmission_time <= contact_end_time
```

### Finite bandwidth

Contact capacity is consumed by previous transmissions.

Routing one packet can therefore affect later packets.

### Scientific priorities

Generated bundles contain:

* source spacecraft;
* creation time;
* size;
* science priority;
* deadline;
* data type.

Traffic types include:

* `TRANSIENT`
* `STAR_FIELD`
* `CALIBRATION`
* `HOUSEKEEPING`

Urgent transient observations receive higher priorities and shorter deadlines.

### Probabilistic failures

Transfers can fail.

Failure probability combines:

* baseline risk;
* weather;
* spacecraft health;
* link reliability.

If transmission fails, time and capacity are still consumed and the complete packet remains at the sender.

The router must then retry or find another route.

---

# Current Prototype Network

| Node Type           |  Count |
| ------------------- | -----: |
| Research satellites |      3 |
| LEO relays          |      6 |
| GEO relays          |      2 |
| Ground stations     |      3 |
| **Total**           | **14** |

The current AI has a fixed 14-action interface.

This gives the competition release a completely reproducible training and benchmark environment.

Support for arbitrary constellation sizes is part of the product roadmap rather than a claim of the current checkpoint.

---

# What Has the AI Learned?

The final benchmark evaluates:

* **20 held-out scenarios**
* **500 science bundles per scenario**
* **10,000 bundles per routing policy**
* identical generated traffic;
* identical stochastic conditions;
* identical network physics;
* pure PPO without hidden Temporal fallback.

| Metric                            |   Temporal |         PPO |
| --------------------------------- | ---------: | ----------: |
| Delivery ratio                    | **75.28%** |      62.16% |
| On-time delivery                  | **71.08%** |      59.47% |
| Priority-weighted timely delivery | **67.21%** |      57.08% |
| Mean successful-delivery latency  |    216.3 s | **181.3 s** |
| Mean hops                         |   **2.23** |        3.39 |
| Transfer failure rate per attempt |      4.39% |   **3.40%** |

The AI does not win every metric.

That is precisely why the validation environment matters.

---

# A 16% Latency Advantage

Among successfully delivered packets, PPO reaches Earth in:

## **181.3 seconds**

versus:

## **216.3 seconds with Temporal routing**

That is approximately:

# **16% lower mean latency**

For astronomy and scientific research, latency can matter.

Consider a transient astronomical event.

An observatory identifies something unusual and produces a high-priority data product.

Getting that information to researchers sooner can allow:

* other telescopes to perform follow-up observations;
* researchers to react before the event evolves;
* additional instruments to change observing strategy.

AstraRoute therefore explores an objective beyond raw network throughput:

> **How quickly can the network return the science that matters?**

---

# Why Does PPO Currently Deliver Fewer Packets?

The simulation also reveals an important weakness in the current learned policy.

PPO behaves more **burstily** than Temporal earliest-arrival routing.

Temporal tends to commit packets toward Earth as soon as it identifies the earliest feasible path.

The learned policy is more willing to distribute packets through relay nodes and wait for favourable downstream opportunities.

```text
Temporal

Packet ---> shortest temporal opportunity ---> Ground


PPO

Packet ---> Relay ---> Relay
                     |
                     | wait for favourable contact
                     v
                   Ground
```

When this succeeds, it can result in excellent latency.

But the benchmark has a hard **1,800-second simulation horizon**.

Packets still staged inside the relay network when the simulation ends cannot complete their route and count against the delivery metrics.

We do not assume that every remaining packet would eventually arrive.

Instead, the result tells us something actionable:

> **The AI needs better awareness of terminal backlog and collapsing deadline slack.**

This is exactly the kind of behaviour the AstraRoute testing platform is intended to expose.

---

# AI + Deterministic Safety

The software already supports:

1. **Temporal routing**
2. **Pure PPO**
3. **PPO with Temporal fallback**
4. **Scheduled switching between routing policies**

The benchmark intentionally uses pure PPO so its weaknesses remain visible.

A deployed version would not need to.

For example:

| Situation                                    | Routing Mode        |
| -------------------------------------------- | ------------------- |
| Healthy network with several routing choices | **AI**              |
| Network congestion or complex relay choices  | **AI**              |
| Comfortable packet deadline                  | **AI**              |
| Deadline becomes critical                    | **Temporal rescue** |
| AI model unavailable                         | **Temporal**        |
| Invalid AI decision                          | **Temporal**        |
| Network state outside validated conditions   | **Temporal**        |
| Protected / mission-critical data            | **Deterministic**   |

This leads to a more useful question than simply asking whether AI is better:

> **When should AI be trusted to make the decision?**

---

# Benchmarking Is Part of the Product

AstraRoute is not designed around one PPO checkpoint.

The included model is the **first example routing policy**.

The larger product idea is the environment around it.

A mission team could train a new routing model and then ask:

```text
NEW AI MODEL
     |
     v
AstraRoute Digital Twin
     |
     +---- normal traffic
     +---- astronomy transient burst
     +---- heavy congestion
     +---- poor weather
     +---- link failures
     +---- degraded spacecraft
     +---- short deadlines
     |
     v
Compare against deterministic router
     |
     v
Performance + failure analysis
     |
     v
Deploy?
 YES / NO / RESTRICTED
```

That gives AI development a measurable safety process.

---

# Interactive Dashboard

The included dashboard allows users to inspect:

* spacecraft and relay positions;
* active communication links;
* scientific packets;
* packet routes;
* transfer failures;
* scientific priorities;
* deadlines;
* ground stations;
* Temporal routing;
* PPO routing;
* final performance.

The frontend and benchmark use the same experiment runner:

```text
src/experiment/runner.py
```

The visual demonstration therefore uses the same contact, capacity and failure semantics as the reported experiment.

---

# Product Roadmap

### Hybrid Mode

Expose AI + Temporal safety routing directly in the dashboard.

### Deadline Rescue

Automatically switch a packet to earliest-arrival routing when deadline slack becomes unsafe.

### Confidence-Based Routing

Only grant the AI authority for network states similar to conditions on which it has been validated.

### Arbitrary Constellations

Move from the fixed 14-node policy to graph-based or variable-size routing models.

### Operational Inputs

Replace simulated values with real:

* orbital ephemerides;
* contact plans;
* ground-station weather;
* spacecraft health;
* storage;
* battery;
* network congestion.

### Flight Hardware Testing

Benchmark routing inference and fallback logic on flight-representative onboard computers.

---

# Reproducibility

The final experiment is locked in:

```text
config/final_experiment.json
```

Committed benchmark evidence is available in:

```text
artifacts/final_experiment/benchmark/
├── summary.json
├── seed_metrics.csv
└── bundle_results.csv
```

The trained model is included at:

```text
RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

No retraining is required to run the submitted prototype.

---

# Technology

* Python
* Stable-Baselines3
* sb3-contrib MaskablePPO
* Gymnasium
* NumPy
* Pandas
* Streamlit
* Temporal routing
* custom orbital/contact simulation
* stochastic communication modelling
* GitHub Actions
* IBM Bob

---

# Tutorial

## 1. Clone the repository

```bash
git clone https://github.com/Harschul/IBM-AI-August-Challenge.git
cd IBM-AI-August-Challenge
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. Verify the release

```bash
python scripts/final/verify_release.py
```

This confirms that the expected configuration, benchmark and trained AI checkpoint belong to the same final experiment.

---

## 5. Launch AstraRoute

```bash
python run_frontend.py
```

Open the local Streamlit address shown in the terminal.

---

## 6. Choose a Routing Policy

Select:

### **Temporal**

Runs deterministic earliest-arrival routing.

### **PPO**

Runs the final trained AI policy.

Select the same held-out scenario under both modes to compare their behaviour under equivalent traffic and stochastic conditions.

---

## 7. Explore the Simulation

Use the timeline to watch:

* physical links appear and disappear;
* science bundles enter the network;
* packets travel between relays;
* failed transfers;
* successful ground delivery.

Select individual packets to inspect:

* scientific priority;
* data type;
* size;
* deadline;
* route;
* attempts;
* failures;
* final delivery state.

---

## 8. Reproduce the Benchmark

```bash
python run_final_benchmark.py
```

Reference results are stored in:

```text
artifacts/final_experiment/benchmark/
```

---

## 9. Run the Tests

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
python scripts/final/verify_release.py
```

---

# Scientific Scope

AstraRoute is a **simulation and research prototype**, not flight software.

The current policy is trained for the fixed 14-node experiment.

Traffic, weather, spacecraft health, link properties and stochastic failures are simulated values rather than measurements from an operational mission.

The benchmark therefore demonstrates behaviour in this controlled environment rather than proving universal superiority of AI routing.

That is intentional.

The purpose of AstraRoute is not to demonstrate that:

> **AI always wins.**

It is to create the environment required to determine:

> **Where does AI create mission value, how much value does it create, and when should a trusted deterministic system take over?**

---

# Submission

**IBM AI Builders Challenge with IBM Bob**
**August Theme — Advance Space Exploration with AI**

AstraRoute combines:

* a scientific satellite-network digital twin;
* a trainable reinforcement-learning environment;
* a trained AI routing policy;
* deterministic safety routing;
* realistic communication uncertainty;
* benchmarking;
* an operator dashboard.

Scientific spacecraft are producing more information than ever.

AI is already beginning to decide **which observations deserve attention**.

Relay networks are beginning to change **how those observations reach Earth**.

### **AstraRoute connects those two trends.**

> **If AI can decide which science matters, the next natural step is helping decide how that science gets home.**
