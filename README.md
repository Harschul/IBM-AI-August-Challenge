# Constellation Routing Simulator

### AI-assisted scientific data delivery for future space relay networks

**IBM AI Builders Challenge — August: Advance Space Exploration with AI**

> **What if scientific data could decide how to get home?**

Spacecraft are becoming better at generating data faster than we are becoming better at returning it.

A research satellite may capture a high-value observation while no ground station is visible. A relay opportunity might appear minutes later. One route may be faster but riskier. Weather may degrade a ground link. Several missions may compete for the same contact window. Some data may be scientifically urgent; other data can wait.

Our prototype turns that communications problem into an **AI decision problem**.

We built a physical multi-orbit network simulator, a reinforcement-learning routing agent, a deterministic temporal router, a stochastic transfer model, a reproducible benchmarking system, and an interactive mission dashboard.

Together, they form a prototype for a future system with two complementary components:

1. **an onboard routing intelligence layer** that decides where scientific data should move next; and
2. **a mission-control digital twin** that allows operators to inspect, benchmark, compare, and safely override that behaviour.

The architecture is deliberately hybrid:

> **AI is allowed to be useful without being allowed to become a single point of failure.**

A deterministic earliest-arrival router remains available as a known safety path.

---

# The problem: space data does not travel through a normal network

Terrestrial networks usually assume that an end-to-end path exists.

Space networks cannot.

A spacecraft can transmit only when geometry, range, bandwidth, capacity and receiver availability align. A connection may exist for only a short window. End-to-end connectivity may never exist at one instant.

Data therefore has to be **stored, carried, forwarded, stored again, and transmitted later** when the next opportunity appears.

Scientific missions make the routing problem even harder:

* a transient observation can be much more valuable than routine telemetry;
* two packets can have completely different deadlines;
* a contact can disappear before a large packet finishes transmitting;
* weather can increase risk at the ground segment;
* a failed transfer consumes time and capacity even if the packet remains onboard;
* choosing one contact now changes the capacity available to later traffic;
* several ground stations may be valid destinations;
* a relay that looks unattractive now may unlock a much better future route.

This is not simply shortest-path routing.

It is sequential decision-making under changing connectivity, constrained resources, uncertainty, deadlines and scientific value.

That is exactly where AI becomes interesting.

---

# Our product vision

We envision the system as an **AI-assisted space data router**.

A mission supplies a constellation scenario and communication assumptions. The simulator propagates the spacecraft, builds a time-dependent physical contact plan, creates scientific data bundles, and exposes the state of the network to the routing layer.

The routing intelligence then decides — hop by hop — where each data bundle should move.

```text
Scientific instrument / onboard storage
                    |
                    v
        Mission-aware data bundle
       priority · size · deadline
                    |
                    v
      +-----------------------------+
      |     ROUTING INTELLIGENCE    |
      |                             |
      |      MaskablePPO AI         |
      |             +               |
      |   deterministic Temporal    |
      |        safety path          |
      +--------------+--------------+
                     |
                     v
            Physical ContactPlan
       geometry · bandwidth · capacity
       weather · health · reliability
                     |
                     v
            Stochastic transfer
          success / failure / retry
                     |
                     v
        LEO / GEO relay constellation
                     |
                     v
           valid ground receiver
                     |
                     v
          Mission-control dashboard
```

The competition release deliberately locks the learned policy to a reproducible **14-node network**.

Orbital geometry, traffic, link properties, risk assumptions and ground stations are configuration-driven. Generalising the learned interface to arbitrary constellation sizes is a natural next step toward a deployable product.

---

# Prototype at a glance

| Capability                     | Competition Prototype                                    |
| ------------------------------ | -------------------------------------------------------- |
| Research spacecraft            | 3                                                        |
| LEO relay satellites           | 6                                                        |
| GEO relay satellites           | 2                                                        |
| Operational ground receivers   | 3                                                        |
| Total routing nodes            | 14                                                       |
| AI policy                      | MaskablePPO reinforcement learning                       |
| AI actions                     | 14 masked next-hop actions                               |
| AI observation                 | 158 features                                             |
| Training                       | 2,000,000 timesteps                                      |
| Parallel training environments | 4                                                        |
| Scientific traffic             | generated bundles with size, type, priority and deadline |
| Transfer model                 | capacity-aware + stochastic failures                     |
| Risk inputs                    | weather, health and link reliability                     |
| Safety path                    | deterministic Temporal earliest-arrival routing          |
| Benchmark                      | 20 held-out scenarios × 500 bundles × 2 algorithms       |
| Dashboard                      | orbit, contacts, packets, failures, routes and metrics   |

The AI does not see only node positions.

Its observation contains information about the current scientific bundle and candidate next hops, including:

* science priority;
* bundle size;
* remaining deadline slack;
* current mission time;
* data rate;
* remaining contact duration;
* range;
* queue / utilisation state;
* storage availability;
* spacecraft health;
* battery state;
* weather risk;
* link reliability.

Invalid actions are masked.

The learned policy can therefore select only a next hop for which a physically feasible transmission currently exists.

---

# Why AI belongs in this system

A deterministic Temporal router is excellent at answering a well-defined question:

> **Given the current contact plan, what feasible route gets this packet to a ground receiver earliest?**

That makes it a strong baseline.

It also makes it an excellent safety fallback.

But a future relay network creates a broader problem.

A routing policy may need to balance:

**latency + scientific importance + deadline slack + congestion + storage + energy + risk + future connectivity + many competing packets.**

As networks become larger, encoding every desired behaviour as hand-written routing rules becomes increasingly difficult.

Our AI agent takes a different approach.

It does not pre-program an entire end-to-end path.

It makes a **next-hop decision**, observes the resulting network state, and decides again.

That is important for onboard autonomy: inference can happen locally, even when ground control is delayed, intermittent, or unavailable.

The goal is therefore not:

> “Replace deterministic networking because AI is fashionable.”

The product hypothesis is:

> **Use learning where the network presents meaningful choices. Use deterministic routing where mission assurance matters more than adaptation.**

And we deliberately benchmark those ideas honestly.

Our headline PPO benchmark uses **pure PPO with no hidden deterministic fallback**.

If the AI performs badly, the benchmark is allowed to show it.

Only then can we make an informed decision about when it deserves operational authority.

---

# Benchmark: what has the AI learned?

The locked final benchmark uses:

* **20 held-out paired scenarios**;
* **500 generated science bundles per scenario**;
* **10,000 evaluated bundles per algorithm**;
* identical traffic-seed families;
* identical stochastic-seed families;
* the same physical contact plan;
* the same capacity semantics;
* the same stochastic transfer model;
* **zero Temporal fallback in the reported PPO result**.

| Metric                            |   Temporal |         PPO | Interpretation                                    |
| --------------------------------- | ---------: | ----------: | ------------------------------------------------- |
| Delivery ratio                    | **75.28%** |      62.16% | Temporal currently delivers more completely       |
| On-time delivery                  | **71.08%** |      59.47% | Temporal currently protects deadlines better      |
| Priority-weighted timely delivery | **67.21%** |      57.08% | PPO still needs stronger science-value scheduling |
| Mean successful-delivery latency  |    216.3 s | **181.3 s** | **PPO is ~35 s / ~16% faster**                    |
| Mean hops                         |   **2.23** |        3.39 | PPO uses a more distributed relay strategy        |
| Transfer failure rate per attempt |      4.39% |   **3.40%** | PPO attempts lower-risk contacts on this metric   |

## The result we care most about: latency

Among the packets successfully delivered by each policy, PPO reaches Earth in:

### **181.3 seconds**

compared with:

### **216.3 seconds for Temporal routing**

That is approximately:

# **35 seconds faster**

### **≈ 16% lower successful-delivery latency**

For an ordinary file transfer, 35 seconds may sound modest.

For time-sensitive scientific observations, it can matter enormously.

Examples include:

* transient astronomy;
* wildfire or disaster imaging;
* space-weather observations;
* anomaly telemetry;
* event-driven follow-up observations;
* time-critical Earth observation.

The usefulness of some data decays every minute it remains trapped onboard.

A future intelligent relay network is therefore not only about delivering more bytes.

It is about delivering the **right bytes sooner**.

---

# The second interesting signal: risk

PPO also records a lower **per-attempt transfer failure rate**:

**3.40% vs 4.39%.**

That is not the same as saying PPO is lower-risk in every possible sense.

The learned policy currently uses more hops and more attempts, which means cumulative exposure and wasted capacity still need optimisation.

But the per-attempt result suggests something valuable:

> **The agent has learned to prefer individual contacts with more favourable transfer characteristics.**

That is exactly the sort of behaviour an intelligent routing layer should learn from a state containing reliability, weather and network conditions.

---

# So why does PPO deliver fewer packets before their deadlines?

This is one of the most interesting results of the prototype.

And we do not want to hide it.

Inspection of the simulated behaviour suggests that the learned policy behaves more **burstily** than Temporal earliest-arrival routing.

Temporal routing tends to commit packets toward Earth as soon as it finds the earliest feasible route.

The PPO agent is more willing to move data around the relay network and exploit later downstream contact opportunities.

Conceptually:

```text
Temporal
--------
packet
  |
  +--> earliest feasible path
             |
             +--> ground


PPO
---
packet
  |
  +--> relay
         |
         +--> relay / wait
                 |
                 +--> favourable downstream opportunity
                               |
                               +--> ground
```

This staging behaviour can produce excellent latency when the final relay opportunity appears.

That helps explain why successfully delivered PPO traffic can reach Earth significantly faster.

But the competition simulation has a hard **1,800-second horizon**.

Traffic that remains staged somewhere in the relay network when the experiment reaches its cutoff cannot complete inside the benchmark.

This hurts:

* delivery ratio;
* deadline success;
* priority-weighted timely delivery.

We should not claim that every packet remaining at the end would eventually have been delivered — the experiment does not simulate beyond the cutoff.

The engineering conclusion is more useful:

> **The AI has discovered a potentially valuable fast-delivery strategy, but it needs stronger deadline/terminal-state training or a deterministic rescue mechanism when remaining slack becomes small.**

And that leads directly to our hybrid architecture.

---

# AI when useful. Deterministic when necessary.

The execution engine already supports multiple operating modes.

### 1. Temporal

Pure deterministic earliest-arrival routing.

### 2. PPO Pure

The learned policy runs without safety substitution.

This is the mode used for the headline benchmark.

### 3. AI + Temporal fallback

The AI receives the first opportunity to make the decision.

If it cannot safely provide one, the deterministic router takes over.

### 4. Scheduled switching

The requested routing policy can change during the mission.

This allows a system to run AI during one phase and deterministic routing during another.

---

# The deterministic safety path

When fallback mode is enabled, failure of the learned model is **not allowed to become a data-loss mode**.

The existing execution engine can fall back when:

* the trained model is unavailable;
* model inference fails;
* the model produces an invalid action;
* no feasible learned action exists;
* the contact available at decision time is no longer valid when the transfer is committed.

This is important.

A real mission should not have to choose between:

**“use AI everywhere”**

and

**“never use AI.”**

There is a much more practical option:

> **Allow AI to earn authority inside a bounded operational envelope and maintain a deterministic route home outside that envelope.**

---

# When would we switch?

A flight-oriented version could implement explicit mission guardrails.

| Situation                                                             | Preferred routing mode                     |
| --------------------------------------------------------------------- | ------------------------------------------ |
| Healthy network, several feasible choices, comfortable deadline slack | **AI**                                     |
| Highly dynamic or congested relay environment                         | **AI**                                     |
| Very high-value / protected telemetry                                 | **Deterministic / certified policy**       |
| Remaining deadline slack falls below threshold                        | **Temporal rescue**                        |
| Model unavailable or produces invalid output                          | **Automatic deterministic fallback**       |
| Current state is outside AI training distribution                     | **Deterministic fallback**                 |
| Contact plan changes unexpectedly                                     | **Fallback / replanning**                  |
| Operator enters protected mission phase                               | **Manual or scheduled deterministic mode** |

This is our product philosophy:

# **AI should earn authority through measurable performance — and it should be easy to take that authority away.**

---

# A simulation designed to make routing difficult

An intelligent router is not interesting if the network around it cannot go wrong.

The prototype therefore does not route over a static graph.

Both algorithms operate inside the same dynamic physical simulation.

---

## Physical contact windows

Spacecraft are propagated through the configured orbital scenario.

Communication links appear and disappear as geometry changes.

A packet can only be transmitted when the entire transmission fits inside the available contact:

```text
departure_time + transmission_time <= contact_end_time
```

The simulator therefore cannot “cheat” by starting a large transfer immediately before contact disappears.

---

## Capacity is consumable

Every contact has finite capacity.

If one packet uses capacity, that capacity is no longer available to traffic arriving later.

One routing decision can therefore influence future routing opportunities.

---

## Transfers can fail

Transfer success is sampled from a seeded stochastic risk model.

Failure probability combines:

* baseline transfer risk;
* **weather risk**;
* spacecraft health;
* configured link reliability.

A failed transfer does not simply reset the simulation.

If transmission fails:

1. time advances to the point where the failure is detected;
2. some of the contact capacity has already been consumed;
3. the complete data bundle remains at the sender;
4. the routing system must retry or find another path.

Failures therefore have real consequences.

---

## Weather matters

Ground stations have different configured weather-risk values.

Ground-facing contacts inherit weather exposure into the stochastic transfer model.

That matters because real space communication systems do not operate in a vacuum once a signal reaches Earth.

Atmospheric conditions matter — particularly as future networks increasingly employ higher-frequency RF and optical communications.

A routing agent should eventually be capable of deciding not only:

> “Which ground station is visible?”

but also:

> “Which available ground station gives this scientific packet the best probability of useful delivery right now?”

---

# Scientific traffic is not all equal

The simulator generates scientific data products with different:

* source spacecraft;
* sizes;
* priorities;
* creation times;
* deadlines;
* data types.

Urgent **TRANSIENT** data receives high scientific priority and shorter time-to-live.

Other traffic includes:

* STAR_FIELD;
* CALIBRATION;
* HOUSEKEEPING.

This gives the routing agent a representation of **scientific value**, not simply bytes moving through a graph.

The broader vision is that this metadata would eventually originate from onboard scientific AI.

---

# Why this is a real technological direction

This project is not based on an imaginary future where satellites suddenly become networked and autonomous.

That transition has already started.

## NASA: store-and-forward networking is becoming infrastructure

NASA's Delay/Disruption Tolerant Networking work uses **store-and-forward** communication specifically because continuous end-to-end connectivity cannot be assumed in space.

NASA describes DTN as an important building block for future lunar and deep-space networking.

NASA's LunaNet work similarly envisions interoperable communications infrastructure around the Moon rather than every mission depending entirely on independent direct-to-Earth links.

The network architecture our prototype explores — spacecraft storing data, passing it between relay nodes, and eventually delivering it through an available ground path — belongs directly to that trajectory.

---

## ESA: relay constellations already move satellite data

ESA's European Data Relay System — often described as the **SpaceDataHighway** — already uses geostationary relay infrastructure to move information from lower-orbit Earth-observation satellites toward the ground with much lower delay than waiting for the next direct ground-station pass.

ESA is also developing **Moonlight**, a lunar communication and navigation infrastructure based on a satellite constellation and dedicated ground segment.

As the number of relays, missions, users and available paths increases, deciding **how traffic should move through those networks** becomes increasingly important.

That is the problem our prototype attacks.

---

# AI already decides which satellite data deserves attention

Another part of this future already exists.

NASA JPL's **OASIS — Onboard Autonomous Science Investigation System** was designed to analyse science data onboard a spacecraft and prioritise the most valuable information for transmission.

NASA has also tested AI-based autonomous Earth-observation targeting, where imagery is analysed onboard and the spacecraft decides how to respond.

ESA's **Φsat-2** runs AI applications directly onboard the spacecraft, including cloud detection. Instead of blindly transmitting every captured image, onboard processing can identify unusable data and prioritise useful information.

That establishes an important progression:

```text
Yesterday
--------
collect everything
      |
      v
send everything to Earth
      |
      v
analyse it


Today
-----
collect data
      |
      v
AI analyses / prioritises onboard
      |
      v
send the most useful information


Our next step
-------------
collect data
      |
      v
AI analyses / prioritises onboard
      |
      v
AI decides HOW that data should move
through the space communications network
      |
      v
researcher receives valuable data sooner
```

Our project asks:

> **Once onboard intelligence knows which data matters, should the network carrying that data become intelligent too?**

We think the answer is yes.

---

# Interactive mission-control prototype

The Streamlit application turns the experiment into an operator-facing system rather than a collection of scripts.

Users can explore:

* the multi-orbit constellation;
* currently available physical links;
* generated scientific packets;
* packets actively moving through the network;
* delivered packets;
* the complete route taken by a selected bundle;
* failed transmission attempts;
* packet size, type, priority and deadline;
* spacecraft metadata;
* ground-receiver metadata;
* benchmark scenarios;
* Temporal and PPO behaviour;
* locked experiment metadata.

The frontend is not a simplified simulation created for presentation.

The benchmark and replay both use the same final execution engine:

```text
src/experiment/runner.py
```

That means the visual demo is governed by the same routing, physical-contact, capacity and stochastic-transfer semantics as the benchmark.

---

# System architecture

```text
config/prototype.yaml
          |
          v
Orbital propagation
          |
          v
Physical snapshots
          |
          v
Time-dependent ContactPlan
          |
          +---------------------------+
          |                           |
          v                           v
Temporal router                 MaskablePPO
earliest arrival                AI next hop
          |                           |
          +-------------+-------------+
                        |
                        v
                 Capacity ledger
                        |
                        v
             Stochastic transfer oracle
         weather + health + reliability
                        |
              +---------+---------+
              |                   |
              v                   v
       transfer fails       transfer succeeds
       time/capacity lost   packet changes holder
              |                   |
              +---------+---------+
                        |
                        v
                  retry / reroute
                        |
                        v
                 ground receiver
```

---

# The learned policy

The final MaskablePPO policy uses:

* **14 routing actions**
* **158 observation features**
* **2,000,000 training timesteps**
* **4 parallel environments**
* **32 bundles per training episode**
* **training seed 42**

The action space represents possible next-hop nodes.

The observation is:

```text
4 bundle-level features

+

14 candidate destinations
×
11 candidate/link features

=

158 features
```

The policy operates **hop by hop**, allowing it to react to the state produced by its previous decision.

---

# Benchmarking is part of the product

We do not view benchmarking as something added for the competition.

It is a central capability.

If AI is ever going to make routing decisions on a real spacecraft, an operator needs to know:

* what deterministic routing would have done;
* what the AI did instead;
* whether latency improved;
* whether science return improved;
* how often fallback occurred;
* whether the policy is becoming unsafe;
* whether a new model is actually better than the deployed one.

Our simulator provides the beginnings of that evaluation environment.

A mission could eventually use it before deployment as a **routing digital twin**:

```text
new AI model
    |
    v
thousands of simulated mission conditions
    |
    +--> nominal geometry
    +--> congestion
    +--> link failures
    +--> bad weather
    +--> degraded spacecraft health
    +--> urgent science bursts
    +--> missed contacts
    |
    v
compare against deterministic safety policy
    |
    v
approve / reject / restrict operating envelope
```

The software is therefore not only a router.

It is also a **benchmarking environment for deciding whether an AI router deserves to fly.**

---

# Reproducibility

The reported experiment is locked by:

```text
config/final_experiment.json
```

The experiment definition identifies the expected:

* physical scenario;
* configuration hash;
* PPO checkpoint;
* traffic seed families;
* stochastic seed families;
* held-out scenario count;
* packets per scenario;
* routing limits;
* benchmark output location.

Committed benchmark evidence is available in:

```text
artifacts/final_experiment/benchmark/
├── summary.json
├── seed_metrics.csv
└── bundle_results.csv
```

Reproduce the benchmark with:

```bash
python run_final_benchmark.py
```

The repository additionally contains:

* regression tests;
* integration tests;
* release verification;
* a locked trained checkpoint;
* GitHub Actions CI;
* Python 3.11 / 3.12 validation.

---

# Run the prototype

Python 3.11+ is recommended.

```bash
python -m venv .venv

source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/final/verify_release.py

python run_frontend.py
```

The trained PPO checkpoint is included at:

```text
RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

No retraining is required.

To run the lightweight regression suite:

```bash
python -m pip install -r requirements-ci.txt

python -m pytest -q

python scripts/final/verify_release.py
```

---

# How IBM Bob helped us build it

IBM Bob was used as an AI engineering partner throughout development.

The project originally consisted of multiple independently developed technical systems:

1. the orbital simulator;
2. the reinforcement-learning environment and PPO model;
3. the Temporal routing implementation;
4. benchmarking;
5. the frontend.

The difficult part was turning these pieces into **one reproducible system**.

Bob assisted with:

* implementation;
* integration;
* debugging;
* interface design;
* regression testing;
* rapid frontend prototyping.

One of its most important roles was helping reason across component boundaries:

```text
orbital state
      |
      v
physical contact generation
      |
      v
routing observations / actions
      |
      v
capacity + stochastic transfers
      |
      v
benchmark execution
      |
      v
interactive replay
```

The components had to agree on:

* node identities;
* packet state;
* action semantics;
* contact windows;
* timing;
* capacity;
* transmission outcomes;
* model interfaces;
* benchmark configuration.

Bob also helped us rapidly iterate on the frontend so that a large amount of mission metadata could be available without overwhelming the main visualisation.

IBM Bob therefore acted not merely as a code generator, but as an **AI-assisted software engineering partner helping transform separate research components into an end-to-end product prototype.**

---

# Why this fits the August Space Exploration challenge

The August challenge asks how AI can make space exploration more intelligent, useful and capable of functioning in complex environments.

Our project applies AI at a foundational layer:

# **the movement of scientific information itself**

A spacecraft can make an extraordinary discovery.

But the discovery has limited value until researchers receive it.

| Judging criterion       | What this project demonstrates                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Technical Execution** | orbital propagation, physical contacts, capacity constraints, stochastic transfers, trained MaskablePPO, deterministic router, CI and reproducible benchmark |
| **Innovation**          | AI operates as a decision layer inside a dynamic space relay network rather than only analysing data after it reaches Earth                                  |
| **Challenge Fit**       | directly addresses how scientific data moves through future distributed space infrastructure                                                                 |
| **Feasibility**         | local next-hop inference, deterministic safety path, configurable simulation and clear DTN/relay-network relevance                                           |
| **Real-World Impact**   | lower data latency can increase the value and reaction speed of time-sensitive science                                                                       |

---

# Where we go next

The competition prototype proves the architecture and gives us something even more valuable:

**a benchmark against which every future routing model can be measured.**

The next product milestones are clear.

### 1. Operator-visible Hybrid Mode

Expose the existing scheduled / fallback capability directly in the dashboard.

An operator should be able to watch:

```text
AI → AI → AI → deadline guardrail → Temporal
```

and see exactly why the switch occurred.

### 2. Deadline rescue

When remaining deadline slack falls below a configured threshold, automatically route the packet using the deterministic earliest-arrival policy.

This directly targets the main weakness identified by the benchmark.

### 3. Model confidence and out-of-distribution gating

The AI should only receive authority when the current network resembles conditions for which it has demonstrated competence.

Unknown state?

Use the safe router.

### 4. Arbitrary constellation support

The competition checkpoint uses a fixed 14-node action space.

A production architecture should move toward a variable-size or graph-based policy capable of ingesting mission-specific constellations.

### 5. Real operational inputs

Replace simulation assumptions with external inputs where available:

* real ephemerides;
* contact plans;
* weather;
* ground-network availability;
* spacecraft state;
* link-health telemetry.

### 6. DTN interoperability

Map the routing intelligence onto Bundle Protocol / contact-plan concepts used by real Delay/Disruption Tolerant Networking systems.

### 7. Hardware-in-the-loop testing

Run policy inference on flight-representative edge hardware and benchmark:

* inference latency;
* memory;
* energy consumption;
* model corruption;
* reboot behaviour;
* deterministic fallback timing.

### 8. Multi-objective AI training

Retrain explicitly across:

* delivery ratio;
* deadline completion;
* successful-delivery latency;
* terminal backlog;
* scientific priority;
* transmission risk;
* wasted capacity;
* energy;
* route length.

The aim is not simply to make the PPO score higher.

It is to discover the **operating region in which learned routing creates mission value**.

---

# Scientific scope and limitations

This repository is a **simulation and proof of concept**.

It is not flight software and does not represent an operational satellite network.

The final trained policy currently uses a fixed 14-node interface.

The following are configured simulation assumptions rather than measurements from an operational mission:

* traffic distributions;
* packet priorities;
* packet deadlines;
* stochastic failure parameters;
* spacecraft health;
* battery state;
* weather-risk values;
* link properties;
* the 1,800-second simulation horizon.

The lower PPO delivery ratio should therefore be interpreted exactly as measured:

**the current learned policy delivers fewer packets within the locked experimental horizon than Temporal earliest-arrival routing.**

The benchmark does not justify assuming that unfinished packets would later be delivered.

Likewise, the lower per-attempt PPO failure rate should not be interpreted as proof of lower total mission risk, because PPO currently uses more hops and attempts.

That transparency is intentional.

The purpose of this project is not to prove that AI is universally superior to deterministic routing.

It is to build the environment required to answer the much more important engineering question:

> **Where is AI better, how much better is it, and when should we switch back to something we already trust?**

---

# Repository structure

```text
config/
    Scenario and locked experiment definitions

src/model/
    Orbital simulation utilities

src/models/
    DataBundle and ContactPlan models

src/routing/
    Temporal earliest-arrival router

src/integration/
    Contacts, capacity, stochastic transfers,
    traffic generation and PPO bridge

src/experiment/
    Shared experiment runner and benchmark

src/frontend/
    Interactive mission dashboard

RL/rl_env_v0/models/
    Final PPO checkpoint and metadata

artifacts/final_experiment/
    Committed benchmark evidence

tests/
    Regression and integration tests

scripts/final/
    Release verification

docs/final/
    Training, experiment and validation documentation
```

---

# Technology

* Python
* Stable-Baselines3
* sb3-contrib MaskablePPO
* Gymnasium
* NumPy
* Pandas
* Streamlit
* custom orbital/contact simulation
* stochastic seeded transfer modelling
* GitHub Actions
* IBM Bob

---

# Competition submission

**IBM AI Builders Challenge with IBM Bob**

### August Theme — Advance Space Exploration with AI

This repository contains:

* the working simulation;
* trained AI routing policy;
* deterministic safety baseline;
* stochastic communications environment;
* interactive dashboard;
* benchmark evidence;
* automated tests;
* reproducibility tooling.

---

# Our thesis

Space missions are becoming networks rather than isolated spacecraft.

Onboard AI is already beginning to decide **which scientific data matters**.

Relay constellations are beginning to change **how that data reaches Earth**.

The next step is to connect those two ideas.

> ## **The future spacecraft should not only understand its data.**
>
> ## **It should understand how to get that data home.**

And when the AI is uncertain, late, unavailable or outside its proven operating envelope:

### **there should always be a safe route home.**
