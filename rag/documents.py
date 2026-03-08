"""
rag/documents.py
================
Source documents for the RAG knowledge base.

Each document is manually chunked into semantically coherent passages.
Manual chunking is intentional — automatic sentence splitting often
breaks mid-concept in technical papers, producing poor retrieval quality.

Each chunk has:
  - id:       unique identifier
  - text:     the passage content
  - metadata: source, section, topic tags for filtering
"""

# ---------------------------------------------------------------------------
# ACN-Data Paper — Lee, Li & Low (2019)
# "ACN-Data: Analysis and Applications of an Open EV Charging Dataset"
# e-Energy '19, Phoenix, AZ. DOI: 10.1145/3307772.3328313
# ---------------------------------------------------------------------------

ACN_PAPER_CHUNKS = [
    {
        "id": "acn_overview",
        "text": (
            "ACN-Data is a publicly available dataset of workplace EV charging released by "
            "researchers at Caltech. It was collected from Adaptive Charging Networks (ACNs) "
            "operated by PowerFlex Systems at two sites in California: the Caltech campus in "
            "Pasadena and the Jet Propulsion Laboratory (JPL) in La Canada. The dataset contains "
            "over 30,000 charging sessions and is updated daily with a two-week privacy delay. "
            "It is available at https://ev.caltech.edu/dataset."
        ),
        "metadata": {"source": "acn_paper", "section": "introduction", "topic": "dataset_overview"},
    },
    {
        "id": "acn_sites",
        "text": (
            "The Caltech ACN has 54 EVSEs (Electric Vehicle Supply Equipment, i.e. charging stations) "
            "in a campus parking garage, plus a 50 kW DC fast charger. It is open to the public and "
            "used by faculty, staff, students, and community members. Because it is near the campus gym, "
            "many drivers charge while exercising in the morning or evening. "
            "The JPL ACN has 52 EVSEs in a parking garage. Access is restricted to JPL employees only, "
            "making it representative of pure workplace charging. EV penetration is very high at JPL, "
            "leading to high EVSE utilization and an informal culture where drivers move their vehicles "
            "after charging completes to free up spaces for others."
        ),
        "metadata": {"source": "acn_paper", "section": "dataset", "topic": "sites"},
    },
    {
        "id": "acn_data_fields",
        "text": (
            "Each charging session in ACN-Data contains the following key fields: "
            "connectionTime (when the EV plugged in), doneChargingTime (time of last non-zero current), "
            "disconnectTime (when the EV unplugged), kWhDelivered (measured energy delivered in kWh), "
            "siteID (site identifier), stationID (unique EVSE identifier), sessionID (unique session ID), "
            "timezone, pilotSignal (time series of pilot signals), chargingCurrent (time series of actual "
            "charging current), userID (unique user identifier, only for claimed sessions), "
            "requestedDeparture (user estimated departure), and kWhRequested (estimated energy demand). "
            "Sessions where a user scanned the QR code and provided input via the mobile app are called "
            "'claimed' sessions. Sessions without user input are called 'unclaimed'."
        ),
        "metadata": {"source": "acn_paper", "section": "dataset", "topic": "data_fields"},
    },
    {
        "id": "acn_adaptive_charging",
        "text": (
            "The Adaptive Charging Network (ACN) uses an online scheduling algorithm based on model "
            "predictive control to deliver each driver's requested energy before their stated departure "
            "time, without exceeding infrastructure capacity. The algorithm optimises charging rates "
            "across all connected EVs simultaneously. Infrastructure elements such as transformers are "
            "deliberately oversubscribed to reduce capital costs, relying on the scheduling algorithm "
            "to manage demand. This means the total simultaneous charging demand is actively managed "
            "rather than each EVSE operating at full power independently."
        ),
        "metadata": {"source": "acn_paper", "section": "dataset", "topic": "charging_algorithm"},
    },
    {
        "id": "acn_weekday_weekend",
        "text": (
            "Both the Caltech and JPL sites show much higher utilization on weekdays than weekends, "
            "consistent with workplace charging patterns. JPL, as a closed campus, has near-zero "
            "charging on weekends and holidays. Caltech, being an open university campus, has "
            "non-trivial weekend usage. Weekday arrival distributions show a morning peak at both sites. "
            "Departures increase as the workday ends, peaking around 5-6pm at both sites. "
            "JPL departures tend to start earlier, consistent with earlier arrival times. "
            "Caltech departures stretch later into the evening due to heterogeneous schedules "
            "and community users."
        ),
        "metadata": {"source": "acn_paper", "section": "user_behavior", "topic": "weekday_weekend_patterns"},
    },
    {
        "id": "acn_pricing_effects",
        "text": (
            "ACN-Data captures the effect of pricing policy changes on charging behavior. "
            "The Caltech ACN was free for its first 2.5 years of operation. When a fee of $0.12/kWh "
            "was introduced on November 1, 2018, both the number of sessions per day and daily energy "
            "delivered decreased significantly. The evening peak around 6pm (attributed to community "
            "users) disappeared after pricing was introduced. The morning arrival peak increased as a "
            "proportion, reflecting a higher share of regular employees with standard work schedules. "
            "At JPL, where demand for charging is high enough to overshadow price sensitivity, "
            "no significant decrease in utilization was observed after pricing was introduced."
        ),
        "metadata": {"source": "acn_paper", "section": "user_behavior", "topic": "pricing_effects"},
    },
    {
        "id": "acn_driver_laxity",
        "text": (
            "Driver laxity measures how much scheduling flexibility exists in a charging session. "
            "It is defined as LAX(i) = d_i - e_i/r_max_i, where d_i is session duration, e_i is "
            "energy demand, and r_max_i is the maximum charging rate. A laxity of zero means the "
            "EV must charge at maximum rate for the entire session to meet demand — no flexibility. "
            "Higher laxity means more flexibility for the scheduler to shift charging in time. "
            "Most weekday sessions at both Caltech and JPL display high laxity, meaning there is "
            "substantial opportunity for demand response and load shifting without inconveniencing drivers. "
            "Weekend sessions tend to have lower laxity, as drivers want to charge quickly and leave."
        ),
        "metadata": {"source": "acn_paper", "section": "user_behavior", "topic": "driver_laxity"},
    },
    {
        "id": "acn_user_input_reliability",
        "text": (
            "A key finding from ACN-Data is that user-provided estimates of departure time and energy "
            "demand are unreliable. Users have little incentive to provide accurate predictions. "
            "Machine learning models trained on historical behavior — specifically Gaussian Mixture "
            "Models (GMMs) — significantly outperform user input for predicting session duration and "
            "energy demand. For Caltech, individual-level GMMs achieved a SMAPE of 15.85% for duration "
            "vs 25.81% for user input. For JPL, 12.25% vs 18.60%. This has important implications for "
            "charging schedulers that rely on user input: statistical prediction from past behavior "
            "produces better outcomes than trusting self-reported estimates."
        ),
        "metadata": {"source": "acn_paper", "section": "user_behavior", "topic": "user_input_reliability"},
    },
    {
        "id": "acn_utilization_definition",
        "text": (
            "In the context of EV charging infrastructure, utilization rate refers to how intensively "
            "charging equipment is being used. It can be measured in several ways: "
            "(1) Session-based utilization: number of charging sessions per EVSE per day. "
            "(2) Energy-based utilization: total kWh delivered per EVSE per day. "
            "(3) Occupation-based utilization: proportion of time an EVSE is occupied by a connected vehicle. "
            "High utilization indicates charging infrastructure is being heavily used and may indicate "
            "a need for capacity expansion. Low utilization suggests underused infrastructure. "
            "In ACN-Data, utilization is measured at both the session and energy level and varies "
            "significantly between weekdays and weekends, and between the Caltech and JPL sites."
        ),
        "metadata": {"source": "acn_paper", "section": "concepts", "topic": "utilization_definition"},
    },
    {
        "id": "acn_energy_demand",
        "text": (
            "Energy demand in EV charging sessions is measured in kilowatt-hours (kWh). "
            "kWhDelivered is the actual measured energy delivered to the vehicle during a session. "
            "kWhRequested is the energy amount the user estimated they needed when they started the session. "
            "These two values often differ: users may underestimate or overestimate their needs. "
            "The ACN scheduling algorithm attempts to deliver the requested energy before departure, "
            "but may deliver more (if the battery is not full) or less (due to congestion or early departure). "
            "Average energy per session varies by site, time of day, and user behavior patterns."
        ),
        "metadata": {"source": "acn_paper", "section": "concepts", "topic": "energy_demand"},
    },
    {
        "id": "acn_solar_integration",
        "text": (
            "ACN-Data has been used to study optimal sizing of on-site solar generation for workplace "
            "EV charging. Using Southern California Edison time-of-use rates and learned charging "
            "distributions from ACN-Data, researchers found that approximately 76 kW of solar capacity "
            "at Caltech could meet about 50% of EV charging demand from solar, saving approximately "
            "$1,000 per month in summer. JPL users tend to arrive earlier than Caltech users, allowing "
            "charging to be scheduled to avoid on-peak rates, which reduces the marginal benefit of "
            "on-site solar at higher solar costs. As solar LCOE decreases, on-site generation becomes "
            "increasingly valuable for reducing both cost and carbon footprint of workplace EV charging."
        ),
        "metadata": {"source": "acn_paper", "section": "applications", "topic": "solar_integration"},
    },
    {
        "id": "acn_duck_curve",
        "text": (
            "The Duck Curve refers to the shape of net electricity demand on the grid when solar "
            "generation is subtracted — it dips in the middle of the day (solar peak) and ramps "
            "sharply upward in the evening (solar drops, demand rises). Controlled EV charging "
            "can help smooth this curve by shifting charging to midday when solar is abundant. "
            "Using ACN-Data distributions, researchers showed that with 2 million EVs under "
            "adaptive control, up and down ramping requirements could be reduced by nearly 50% "
            "with only a 0.6% increase in peak demand. The JPL charging distribution — concentrated "
            "during working hours — is more effective for Duck Curve smoothing than the Caltech "
            "distribution, which includes evening community charging."
        ),
        "metadata": {"source": "acn_paper", "section": "applications", "topic": "duck_curve_grid"},
    },
    {
        "id": "acn_dataset_limitations",
        "text": (
            "ACN-Data has several important limitations to be aware of when drawing conclusions: "
            "(1) Workplace charging only — Caltech and JPL represent workplace and semi-public charging. "
            "Residential and public fast charging behavior may differ significantly. "
            "(2) Geographic bias — both sites are in Southern California, which has mild weather, "
            "high EV adoption, and specific electricity tariff structures that may not generalise globally. "
            "(3) Single network operator — all data is from PowerFlex-managed ACNs using adaptive "
            "scheduling, so charging patterns partly reflect the algorithm's behaviour, not purely "
            "natural driver demand. "
            "(4) Temporal coverage — the dataset spans approximately 2018–2021. EV adoption, vehicle "
            "efficiency, and user behaviour have continued to evolve since then."
        ),
        "metadata": {"source": "acn_paper", "section": "limitations", "topic": "dataset_limitations"},
    },
    {
        "id": "acn_chargegpt_context",
        "text": (
            "ChargeGPT is a conversational AI assistant built on top of ACN-Data for the purpose of "
            "enabling natural language interaction with EV charging infrastructure data. "
            "It combines two components: a structured query engine that translates natural language "
            "questions into SQL queries executed against a DuckDB database of ACN-Data sessions, "
            "and a RAG (Retrieval-Augmented Generation) knowledge base containing the ACN-Data paper "
            "and related documentation. Numerical questions about sessions, energy, utilization, and "
            "patterns are answered by the structured query engine using real data. Conceptual and "
            "methodological questions are answered using the RAG knowledge base. "
            "The system is designed to reduce hallucination in LLM responses about EV charging data "
            "by grounding answers in real dataset-derived evidence."
        ),
        "metadata": {"source": "project", "section": "system", "topic": "chargegpt_overview"},
    },
]


def get_all_chunks() -> list[dict]:
    """Return all document chunks for indexing."""
    return ACN_PAPER_CHUNKS


def get_chunks_by_topic(topic: str) -> list[dict]:
    """Filter chunks by topic tag."""
    return [c for c in ACN_PAPER_CHUNKS if c["metadata"].get("topic") == topic]

# Append out-of-scope boundary chunk
ACN_PAPER_CHUNKS.append({
    "id": "acn_out_of_scope",
    "text": (
        "ACN-Data does not contain grid carbon intensity data, electricity prices, "
        "weather data, vehicle make or model information, or battery capacity data. "
        "Questions about carbon intensity, CO2 emissions, emissions implications, "
        "or carbon footprint of EV charging cannot be answered directly from ACN-Data alone. "
        "Carbon intensity varies by grid region and time of day based on the generation mix "
        "(solar, wind, gas, coal). In California, the CAISO grid tends to have lower carbon "
        "intensity midday when solar generation peaks, and higher intensity in the evening "
        "when solar drops and gas peakers ramp up. To answer emissions questions precisely, "
        "ACN-Data would need to be combined with a grid carbon intensity dataset such as "
        "WattTime or the CAISO API. The ACN-Data paper does discuss the potential for "
        "adaptive EV charging to smooth the Duck Curve and reduce grid ramping requirements, "
        "which indirectly reduces the need for high-carbon peaker plants."
    ),
    "metadata": {"source": "project", "section": "limitations", "topic": "out_of_scope"},
})