# Brain-inspired software and biological boundaries

This internal evidence note supports the plain comparison in
[`USER_GUIDE.md`](USER_GUIDE.md). It records what the current OmniCortex source
actually implements, which biological observations motivate the terminology,
and where the analogy stops. It is not evidence of consciousness, biological
equivalence, human-level intelligence, trained quality, or perfect memory.

## Plain answer

OmniCortex is a digital adaptive model, not a simulated human organism. Its
learned state is stored in software contacts and tensors. Its activity is
computed on processors. Its sensors are admitted data and devices. Its working
memory is saved/transient machine state. Its growth, decay, resource limits and
inspection are explicit programs.

The design deliberately borrows several useful ideas from neuroscience:

- discrete units can become active and influence other units;
- connection efficacy can change with activity and timing;
- current activity and longer-lasting state use different timescales;
- memory can involve sparse, distributed ensembles rather than one stored
  sentence per fact;
- specialized regions can overlap and communicate instead of acting as
  isolated conventional experts;
- interference, decay, consolidation and resource pressure are part of
  learning rather than pretending retention is perfect.

Those similarities are functional analogies. The implementation does not copy
the molecular composition, cell diversity, anatomy, development, metabolism,
body, or lived experience of a biological brain.

## Exact software mapping

| User-facing term | Current software meaning | Important boundary |
| --- | --- | --- |
| Neuron | An indexed software unit with transient numeric state; the router uses `LIFPopulation` in `engine/omni_core/spiking.py`. | It is not a living cell and has no membrane, genes, organelles, or neurotransmitters. |
| Synapse/parameter | A packed ternary sign plus discrete strength/resistance/timing state, with independently identified repeated contacts. See `docs/v2/weighted-substrate.md`, `docs/v2/weighted-memory.md`, and `engine/omni_core/sparse_router_state.py`. | `-1`, `0`, and `+1` are computational signs. They do not identify a biological receptor or transmitter. |
| Spike | A thresholded LIF activity event used by the recurrent router. Exact captured spikes and contact transmissions are inspection evidence only when recorded at computation time. | A software spike is not an ion-driven action potential, and a nonzero connection is not proof that it fired. |
| STDP/plasticity | Bounded eligibility/timing updates can change native sign, integer strength/resistance, or allocate an admitted repeated contact. | This adapts a digital rule inspired by experiments; it is not a complete model of biological synaptic plasticity. |
| Working memory | Recent token and sensory activity, active vectors, paged workspace items, and recurrent liquid state in `engine/omni_core/brain.py`, `working_activity.py`, and `liquid.py`. | It has explicit machine capacities, paging and eviction; it is not a measured biological working-memory capacity. |
| Idea/assembly | A distributed native row/assembly and its actual recorded connections. Labels and indexes are inspection metadata, not a separate authoritative fact database. | An assembly does not prove semantic understanding or one-to-one correspondence with a human concept. |
| Routing/specialization | One interconnected brain has cortical, sensory, memory, action, imagination and sparse recurrent routing regions. The whole-brain map uses actual saved owners/links and refuses invented cross-region connections. | A software region name is not proof that it reproduces a named anatomical area or all of its functions. |
| Senses | Text plus explicitly admitted image, audio, video, screen, camera, microphone or file data enters typed native modality paths. | A decoder and device lease are not biological vision, hearing, touch, interoception or embodiment. |
| Forgetting | Activity decay, interference, workspace eviction, stability/resistance, later learning and explicit deletion can change what remains active or saved. | Persistence is not guaranteed recall, and loss or change is not claimed to reproduce a particular human forgetting mechanism. |
| Traceability | Studio can page exact recorded contacts, strengths, spikes, transmissions, actions, provenance and receipts. | Stored numbers do not reveal subjective experience, hidden intentions, a complete meaning map, or a guaranteed factual explanation. |

The native sign plane uses two physical bits per logical ternary sign, but a
complete contact and brain require strength/resistance bytes, repeated-contact
endpoints, timing/control state, indexes, caches, activity, storage and runtime
overhead. Neither this note nor the product assigns the biological brain an
equivalent bit capacity.

## Primary biological evidence used for the comparison

These are research anchors for specific observations, not validation studies of
OmniCortex:

1. Hodgkin and Huxley's voltage-clamp work gave a quantitative account of ionic
   membrane currents and nerve excitation. It supports distinguishing a real
   electrochemical action potential from a software LIF event:
   [Hodgkin & Huxley, 1952](https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/).
2. Long-lasting changes in synaptic transmission were recorded after patterned
   stimulation in the rabbit dentate area:
   [Bliss & Gardner-Medwin, 1973](https://pubmed.ncbi.nlm.nih.gov/4727085/).
3. Correlated pre/post spiking changed cultured hippocampal synapses, with the
   direction and extent depending on timing, starting strength and cell type:
   [Bi & Poo, 1998](https://pubmed.ncbi.nlm.nih.gov/9852584/). This motivates
   timing-sensitive digital plasticity but does not make the rule biologically
   complete.
4. Task-related firing persisted through a delay in monkey prefrontal and
   mediodorsal thalamic neurons:
   [Fuster & Alexander, 1971](https://pubmed.ncbi.nlm.nih.gov/4998337/). This is
   a biological observation behind the working-activity analogy, not a fixed
   token-window formula.
5. Mouse experiments identified sparse memory-related ensembles and showed
   recall after their optogenetic reactivation:
   [Liu et al., 2012](https://pubmed.ncbi.nlm.nih.gov/22441246/). Later mapping
   reported engram ensembles distributed across multiple regions:
   [Roy et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35379803/). These results
   motivate distributed assemblies; they do not show that any particular
   OmniCortex assembly has a human memory's meaning.
6. Manipulating hippocampal neurogenesis changed forgetting in several animal
   experiments:
   [Akers et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24812394/). It supports
   treating forgetting and circuit change as real biological phenomena, not
   equating OmniCortex's explicit decay rule with neurogenesis.

## Claims the product may make

The current implementation can accurately say that it is:

- brain-inspired and natively adaptive;
- learned through packed ternary contacts with integer strength and repeated
  connection state;
- equipped with LIF-style recurrent activity, timing-sensitive plasticity,
  working/liquid state, distributed assemblies and multimodal pathways;
- designed as one interconnected software brain with overlapping regions;
- inspectable at the level of exact stored and recorded software state.

It must not claim that it:

- is conscious, sentient, alive, biologically equivalent, or a digital human;
- faithfully simulates all neuron, synapse, chemical, anatomical or bodily
  processes;
- has a known brain-equivalent parameter or bit count;
- remembers everything, never forgets, understands every stored item, or can
  recover a complete fact list from its weights;
- exposes private chain of thought merely because it can emit a voluntary
  model-authored Ponder self-report;
- has human-level or AGI capability without direct trained evidence.

## Evidence boundary

This note was reconciled against the current weighted-substrate, working-memory,
router, modality and whole-brain inspection source. The cited experiments
support the biology column only. Focused software tests establish particular
representations and transactions; they do not compare OmniCortex behavior with
a biological brain. No brain was created, loaded or trained for this document,
and no biological-capacity estimate was derived.
