# Backbone Integrations

Integration code should remain thin: backbone-independent scoring and planning
live in `src/satome`, while these directories own sequence packing, model-layer
hooks, and task-specific state restoration.

An integration is not considered complete until it passes both:

1. a fixed-seed baseline-versus-SaToMe smoke test; and
2. parity against the corresponding accepted-paper benchmark configuration.
