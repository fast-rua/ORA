# wait_ticks

Hold all currently applied pose/jaw commands for 1..20 native ticks. On
RoboDojo one tick is 10 physics steps / 40 ms. The trusted executor enforces
the parent and episode budgets and publishes state. Replay compares recorded
actions and receipts; no simulator fallback.
