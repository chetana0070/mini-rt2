# Current Project Status

## Milestone 1: UR5e + Robotiq140 Grasping

The original Robotiq85 setup was not producing reliable object contact for the milk grasp. Switching to Robotiq140Gripper solved the contact problem and produced successful lift behavior.

## Milestone 2: Demonstration Collection

A robomimic-style collector was created for UR5e PickPlace rollouts.

The collector stores:
- images
- robot states
- object states
- actions
- phase IDs
- success labels
- lift metrics

## Milestone 3: Low-Dim Behavior Cloning

The low-dimensional BC policy used simulator object position and target position.

Result:
- 18 successful demos
- 7/10 closed-loop successes

This proves that the manipulation behavior is learnable.

## Milestone 4: Image Behavior Cloning

The image BC policy used:
- agentview image
- robot end-effector position
- gripper joint position
- phase ID
- time

It excluded:
- object position
- target XY
- YOLO world estimate

Result:
- Offline imitation was good
- Closed-loop evaluation failed with small data

Interpretation:
The policy learned an average trajectory instead of robust image-based object localization.

## Current Direction

We are collecting digital cousin demonstrations in small chunks, then retraining image BC on a larger and more varied success-only dataset.

## Next Technical Steps

1. Collect 20-trial demo chunks
2. Filter success-only demos
3. Merge success demos
4. Retrain image BC
5. Evaluate closed-loop success rate
6. Add push and slide skills
7. Build kitchen and bedroom cousin scenes
8. Add Ollama task planner
