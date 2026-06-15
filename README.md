# Digital Cousin Generalization for Vision-Based UR5e Household Manipulation

This project builds a simulated household manipulation pipeline for a UR5e robot with a Robotiq140 gripper. The system learns manipulation skills from Robosuite/MuJoCo demonstrations and studies whether digital cousin data improves vision-based robot generalization.

## Project Goal

The long-term goal is to build a language-guided household manipulation system:

User instruction -> Ollama task planner -> scene/object/skill selection -> perception -> learned robot policy -> UR5e execution -> success evaluation

Example tasks:
- Pick the milk from a kitchen counter
- Place the can into a tray
- Slide the book on a bedroom table
- Push a cup away from the table edge
- Clear the bedside table

## Current Robot Setup

- Simulator: Robosuite / MuJoCo
- Robot: UR5e
- End effector: Robotiq140Gripper
- Base task: PickPlace
- Camera: agentview
- Current object: milk
- Control output: 7D action: x, y, z, rx, ry, rz, gripper

## Current Pipeline

1. Build UR5e + Robotiq140 environment
2. Create a stable side-grasp expert teacher
3. Collect raw demonstrations
4. Filter success-only demonstrations
5. Train low-dimensional oracle behavior cloning
6. Evaluate low-dimensional policy closed-loop
7. Train image-conditioned behavior cloning
8. Evaluate image policy closed-loop
9. Collect digital cousin demonstration chunks
10. Retrain image policy on larger varied data
11. Expand to kitchen and bedroom environments
12. Add multiple manipulation skills
13. Add Ollama as a language task planner

## Implemented Components

### Demo Collection

Main script:

scripts/ur5e_bc/collect_ur5e_demos.py

The collector generates robomimic-style HDF5 rollouts with:
- actions
- simulator states
- rewards
- dones
- agentview images
- robot end-effector position
- gripper joint state
- object position
- YOLO world estimate
- target XY
- phase ID
- success flag
- lift metrics

Stable teacher configuration:
- robot: UR5e
- gripper: Robotiq140Gripper
- grasp mode: side
- side: x_minus
- side_pre_dist: 0.140
- side_contact_dist: -0.010
- side_z_add: 0.030
- q_low: 0.40
- q_high: 0.58
- close_max_steps: 140
- horizon: 900

### Low-Dimensional Behavior Cloning

Scripts:
- scripts/ur5e_bc/train_lowdim_bc.py
- scripts/ur5e_bc/check_lowdim_bc_offline.py
- scripts/ur5e_bc/eval_lowdim_bc_closed_loop.py

Inputs:
- object_pos
- target_xy
- robot0_eef_pos
- robot0_gripper_qpos
- phase_id
- time

Result:
- 18 success-only demonstrations
- Low-dimensional closed-loop policy reached 7/10 successes

This is treated as the oracle upper-bound baseline because it uses privileged simulator information.

### Image-Conditioned Behavior Cloning

Scripts:
- scripts/ur5e_bc/train_image_bc.py
- scripts/ur5e_bc/check_image_bc_offline.py
- scripts/ur5e_bc/eval_image_bc_closed_loop.py

Inputs:
- agentview_image
- robot0_eef_pos
- robot0_gripper_qpos
- phase_id
- time

Excluded privileged inputs:
- object_pos
- target_xy
- yolo_world

Result:
- Image BC trained successfully
- Offline imitation looked good
- Closed-loop image policy failed with small data

Interpretation:
The 18-demo image policy learned an average trajectory instead of robust visual localization. This motivates digital cousin data expansion.

## Digital Cousin Direction

A digital twin is one fixed scene that closely matches a target setup.

A digital cousin is a similar scene with the same semantic and geometric affordance, but with varied object positions, layouts, lighting, distractors, and appearances.

Current digital cousin plan:
- Collect many successful milk-pick demos with varied object positions
- Retrain image BC on the larger varied dataset
- Compare small-data image BC vs cousin-trained image BC
- Expand to kitchen and bedroom scenes

## Planned Kitchen Environment

Objects:
- milk
- can
- bread
- bowl
- plate
- spoon
- tray
- counter

Skills:
- pick
- place
- push
- slide
- later: open drawer
- later: press button
- later: wipe counter

## Planned Bedroom Environment

Objects:
- book
- remote
- cup
- water bottle
- phone
- bedside table
- tray

Skills:
- pick
- place
- push
- slide
- later: open drawer
- later: press lamp switch
- later: clear bedside table

## Planned Skill Library

Initial skills:
- pick
- place
- push
- slide

Advanced skills:
- open drawer
- close drawer
- press button
- wipe surface
- sort objects
- clear tabletop

## Ollama Planner

Ollama will be used as a high-level task planner, not as a raw robot controller.

Example input:

Pick the milk from the kitchen counter.

Example structured output:

{
  "scene": "kitchen",
  "object": "milk",
  "skill": "pick",
  "target": null
}

The learned robot policy performs the actual low-level control.

## Repository Notes

Datasets and checkpoints are intentionally not committed to Git by default.

Ignored local outputs:
- datasets/
- runs/

To share large datasets or model checkpoints, use Git LFS or GitHub release artifacts.

## Current Research Question

Can digital cousin training improve vision-based UR5e manipulation generalization across kitchen and bedroom scenes?

## Baselines

Planned comparisons:
- Oracle low-dimensional BC
- Small-data image BC
- Digital-cousin image BC
- YOLO-assisted BC
- Ollama-guided multi-skill policy

## Status

Completed:
- UR5e + Robotiq140 working setup
- Stable side-grasp teacher
- Robomimic-style dataset writer
- Low-dimensional BC training and evaluation
- Image-conditioned BC training and offline validation
- Initial closed-loop image BC failure analysis

In progress:
- Digital cousin demonstration collection
- Larger success-only image BC dataset

Next:
- Collect demo chunks
- Filter success-only demos
- Retrain image BC
- Evaluate closed-loop image BC
- Add push/slide skills
- Build kitchen and bedroom cousin scenes
