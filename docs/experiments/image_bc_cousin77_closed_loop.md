# Image BC Cousin77 Closed-Loop Evaluation

## Dataset

- Raw demos: 80
- Success-only demos: 77
- Dataset: datasets/ur5e_robomimic/processed/ur5e_milk_r140_cousin77_success_only.hdf5

## Model

- Checkpoint: runs/ur5e_bc/image_milk_r140_cousin77_bc.pt
- Inputs:
  - agentview_image
  - robot0_eef_pos
  - robot0_gripper_qpos
  - phase_id
  - time
- Excluded privileged inputs:
  - object_pos
  - target_xy
  - yolo_world

## Offline Training Result

- Best validation MSE: 0.01394
- This improved over the earlier 18-demo image BC model.

## Closed-Loop Result

- Visible evaluation: 0/1
- Batch evaluation: 0/10

## Failure Mode

The policy follows the demonstrated phase structure but does not reliably ground its approach in the object location. During side approach, the gripper can push the milk away or knock it down before the close phase. The policy then continues closing and lifting even though the object is no longer grasped.

## Interpretation

Digital cousin data improved offline imitation loss but did not solve closed-loop visual grounding for pure image behavior cloning.

## Next Step

Build a YOLO-assisted or object-conditioned policy using explicit object target information in addition to image and robot state.
