# Changelog – Unitree G1 Description

Upstream model history is preserved below, including variants that are not
bundled with FARO. FARO includes only `g1.xml` and its referenced meshes.

## [2025-08-7]
- Update the right hip joint limit of the upstream `g1_mjx.xml` to match the real robot limits

## [2025-05-30]

- Add MJX variant of [g1.xml](g1.xml), with manually designed collision geoms and contact pairs.

## [2024-12-10]

- Use updated models from Unitree's official [repo](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description) (sha: c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65).
  - Model without hands: [g1_29dof_rev_1_0](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml)
  - Model with hands: [g1_29dof_with_hand_rev_1_0](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml)

## [2024-05-20]

- Initial release.
