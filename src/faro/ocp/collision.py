import casadi as ca
import pinocchio.casadi as cpin
from faro.configs.identifiers import interface_name


def contact_collision_multiplier(scene, features, group, pair, params, *, trajectory=False):
    """Exclude only the collision geometry participating in a declared contact."""
    def object_target(source, name):
        return features.object_weight(name, params[interface_name(source) + '_env_feat_oh'])

    def ee_target(source, target):
        return params[interface_name(source) + '_env_feat_oh'][scene.features.ee_ids[target] - 1]

    def patches(rid, capsule):
        return [('ee', rid, patch) for patch, names in scene.robots[rid].contact_capsules.items()
                if capsule in names]

    factor = 1
    if group == 'mv_box_mv_box':
        factor *= (1 - object_target(('obj', pair.name_i), pair.name_j))
        factor *= (1 - object_target(('obj', pair.name_j), pair.name_i))
    elif group == 'mv_box_plane' or (trajectory and group == 'mv_box_st_box'):
        if scene.include_robot_ees:
            factor *= 1 - object_target(('obj', pair.name_i), pair.name_j)
        else:
            factor *= 1 - ca.sum1(params[pair.name_i + '_env_feat_oh'])
    elif group == 'robot_plane':
        patch = scene.robots[pair.robot_i].plane_contact_exemptions.get(pair.local_i)
        if patch is not None:
            factor *= 1 - object_target(('ee', pair.robot_i, patch), pair.name_j)

    if not scene.include_robot_ees:
        return factor
    if group in ('self_coll', 'robot_robot'):
        for a in patches(pair.robot_i, pair.local_i):
            for b in patches(pair.robot_j, pair.local_j):
                if a != b:
                    factor *= (1 - ee_target(a, b)) * (1 - ee_target(b, a))
    elif group in ('robot_mv_box', 'robot_st_box', 'robot_plane'):
        for source in patches(pair.robot_i, pair.local_i):
            factor *= 1 - object_target(source, pair.name_j)
            if pair.name_j in scene.env_objects:
                factor *= 1 - ee_target(('obj', pair.name_j), source)
    return factor


def object_poses(states):
    """Convert each object pose once, independently of collision pair count."""
    poses = {}
    for name, state in states.items():
        transform = cpin.XYZQUATToSE3(state[:7])
        poses[name] = (ca.vec(transform.rotation), transform.translation)
    return poses


def collision_distances(groups, robot_poses, env_poses, params):
    """Return raw distances; activation and bounds belong to each formulation."""
    endpoint_poses = {
        "self_coll": (robot_poses, robot_poses),
        "robot_robot": (robot_poses, robot_poses),
        "mv_box_mv_box": (env_poses, env_poses),
        "mv_box_st_box": (env_poses, env_poses),
        "robot_mv_box": (robot_poses, env_poses),
        "robot_st_box": (robot_poses, env_poses),
        "robot_plane": (robot_poses, env_poses),
        "mv_box_plane": (env_poses, env_poses),
    }
    distances = {}
    distance = local_approx_sign_dist()
    for group, (poses_i, poses_j) in endpoint_poses.items():
        distances[group] = {}
        for name, pair in groups.get(group, {}).items():
            rotation_i, position_i = poses_i[pair.name_i]
            rotation_j, position_j = poses_j[pair.name_j]
            collision = params[name]
            distances[group][name] = distance(
                position_i, rotation_i, position_j, rotation_j,
                collision[:3], collision[3:6], collision[6:9],
            )
    return distances

def local_approx_sign_dist():
    p1_W, R1_W = ca.SX.sym("p1_W", 3), ca.SX.sym("R1_W", 9)
    p2_W, R2_W = ca.SX.sym("p2_W", 3), ca.SX.sym("R2_W", 9)

    p1_coll_loc = ca.SX.sym("p1_coll_loc", 3)
    p2_coll_loc = ca.SX.sym("p2_coll_loc", 3)

    n = ca.SX.sym("n", 3)

    R1_W_ = ca.reshape(R1_W, 3, 3)
    R2_W_ = ca.reshape(R2_W, 3, 3)

    p1_coll_W =  R1_W_ @ p1_coll_loc + p1_W
    p2_coll_W =  R2_W_ @ p2_coll_loc + p2_W

    distance = n.T @ (p2_coll_W - p1_coll_W)

    return ca.Function(
        "local_approx_sign_dist",
        [p1_W, R1_W, p2_W, R2_W, p1_coll_loc, p2_coll_loc, n],
        [distance],
        ["p1_W", "R1_W", "p2_W", "R2_W", "p1_coll_loc", "p2_coll_loc", "n"],
        ["distance"],
    )
