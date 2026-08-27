import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


PROJECT_ROOT = Path(".")

POSES = ["ok", "paper", "rock", "scissors", "thefinger"]
START_INDEX = 0
LIMIT = 200
SAMPLES = []
SAMPLES_BY_POSE = {
}

FITTED_DIR = None
OUT_ROOT = None

SKIP_EXISTING = True
FLAT_OUTPUT = False

ARMATURE_NAME = "Rig"
HAND_OBJECTS = ["Hand", "Nails"]

CAMERA_ANGLES_DEG = [-45, 0, 30, 45, 60, 90, 120, 135]
CAMERA_RADIUS = 40.0
CAMERA_HEIGHT = 8.0
ORTHO_SCALE = 38.0
CAMERA_TYPE = "PERSP"  # "PERSP" or "ORTHO"
CAMERA_LENS = 40.0
RENDER_RES = 256
RENDER_ENGINE = "CYCLES"
CYCLES_SAMPLES = 64
FORCE_WORLD_CAMERA_VISIBLE = True

USE_BONE_SCALE = False
BONE_SCALE_MIN = 0.75
BONE_SCALE_MAX = 1.25

PREFIX = "BATCH_RENDER_"
DBG_PREFIX = "HIER_DBG_"
SUPPORTED_POSES = {"ok", "paper", "rock", "scissors", "thefinger", "thumbup", "thumbdown"}
POSE_ALIASES = {
    "the-finger": "thefinger",
    "the_finger": "thefinger",
    "middlefinger": "thefinger",
    "scrissors": "scissors",
}

BONE_SEGMENTS = [
    ("thumb_trapez", 0, 1),
    ("thumb_meta", 1, 2),
    ("thumb_prox", 2, 3),
    ("thumb_dist", 3, 4),

    ("index_meta", 0, 5),
    ("index_prox", 5, 6),
    ("index_midd", 6, 7),
    ("index_dist", 7, 8),

    ("midd_meta", 0, 9),
    ("midd_prox", 9, 10),
    ("midd_midd", 10, 11),
    ("midd_dist", 11, 12),

    ("ring_meta", 0, 13),
    ("ring_prox", 13, 14),
    ("ring_midd", 14, 15),
    ("ring_dist", 15, 16),

    ("pinky_meta", 0, 17),
    ("pinky_prox", 17, 18),
    ("pinky_midd", 18, 19),
    ("pinky_dist", 19, 20),
]


def get_obj(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Missing object: {name}")
    return obj


def ensure_object_mode():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def clear_rig_animation(arm):
    arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()


def set_active_pose(arm):
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    arm.hide_viewport = False
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")


def clear_pose(arm):
    set_active_pose(arm)
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.context.view_layer.update()


def show_render_objects():
    for obj in bpy.data.objects:
        if obj.name.startswith(DBG_PREFIX):
            obj.hide_viewport = True
            obj.hide_render = True

    ico = bpy.data.objects.get("Icosphere")
    if ico:
        ico.hide_viewport = True
        ico.hide_render = True

    arm = bpy.data.objects.get(ARMATURE_NAME)
    if arm:
        arm.hide_viewport = False
        arm.hide_render = True

    for name in HAND_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_viewport = False
            obj.hide_render = False


def rotate_posebone_y_to_target(arm, bone_name, target_dir_world):
    pb = arm.pose.bones[bone_name]

    target_dir_world = target_dir_world.normalized()
    current_dir_world = (arm.matrix_world.to_3x3() @ pb.y_axis).normalized()

    if current_dir_world.length < 1e-8 or target_dir_world.length < 1e-8:
        print("[SKIP invalid dir]", bone_name)
        return

    q_world = current_dir_world.rotation_difference(target_dir_world)

    head_arm = pb.head.copy()
    r_obj = arm.matrix_world.to_3x3()
    q_arm = r_obj.inverted().to_quaternion() @ q_world @ r_obj.to_quaternion()

    t1 = Matrix.Translation(head_arm)
    t2 = Matrix.Translation(-head_arm)
    r4 = q_arm.to_matrix().to_4x4()

    pb.matrix = t1 @ r4 @ t2 @ pb.matrix
    bpy.context.view_layer.update()


def rest_bone_lengths_world(arm):
    lengths = {}
    r_obj = arm.matrix_world.to_3x3()

    for bone_name, _, _ in BONE_SEGMENTS:
        pb = arm.pose.bones[bone_name]
        rest_vec = r_obj @ (pb.tail - pb.head)
        lengths[bone_name] = max(rest_vec.length, 1e-8)

    return lengths


def apply_bone_length_scale(arm, bone_name, target_len, rest_lengths, scale_min, scale_max):
    pb = arm.pose.bones[bone_name]
    scale_y = target_len / rest_lengths[bone_name]
    scale_y = max(scale_min, min(scale_max, scale_y))
    pb.scale.y = scale_y
    bpy.context.view_layer.update()


def apply_retarget(fitted_npy, use_bone_scale=True, bone_scale_min=0.75, bone_scale_max=1.25):
    arm = get_obj(ARMATURE_NAME)

    clear_rig_animation(arm)
    clear_pose(arm)
    bpy.context.view_layer.update()
    rest_lengths = rest_bone_lengths_world(arm)

    fitted = np.load(fitted_npy).astype(np.float32)
    if fitted.shape != (21, 3):
        raise ValueError(f"Expected (21,3), got {fitted.shape}: {fitted_npy}")

    wrist_world = arm.matrix_world @ arm.pose.bones["radius_ulna"].tail
    target_world_pts = [wrist_world + Vector(fitted[i].tolist()) for i in range(21)]

    set_active_pose(arm)

    for bone_name, j0, j1 in BONE_SEGMENTS:
        target_dir = target_world_pts[j1] - target_world_pts[j0]
        if target_dir.length < 1e-8:
            continue
        rotate_posebone_y_to_target(arm, bone_name, target_dir)
        if use_bone_scale:
            apply_bone_length_scale(
                arm,
                bone_name,
                target_dir.length,
                rest_lengths,
                bone_scale_min,
                bone_scale_max,
            )

    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")


def clear_old_cameras():
    for obj in list(bpy.context.scene.objects):
        if obj.name.startswith(PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)


def get_hand_center():
    pts = []

    for name in HAND_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue

        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))

    if not pts:
        raise RuntimeError("Cannot find Hand/Nails render bounds.")

    return sum(pts, Vector((0, 0, 0))) / len(pts)


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def get_or_create_camera(name):
    cam = bpy.data.objects.get(name)
    if cam is not None:
        if cam.type != "CAMERA":
            raise ValueError(f"Object exists but is not a camera: {name}")
        return cam

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = name
    return cam


def position_camera(cam, angle_deg, target, camera_radius, camera_height, ortho_scale):
    a = math.radians(angle_deg)

    x = target.x + camera_radius * math.sin(a)
    y = target.y - camera_radius * math.cos(a)
    z = target.z + camera_height

    cam.location = (x, y, z)
    cam.data.type = CAMERA_TYPE
    if CAMERA_TYPE == "ORTHO":
        cam.data.ortho_scale = ortho_scale
    else:
        cam.data.lens = CAMERA_LENS
    cam.data.clip_end = 1000

    look_at(cam, target)
    return cam


def setup_render(resolution):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.engine = RENDER_ENGINE
    scene.render.use_persistent_data = False

    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.use_denoising = True

    setup_world_background(scene)


def setup_world_background(scene):
    world = scene.world
    if world is None:
        print("[WARN] Scene has no World assigned. HDRI background cannot render.")
        return

    print("World:", world.name, "use_nodes=", world.use_nodes)

    if FORCE_WORLD_CAMERA_VISIBLE and hasattr(world, "cycles_visibility"):
        world.cycles_visibility.camera = True
        print("World camera visibility forced ON.")

    if world.use_nodes and world.node_tree:
        node_names = [node.name for node in world.node_tree.nodes]
        print("World nodes:", node_names)


def sample_out_paths(out_root, sample_id, flat_output):
    if flat_output:
        sample_dir = out_root
        return [sample_dir / f"{sample_id}_cam_{i:02d}.png" for i in range(len(CAMERA_ANGLES_DEG))]

    sample_dir = out_root / sample_id
    return [sample_dir / f"{sample_id}_cam_{i:02d}.png" for i in range(len(CAMERA_ANGLES_DEG))]


def render_sample(cameras, out_paths):
    scene = bpy.context.scene
    bpy.context.view_layer.update()

    for cam, out_path in zip(cameras, out_paths):
        if SKIP_EXISTING and out_path.exists():
            print("Skip rendered camera:", out_path)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.camera = cam
        scene.render.filepath = str(out_path)

        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)

        print("Rendered:", out_path)


def build_cameras():
    cameras = []
    for i, angle in enumerate(CAMERA_ANGLES_DEG):
        cam = get_or_create_camera(f"{PREFIX}cam_{i}_{angle}")
        cameras.append(cam)

    update_cameras(cameras)
    return cameras


def update_cameras(cameras):
    target = get_hand_center()
    print("Render target:", target)

    for cam, angle in zip(cameras, CAMERA_ANGLES_DEG):
        position_camera(
            cam,
            angle,
            target,
            CAMERA_RADIUS,
            CAMERA_HEIGHT,
            ORTHO_SCALE,
        )


def collect_samples(fitted_dir, requested_samples, start_index, limit):
    if requested_samples:
        sample_paths = []
        for sample in requested_samples:
            path = fitted_dir / f"{sample}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing requested sample: {path}")
            sample_paths.append(path)
    else:
        sample_paths = sorted(fitted_dir.glob("*.npy"))

    if start_index < 0:
        raise ValueError(f"START_INDEX must be >= 0, got {start_index}")

    sample_paths = sample_paths[start_index:]

    if limit is not None:
        sample_paths = sample_paths[:limit]

    return sample_paths


def normalize_pose_name(pose):
    pose = pose.strip().lower()
    pose = POSE_ALIASES.get(pose, pose)
    if pose not in SUPPORTED_POSES:
        raise ValueError(f"Unsupported pose: {pose}")
    return pose


def pose_fitted_dir(pose, total_poses):
    if FITTED_DIR:
        if total_poses != 1:
            raise ValueError("FITTED_DIR override is only supported when POSES has one pose.")
        return Path(FITTED_DIR)
    return PROJECT_ROOT / f"batch_{pose}" / "fitted"


def pose_out_root(pose, total_poses):
    if OUT_ROOT:
        root = Path(OUT_ROOT)
        if total_poses == 1:
            return root
        return root / pose
    return PROJECT_ROOT / f"renders_{pose}"


def render_pose(pose, total_poses, cameras):
    fitted_dir = pose_fitted_dir(pose, total_poses)
    out_root = pose_out_root(pose, total_poses)
    out_root.mkdir(parents=True, exist_ok=True)

    requested_samples = SAMPLES_BY_POSE.get(pose, SAMPLES)
    sample_paths = collect_samples(fitted_dir, requested_samples, START_INDEX, LIMIT)
    if not sample_paths:
        raise RuntimeError(f"No .npy samples found in {fitted_dir}")

    print("\n=========================================================")
    print("Pose:", pose)
    print("Input fitted dir:", fitted_dir)
    print("Output root:", out_root)
    print("Samples:", len(sample_paths))
    print("=========================================================")

    rendered = 0
    skipped = 0
    failed = 0

    for idx, fitted_npy in enumerate(sample_paths, start=1):
        sample_id = fitted_npy.stem
        out_paths = sample_out_paths(out_root, sample_id, FLAT_OUTPUT)

        if SKIP_EXISTING and all(path.exists() for path in out_paths):
            skipped += 1
            print(f"[{pose} {idx}/{len(sample_paths)}] Skip existing:", sample_id)
            continue

        print(f"[{pose} {idx}/{len(sample_paths)}] Retarget:", sample_id)
        try:
            apply_retarget(
                fitted_npy,
                use_bone_scale=USE_BONE_SCALE,
                bone_scale_min=BONE_SCALE_MIN,
                bone_scale_max=BONE_SCALE_MAX,
            )
            show_render_objects()
            update_cameras(cameras)
            render_sample(cameras, out_paths)
            rendered += 1
        except Exception as exc:
            failed += 1
            print("[FAILED]", fitted_npy, exc)

    print(f"POSE DONE: {pose} | rendered={rendered}, skipped={skipped}, failed={failed}")
    return {"pose": pose, "rendered": rendered, "skipped": skipped, "failed": failed}


def main():
    poses = [normalize_pose_name(pose) for pose in POSES]
    if not poses:
        raise RuntimeError("POSES is empty.")

    setup_render(RENDER_RES)
    show_render_objects()
    clear_old_cameras()
    cameras = build_cameras()

    summaries = []
    for pose in poses:
        summaries.append(render_pose(pose, len(poses), cameras))

    clear_old_cameras()
    print("\nDONE ALL POSES")
    for item in summaries:
        print(
            item["pose"],
            "rendered=", item["rendered"],
            "skipped=", item["skipped"],
            "failed=", item["failed"],
        )


main()
