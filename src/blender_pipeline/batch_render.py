import bpy
import math
import json
import argparse
from pathlib import Path

import numpy as np
from mathutils import Vector, Matrix


PROJECT_ROOT = Path(".")

ARMATURE_NAME = "Rig"
TARGET_OBJECT_NAMES = ["Hand", "Nails"]

SUPPORTED_POSES = ["ok", "paper", "rock", "scissors", "thefinger", "thumbup", "thumbdown"]

DEFAULT_VIEWS = [-45, 0, 30, 45, 60, 90, 120, 135]

BONE_MAP = [
    ((0, 1), "thumb_trapez"),
    ((1, 2), "thumb_meta"),
    ((2, 3), "thumb_prox"),
    ((3, 4), "thumb_dist"),

    ((0, 5), "index_meta"),
    ((5, 6), "index_prox"),
    ((6, 7), "index_midd"),
    ((7, 8), "index_dist"),

    ((0, 9), "midd_meta"),
    ((9, 10), "midd_prox"),
    ((10, 11), "midd_midd"),
    ((11, 12), "midd_dist"),

    ((0, 13), "ring_meta"),
    ((13, 14), "ring_prox"),
    ((14, 15), "ring_midd"),
    ((15, 16), "ring_dist"),

    ((0, 17), "pinky_meta"),
    ((17, 18), "pinky_prox"),
    ((18, 19), "pinky_midd"),
    ((19, 20), "pinky_dist"),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--poses", required=True, help="Example: ok,paper,rock,scissors,thefinger")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--views", default=",".join(map(str, DEFAULT_VIEWS)))
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "rendered_dataset"))

    parser.add_argument("--camera-radius", type=float, default=35.0)
    parser.add_argument("--camera-height", type=float, default=8.0)
    parser.add_argument("--ortho-scale", type=float, default=28.0)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)

    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"])

    return parser.parse_args()


def get_armature():
    arm = bpy.data.objects.get(ARMATURE_NAME)
    if arm is None:
        raise RuntimeError(f"Missing armature: {ARMATURE_NAME}")
    return arm


def clear_pose(arm):
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)

    for pb in arm.pose.bones:
        pb.matrix_basis.identity()

    bpy.context.view_layer.update()


def hide_debug_objects():
    prefixes = [
        "HIER_DBG",
        "DBG",
        "debug",
        "Skeleton",
        "skeleton",
    ]

    for obj in bpy.data.objects:
        if any(obj.name.startswith(p) for p in prefixes):
            obj.hide_render = True
            obj.hide_viewport = True


def configure_render(args):
    scene = bpy.context.scene

    scene.render.engine = args.engine

    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True

        prefs = bpy.context.preferences
        prefs.addons["cycles"].preferences.compute_device_type = "METAL"
        scene.cycles.device = "GPU"

    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"

    hide_debug_objects()


def normalize_pose_name(pose):
    pose = pose.strip().lower()
    aliases = {
        "the-finger": "thefinger",
        "the_finger": "thefinger",
        "middlefinger": "thefinger",
        "scrissors": "scissors",
    }
    pose = aliases.get(pose, pose)

    if pose not in SUPPORTED_POSES:
        raise ValueError(f"Unsupported pose: {pose}")

    return pose


def sample_id_from_path(path):
    stem = path.stem
    suffixes = [
        "_fitted_template_joints",
        "_fitted",
        "_template_joints",
    ]

    for s in suffixes:
        if stem.endswith(s):
            stem = stem[:-len(s)]

    return stem


def collect_fitted_files(pose, limit):
    fit_dir = PROJECT_ROOT / f"batch_{pose}" / "fitted"
    if not fit_dir.exists():
        raise FileNotFoundError(f"Missing fitted dir: {fit_dir}")

    files = sorted(fit_dir.glob("*.npy"))
    if limit is not None:
        files = files[:limit]

    return files


def unit(v, eps=1e-8):
    n = v.length
    if n < eps:
        return None
    return v / n


def retarget_skeleton_to_rig(joints, arm):
    clear_pose(arm)

    for (j0, j1), bone_name in BONE_MAP:
        if bone_name not in arm.pose.bones:
            print(f"[WARN] missing bone: {bone_name}")
            continue

        pb = arm.pose.bones[bone_name]

        target_vec_np = joints[j1] - joints[j0]
        target_vec = Vector((float(target_vec_np[0]), float(target_vec_np[1]), float(target_vec_np[2])))
        target_dir = unit(target_vec)

        current_dir = unit(pb.tail - pb.head)

        if target_dir is None or current_dir is None:
            continue

        try:
            q = current_dir.rotation_difference(target_dir)
        except Exception:
            continue

        head = pb.head.copy()
        M = pb.matrix.copy()

        R4 = q.to_matrix().to_4x4()
        T_head = Matrix.Translation(head)
        T_back = Matrix.Translation(-head)

        pb.matrix = T_head @ R4 @ T_back @ M
        bpy.context.view_layer.update()


def get_hand_center():
    points = []

    for name in TARGET_OBJECT_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue

        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))

    if not points:
        return Vector((0, 0, 0))

    center = sum(points, Vector((0, 0, 0))) / len(points)
    return center


def get_or_create_camera(name):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj

    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    return cam_obj


def look_at(obj, target):
    direction = target - obj.location
    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()


def setup_camera(view_angle, args):
    center = get_hand_center()

    rad = math.radians(view_angle)
    loc = center + Vector((
        args.camera_radius * math.sin(rad),
        -args.camera_radius * math.cos(rad),
        args.camera_height,
    ))

    cam = get_or_create_camera(f"RENDER_cam_{view_angle}")
    cam.location = loc
    look_at(cam, center)

    cam.data.type = "ORTHO"
    cam.data.ortho_scale = args.ortho_scale

    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()

    return cam


def render_one(joints_path, pose, views, args, out_root):
    arm = get_armature()

    joints = np.load(joints_path).astype(np.float32)
    if joints.shape != (21, 3):
        raise ValueError(f"Invalid joints shape {joints.shape}: {joints_path}")

    retarget_skeleton_to_rig(joints, arm)

    sample_id = sample_id_from_path(joints_path)
    sample_out = out_root / pose / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    rows = []

    for view in views:
        setup_camera(view, args)

        out_path = sample_out / f"view_{view}.png"
        bpy.context.scene.render.filepath = str(out_path)

        bpy.ops.render.render(write_still=True)

        rows.append({
            "pose": pose,
            "sample_id": sample_id,
            "view": view,
            "image_path": str(out_path),
            "skeleton_path": str(joints_path),
        })

    return rows


def main():
    args = parse_args()
    configure_render(args)

    poses = [normalize_pose_name(p) for p in args.poses.split(",")]
    views = [int(float(v.strip())) for v in args.views.split(",") if v.strip()]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    print("POSES:", poses)
    print("VIEWS:", views)
    print("OUT:", out_root)

    for pose in poses:
        files = collect_fitted_files(pose, args.limit)

        print(f"\n=== RENDER POSE: {pose} ===")
        print("Files:", len(files))

        for idx, path in enumerate(files, start=1):
            print(f"[{pose}] {idx}/{len(files)} {path.name}")

            try:
                rows = render_one(path, pose, views, args, out_root)
                all_metadata.extend(rows)
            except Exception as e:
                print("[FAILED]", path, e)

    metadata_path = out_root / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\nDONE")
    print("Rendered images:", len(all_metadata))
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()