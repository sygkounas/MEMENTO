import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
from matplotlib import text
import numpy as np
import open3d as o3d
import torch
from huggingface_hub import login
from PIL import Image
from scipy.spatial.transform import Rotation
from transformers import Sam3Model, Sam3Processor

from affordance_perception.utils import overlay_masks


DEFAULT_TEXTS = [
    # NO leftmost/middle/rightmost — handled separately
    "green cube",
    "blue cube",
    "purple cube",
    "brown cube",
     "leftmost red cross on table",
"middle red cross on table",
"rightmost red cross on table",
]

# CROSS_LABELS = [
#     "leftmost red cross on table",
# "middle red cross on table",
# "rightmost red cross on table",
# ]

# CROSS_QUERY =  "red square ring on tabletop"
class ObjectGrounding:
    def __init__(
        self,
        panda_T_camera: np.ndarray,
        hf_token: Optional[str] = None,
        output_dir: str = ".",
        save_overlay: bool = True,
        save_pointclouds: bool = False,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set.")

        login(token=token)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.panda_T_camera = np.asarray(panda_T_camera, dtype=np.float64)
        # print("panda_T_camera used by ObjectGrounding:")
        # print(self.panda_T_camera)
        # self.panda_T_camera[0, 3] += 0.011  # ← small manual adjustment to better center on objects
        # self.panda_T_camera[1, 3] -= 0.024
        
        print("panda_T_camera used by ObjectGrounding:")
        print(self.panda_T_camera)
        

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_overlay = save_overlay
        self.save_pointclouds = save_pointclouds

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.model.eval()
    def _get_tabletop_crop(self, color_bgr: np.ndarray, pad: int = 20):
        """
        Find the white tabletop/paper area and return a crop.
        The crop still contains all objects on the table, but removes most background.
        """
        h, w = color_bgr.shape[:2]

        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

        # White paper/table: low saturation, high value.
        # Tune if needed.
        lower = np.array([0, 0, 120], dtype=np.uint8)
        upper = np.array([180, 80, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower, upper)

        # Ignore extreme image borders to avoid robot/background dominating.
        roi_mask = np.zeros_like(white_mask)
        roi_mask[int(0.08 * h): int(0.90 * h), int(0.05 * w): int(0.95 * w)] = 255
        white_mask = cv2.bitwise_and(white_mask, roi_mask)

        kernel = np.ones((9, 9), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(
            white_mask,
            connectivity=8,
        )

        if num_labels <= 1:
            # fallback: fixed central tabletop crop
            x1, y1, x2, y2 = 170, 70, min(w, 820), min(h, 470)
        else:
            # largest white component
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

            x = int(stats[largest, cv2.CC_STAT_LEFT])
            y = int(stats[largest, cv2.CC_STAT_TOP])
            bw = int(stats[largest, cv2.CC_STAT_WIDTH])
            bh = int(stats[largest, cv2.CC_STAT_HEIGHT])

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

        crop_bgr = color_bgr[y1:y2, x1:x2].copy()

        # Save debug image showing what is passed to SAM3
        cv2.imwrite(str(self.output_dir / "sam3_tabletop_crop.png"), crop_bgr)

        return crop_bgr, (x1, y1, x2, y2)


    def _crop_mask_to_full_mask(self, crop_mask, full_shape, roi):
        """
        Convert a SAM3 crop mask back into full image coordinates.
        """
        x1, y1, x2, y2 = roi
        full_h, full_w = full_shape[:2]

        if hasattr(crop_mask, "cpu"):
            crop_mask = crop_mask.cpu().numpy()
        else:
            crop_mask = np.asarray(crop_mask)

        crop_mask = crop_mask.astype(bool)

        target_h = y2 - y1
        target_w = x2 - x1

        if crop_mask.shape != (target_h, target_w):
            crop_mask = cv2.resize(
                crop_mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        full_mask = np.zeros((full_h, full_w), dtype=bool)
        full_mask[y1:y2, x1:x2] = crop_mask

        return full_mask

    def detect(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        texts: Optional[Sequence[str]] = None,
    ) -> Dict:
        if texts is None or len(texts) == 0:
            texts = DEFAULT_TEXTS

        if color_bgr is None or depth_m is None:
            return self._empty_result("Missing color or depth image")

        if color_bgr.shape[:2] != depth_m.shape[:2]:
            return self._empty_result(
                f"Color/depth shape mismatch: color={color_bgr.shape}, depth={depth_m.shape}"
            )

        # color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(color_rgb).convert("RGB")
        
        crop_bgr, roi = self._get_tabletop_crop(color_bgr)

        color_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_rgb).convert("RGB")

        full_image = Image.fromarray(
            cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        ).convert("RGB")

        all_masks: List[torch.Tensor] = []
        all_labels: List[str] = []
        all_scores: List[float] = []

        # for text in texts:
        #     masks, scores = self._run_sam3(image, text)
        for text in texts:
           

            masks, scores = self._run_sam3(image, text)

            if len(masks) == 0:
                continue

            output_label = text   # ← ADD THIS

            if output_label == "wooden cube":
                best_idx = self._select_non_table_mask(
                    masks=masks,
                    scores=scores,
                    image_shape=color_bgr.shape,
                )
                if best_idx is None:
                    continue
            else:
                best_idx = int(np.argmax([float(s) for s in scores])) if len(scores) else 0

            # all_masks.append(masks[best_idx])
            full_mask = self._crop_mask_to_full_mask(
                masks[best_idx],
                color_bgr.shape,
                roi,
            )

            all_masks.append(full_mask)
            all_scores.append(float(scores[best_idx]) if len(scores) else 1.0)
            all_labels.append(output_label)

        # all_masks, all_labels, all_scores = self._detect_crosses_sorted(
        #     image=image,
        #     all_masks=all_masks,
        #     all_labels=all_labels,
        #     all_scores=all_scores,
        # )
        # all_masks, all_labels, all_scores = self._detect_crosses_sorted(
        #     image=image,
        #     all_masks=all_masks,
        #     all_labels=all_labels,
        #     all_scores=all_scores,
        #     full_shape=color_bgr.shape,
        #     roi=roi,
        # )
        if self.save_overlay:
            overlay_path = self._save_overlay(full_image, all_masks, all_scores, all_labels)
        else:
            overlay_path = None

        # table_mask_path = None
        # if "brown tabletop" in all_labels:
        #     table_mask = all_masks[all_labels.index("brown tabletop")].cpu().numpy().astype(bool)
        #     table_mask_path = self.output_dir / "table_mask.npy"
        #     np.save(table_mask_path, table_mask)
        # in detect(), after all detections but before _extract_poses:
        # all_masks, all_labels, all_scores = self._dedupe_by_xy(
        #     all_masks, all_labels, all_scores
        # )
        poses = self._extract_poses(
            masks=all_masks,
            labels=all_labels,
            depth_m=depth_m,
            color_bgr=color_bgr,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        return {
            "success": True,
            "message": f"Detected {len(poses)} objects",
            "labels": [p["label"] for p in poses],
            "scores": all_scores,
            "masks": all_masks,
            "all_labels": all_labels,
            "poses": poses,
            # "table_mask_path": str(table_mask_path) if table_mask_path else None,
            "overlay_path": str(overlay_path) if overlay_path else None,
        }

    def _run_sam3(self, image: Image.Image, text: str):
        inputs = self.processor(
            images=image,
            text=[text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.05,
            mask_threshold=0.2,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        return results["masks"], results.get("scores", [])

    def _save_overlay(self, image: Image.Image, masks, scores, labels) -> Path:
        combined = {
            "masks": masks,
            "scores": scores,
            "labels": labels,
        }

        overlay = overlay_masks(image, combined)
        output_path = self.output_dir / "sam3_result.png"
        overlay.save(output_path)

        return output_path

   

    @staticmethod
    def _mask_center_x(mask) -> Optional[float]:
        if hasattr(mask, "cpu"):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask).astype(bool)

        _, xs = np.where(mask_np)

        if len(xs) == 0:
            return None

        return float(xs.mean())

    @staticmethod
    def _select_non_table_mask(masks, scores, image_shape, max_area_ratio: float = 0.20):
        h, w = image_shape[:2]
        image_area = float(h * w)

        candidates = []

        for i, mask in enumerate(masks):
            mask_np = mask.cpu().numpy().astype(bool)
            area_ratio = float(mask_np.sum()) / image_area

            # Brown cube should not occupy a large part of the image.
            # Large masks are more likely to be tabletop/background.
            if area_ratio > max_area_ratio:
                continue

            score = float(scores[i]) if len(scores) else 1.0
            candidates.append((score, i))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]
    # def mean_object_size(self, pc: np.ndarray) -> float:
    #     print("[DEBUG] using mean_object_size / center pose estimator")

    #     low = np.percentile(pc, 5, axis=0)
    #     high = np.percentile(pc, 95, axis=0)
    #     center_panda = 0.5 * (low + high)

    #     print("[DEBUG] low=", low, "high=", high, "center=", center_panda)

    #     mean_panda = center_panda
    #     median_panda = center_panda
    #     rot_panda = np.eye(3)

    #     return mean_panda, median_panda, rot_panda
    def _extract_poses(
        self,
        masks,
        labels,
        depth_m: np.ndarray,
        color_bgr: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> List[Dict]:
        poses = []

        R = self.panda_T_camera[:3, :3]
        t = self.panda_T_camera[:3, 3]

        for mask, label in zip(masks, labels):
            if label == "brown tabletop":
                continue

            # mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            if hasattr(mask, "cpu"):
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            else:
                mask_np = np.asarray(mask).astype(np.uint8) * 255
            eroded = cv2.erode(mask_np, np.ones((3, 3), np.uint8), iterations=2)
            mask_bool = eroded == 255

            pc_cam, pc_color = self._mask_to_pointcloud(
                mask_bool=mask_bool,
                depth_m=depth_m,
                color_bgr=color_bgr,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )

            if len(pc_cam) == 0:
                continue

            ind = self.object_prefilter(pc_cam)

            if len(ind):
                pc_cam = pc_cam[ind]
                pc_color = pc_color[ind]

            if len(pc_cam) == 0:
                continue

            pc_panda = (R @ pc_cam.T).T + t
            mean_panda, median_panda, rot_panda = self.get_grasp_pose(pc_panda.copy())
            quat_panda = Rotation.from_matrix(rot_panda).as_quat()

            if self.save_pointclouds:
                self._save_object_pcd(label, pc_panda, pc_color)

            poses.append(
                {
                    "label": label,
                    "mean_panda": mean_panda,
                    "median_panda": median_panda,
                    "quat_panda": quat_panda,
                    "pc_cam": pc_cam,
                    "pc_panda": pc_panda,
                    "pc_color": pc_color,
                }
            )

        return poses

    @staticmethod
    def _mask_to_pointcloud(
        mask_bool: np.ndarray,
        depth_m: np.ndarray,
        color_bgr: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        indexes = np.argwhere(mask_bool)

        if len(indexes) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        z = depth_m[indexes[:, 0], indexes[:, 1]]
        x = (indexes[:, 1].astype(float) - cx) * z / fx
        y = (indexes[:, 0].astype(float) - cy) * z / fy

        valid = z > 0

        pc = np.hstack((x[:, None], y[:, None], z[:, None]))[valid]
        pc_color = color_bgr[indexes[:, 0], indexes[:, 1], ::-1][valid]

        return pc, pc_color

    def _save_object_pcd(self, label: str, pc_panda: np.ndarray, pc_color: np.ndarray) -> None:
        safe_label = label.replace(" ", "_").replace("/", "_")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_panda))
        pcd.colors = o3d.utility.Vector3dVector(pc_color / 255.0)

        o3d.io.write_point_cloud(str(self.output_dir / f"{safe_label}_panda.ply"), pcd)

    @staticmethod
    def object_prefilter(pc: np.ndarray) -> np.ndarray:
        if len(pc) < 8:
            return np.arange(len(pc))

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)

        return np.asarray(ind)

    @staticmethod
    def get_surface_normal_direction(pc: np.ndarray) -> np.ndarray:
        mean = pc[:, :2].mean(axis=0)
        minidx = np.linalg.norm(pc[:, :2] - mean, axis=1).argmin()
        direction = pc[minidx, :2] - mean

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.array([1.0, 0.0])

        return direction / norm

    # @staticmethod
    # def get_grasp_pose(pc: np.ndarray):
    #     dim = pc.max(axis=0) - pc.min(axis=0)
    #     delta = dim.min()

    #     median = np.median(pc, axis=0)
    #     mean = np.mean(pc, axis=0)

    #     diff = pc[:, :2] - mean[:2]
    #     cov = np.cov(diff.T)

    #     eigvals, eigvecs = np.linalg.eig(cov)
    #     sort_idx = np.argsort(eigvals)

    #     direction = ObjectGrounding.get_surface_normal_direction(pc)
    #     cos_t = np.dot(direction, eigvecs[:, sort_idx[0]])

    #     if cos_t < 0:
    #         eigvecs[:, sort_idx[0]] *= -1

    #     rot = np.eye(3)
    #     rot[:2, :2] = eigvecs[:, [sort_idx[1], sort_idx[0]]]
    #     rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])

    #     if rot[2, 2] > 0:
    #         rot[:, 2] *= -1
    #         rot[:, 1] *= -1

    #     mean_rot = rot.T @ mean + np.array([0.0, delta * 0.5, 0.0])
    #     median_rot = rot.T @ median + np.array([0.0, delta * 0.5, 0.0])

    #     return rot @ mean_rot, rot @ median_rot, rot
    
    @staticmethod
    def get_grasp_pose(pc: np.ndarray):
        dim = pc.max(axis=0) - pc.min(axis=0)
        median = np.median(pc, axis=0)
        mean = np.mean(pc, axis=0)

        diff = pc[:, :2] - mean[:2]
        cov = np.cov(diff.T)

        eigvals, eigvecs = np.linalg.eig(cov)
        sort_idx = np.argsort(eigvals)

        direction = ObjectGrounding.get_surface_normal_direction(pc)
        cos_t = np.dot(direction, eigvecs[:, sort_idx[0]])

        if cos_t < 0:
            eigvecs[:, sort_idx[0]] *= -1

        rot = np.eye(3)
        rot[:2, :2] = eigvecs[:, [sort_idx[1], sort_idx[0]]]
        rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])

        if rot[2, 2] > 0:
            rot[:, 2] *= -1
            rot[:, 1] *= -1

        # For top-down cube picking, do not shift by delta * 0.5.
        return mean, median, rot

    @staticmethod
    def _empty_result(message: str) -> Dict:
        return {
            "success": False,
            "message": message,
            "labels": [],
            "scores": [],
            "masks": [],
            "all_labels": [],
            "poses": [],
            # "table_mask_path": None,
            "overlay_path": None,
        }
        
   
