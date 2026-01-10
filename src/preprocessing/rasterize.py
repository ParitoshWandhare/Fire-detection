"""
Rasterization utilities for creating fire masks from point data or manual annotations.
Supports both automatic fire detection and manual annotation workflows.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

from data_io.geo import (
    GeoTransformer,
    FirePointMapper,
    detect_fire_pixels_auto,
    FireDetectionConfig,
    load_fire_detection_config,
)

logger = logging.getLogger(__name__)


@dataclass
class FireAnnotation:
    """Container for manual fire annotations."""
    image_filename: str
    state: str
    date: datetime
    fire_points: List[Tuple[float, float]]  # (lat, lon) coordinates
    fire_polygons: List[List[Tuple[float, float]]]  # List of polygon vertices
    annotation_method: str  # 'manual', 'auto', 'hybrid'
    annotator: str
    created_at: datetime
    confidence: float = 1.0  # 0-1 confidence score


class FireRasterizer:
    """Convert fire annotations to binary masks."""

    def __init__(self, geo_transformer: GeoTransformer):
        self.geo_transformer = geo_transformer
        self.fire_mapper = FirePointMapper(geo_transformer)

    def points_to_mask(self, fire_points: List[Tuple[float, float]], radius: int = 3) -> np.ndarray:
        """Convert fire points to binary mask."""
        return self.fire_mapper.create_binary_mask(fire_points, radius=radius)

    def polygons_to_mask(self, fire_polygons: List[List[Tuple[float, float]]]) -> np.ndarray:
        """Convert fire polygons to binary mask."""
        mask = np.zeros((self.geo_transformer.height, self.geo_transformer.width), dtype=np.uint8)

        for polygon in fire_polygons:
            if len(polygon) < 3:
                logger.warning("Polygon has fewer than 3 vertices, skipping")
                continue

            pixel_polygon = []
            for lat, lon in polygon:
                if self.geo_transformer.is_point_in_bounds(lat, lon):
                    row, col = self.geo_transformer.latlon_to_pixel(lat, lon)
                    pixel_polygon.append([col, row])  # OpenCV expects (x, y)

            if len(pixel_polygon) >= 3:
                pixel_polygon = np.array(pixel_polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pixel_polygon], 1)

        return mask

    def annotation_to_mask(self, annotation: FireAnnotation) -> np.ndarray:
        """Convert complete annotation to binary mask."""
        mask = np.zeros((self.geo_transformer.height, self.geo_transformer.width), dtype=np.uint8)

        if annotation.fire_points:
            point_mask = self.points_to_mask(annotation.fire_points)
            mask = np.logical_or(mask, point_mask).astype(np.uint8)

        if annotation.fire_polygons:
            polygon_mask = self.polygons_to_mask(annotation.fire_polygons)
            mask = np.logical_or(mask, polygon_mask).astype(np.uint8)

        return mask


class AutoFireDetector:
    """Automatic fire detection from satellite imagery using configuration."""

    def __init__(self, config: Optional[FireDetectionConfig] = None, data_config_path: Optional[str] = None):
        if config is not None:
            self.config = config
        else:
            self.config = load_fire_detection_config(data_config_path)

        logger.info("AutoFireDetector initialized with parameters:")
        logger.info(f"  Red threshold: {self.config.red_threshold}")
        logger.info(f"  Orange ratio: {self.config.orange_ratio}")
        logger.info(f"  Brightness threshold: {self.config.brightness_threshold}")
        logger.info(f"  Green max: {self.config.green_max}")
        logger.info(f"  Blue max: {self.config.blue_max}")
        logger.info(f"  Min fire area: {self.config.min_fire_area}")

    def detect_fires(self, image: np.ndarray) -> List[Tuple[int, int]]:
        return detect_fire_pixels_auto(image, self.config)

    def extract_fire_polygons(
        self,
        fire_pixels: List[Tuple[int, int]],
        geo_transformer: GeoTransformer,
        min_area: int = None,
        simplify_epsilon: float = 2.0,
    ) -> List[List[Tuple[float, float]]]:

        if not fire_pixels:
            return []

        if min_area is None:
            min_area = self.config.min_fire_area

        mask = np.zeros((geo_transformer.height, geo_transformer.width), dtype=np.uint8)

        for row, col in fire_pixels:
            if 0 <= row < geo_transformer.height and 0 <= col < geo_transformer.width:
                mask[row, col] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_polygons = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            approx = cv2.approxPolyDP(contour, simplify_epsilon, True)

            polygon_latlon = []
            for point in approx:
                col, row = point[0]
                try:
                    lat, lon = geo_transformer.pixel_to_latlon(int(row), int(col))
                    polygon_latlon.append((float(lat), float(lon)))
                except Exception as e:
                    logger.warning(f"Could not convert pixel ({row}, {col}) to lat/lon: {e}")

            if len(polygon_latlon) >= 3:
                fire_polygons.append(polygon_latlon)

        logger.info(f"Extracted {len(fire_polygons)} fire polygons from {len(fire_pixels)} pixels")
        return fire_polygons

    def cluster_fire_pixels(self, fire_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        if not fire_pixels:
            return []

        try:
            from sklearn.cluster import DBSCAN

            pixels = np.array(fire_pixels)

            clustering = DBSCAN(
                eps=10,
                min_samples=self.config.min_fire_area,
            ).fit(pixels)

            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label != -1:
                    clusters.setdefault(label, []).append(tuple(pixels[i]))

            return list(clusters.values())

        except ImportError:
            logger.warning("sklearn not available, using simple distance-based clustering")
            return self._simple_clustering(fire_pixels)

    def _simple_clustering(self, fire_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        if not fire_pixels:
            return []

        clusters = []
        remaining_pixels = fire_pixels.copy()
        max_distance = 10

        while remaining_pixels:
            current_cluster = [remaining_pixels.pop(0)]
            i = 0

            while i < len(remaining_pixels):
                pixel = remaining_pixels[i]
                min_dist = float("inf")

                for cluster_pixel in current_cluster:
                    dist = np.sqrt((pixel[0] - cluster_pixel[0]) ** 2 + (pixel[1] - cluster_pixel[1]) ** 2)
                    min_dist = min(min_dist, dist)

                if min_dist <= max_distance:
                    current_cluster.append(remaining_pixels.pop(i))
                else:
                    i += 1

            if len(current_cluster) >= self.config.min_fire_area:
                clusters.append(current_cluster)

        return clusters

    def pixels_to_annotation(
        self,
        image_filename: str,
        state: str,
        date: datetime,
        fire_pixels: List[Tuple[int, int]],
        geo_transformer: GeoTransformer,
    ) -> FireAnnotation:

        fire_points = []

        for row, col in fire_pixels:
            try:
                lat, lon = geo_transformer.pixel_to_latlon(row, col)
                fire_points.append((float(lat), float(lon)))
            except Exception as e:
                logger.warning(f"Could not convert pixel ({row}, {col}) to lat/lon: {e}")

        fire_polygons = self.extract_fire_polygons(
            fire_pixels,
            geo_transformer,
            min_area=self.config.min_fire_area,
            simplify_epsilon=2.0,
        )

        logger.info(f"Created annotation with {len(fire_points)} points and {len(fire_polygons)} polygons")

        return FireAnnotation(
            image_filename=image_filename,
            state=state,
            date=date,
            fire_points=fire_points,
            fire_polygons=fire_polygons,
            annotation_method="auto_optimized",
            annotator=f"auto_detector_r{self.config.red_threshold}_or{self.config.orange_ratio}",
            created_at=datetime.now(),
            confidence=0.85,
        )


class AnnotationManager:
    """Manage fire annotations with persistence."""

    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self._annotations = {}
        self._load_all_annotations()

    def _get_annotation_path(self, filename: str) -> Path:
        base_name = Path(filename).stem
        return self.annotations_dir / f"{base_name}_annotation.json"

    def _load_all_annotations(self):
        for annotation_file in self.annotations_dir.glob("*_annotation.json"):
            try:
                self._load_annotation(annotation_file)
            except Exception as e:
                logger.warning(f"Could not load annotation {annotation_file}: {e}")

    def _load_annotation(self, annotation_path: Path):
        with open(annotation_path, "r") as f:
            data = json.load(f)

        try:
            date = datetime.fromisoformat(data["date"])
        except Exception:
            date = datetime.now()

        try:
            created_at = datetime.fromisoformat(data["created_at"])
        except Exception:
            created_at = datetime.now()

        annotation = FireAnnotation(
            image_filename=data["image_filename"],
            state=data["state"],
            date=date,
            fire_points=data.get("fire_points", []),
            fire_polygons=data.get("fire_polygons", []),
            annotation_method=data.get("annotation_method", "manual"),
            annotator=data.get("annotator", "unknown"),
            created_at=created_at,
            confidence=data.get("confidence", 1.0),
        )

        self._annotations[data["image_filename"]] = annotation

    def save_annotation(self, annotation: FireAnnotation):
        annotation_path = self._get_annotation_path(annotation.image_filename)

        data = {
            "image_filename": annotation.image_filename,
            "state": annotation.state,
            "date": annotation.date.isoformat(),
            "fire_points": annotation.fire_points,
            "fire_polygons": annotation.fire_polygons,
            "annotation_method": annotation.annotation_method,
            "annotator": annotation.annotator,
            "created_at": annotation.created_at.isoformat(),
            "confidence": annotation.confidence,
        }

        with open(annotation_path, "w") as f:
            json.dump(data, f, indent=2)

        self._annotations[annotation.image_filename] = annotation
        logger.info(
            f"Saved annotation for {annotation.image_filename} with {len(annotation.fire_polygons)} polygons"
        )

    def get_annotation(self, filename: str) -> Optional[FireAnnotation]:
        return self._annotations.get(filename)

    def has_annotation(self, filename: str) -> bool:
        return filename in self._annotations

    def list_annotated_files(self) -> List[str]:
        return list(self._annotations.keys())

    def get_annotations_by_method(self, method: str) -> List[FireAnnotation]:
        return [ann for ann in self._annotations.values() if ann.annotation_method == method]

    def get_statistics(self) -> Dict:
        total_annotations = len(self._annotations)
        methods = {}
        states = {}
        total_fire_points = 0
        total_fire_polygons = 0

        for ann in self._annotations.values():
            methods[ann.annotation_method] = methods.get(ann.annotation_method, 0) + 1
            states[ann.state] = states.get(ann.state, 0) + 1
            total_fire_points += len(ann.fire_points)
            total_fire_polygons += len(ann.fire_polygons)

        return {
            "total_annotations": total_annotations,
            "methods": methods,
            "states": states,
            "total_fire_points": total_fire_points,
            "total_fire_polygons": total_fire_polygons,
            "avg_fire_points_per_image": total_fire_points / max(total_annotations, 1),
            "avg_fire_polygons_per_image": total_fire_polygons / max(total_annotations, 1),
        }


class MaskGenerator:
    """Generate training masks from annotations."""

    def __init__(self, annotation_manager: AnnotationManager):
        self.annotation_manager = annotation_manager

    def generate_mask_for_image(self, filename: str, geo_transformer: GeoTransformer) -> Optional[np.ndarray]:
        annotation = self.annotation_manager.get_annotation(filename)
        if not annotation:
            logger.warning(f"No annotation found for {filename}")
            return None

        rasterizer = FireRasterizer(geo_transformer)
        return rasterizer.annotation_to_mask(annotation)

    def generate_all_masks(self, geo_transformer_factory, output_dir: str) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_mapping = {}

        for filename in self.annotation_manager.list_annotated_files():
            try:
                geo_transformer = geo_transformer_factory(filename)
                mask = self.generate_mask_for_image(filename, geo_transformer)

                if mask is not None:
                    mask_filename = Path(filename).stem + "_mask.png"
                    mask_path = output_dir / mask_filename
                    cv2.imwrite(str(mask_path), mask * 255)
                    mask_mapping[filename] = str(mask_path)
                    logger.info(f"Generated mask: {mask_path}")

            except Exception as e:
                logger.error(f"Error generating mask for {filename}: {e}")

        return mask_mapping


def create_manual_annotation_template(filename: str, state: str, date: datetime) -> FireAnnotation:
    return FireAnnotation(
        image_filename=filename,
        state=state,
        date=date,
        fire_points=[],
        fire_polygons=[],
        annotation_method="manual",
        annotator="human",
        created_at=datetime.now(),
        confidence=1.0,
    )
