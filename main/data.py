from dataclasses import dataclass, field
import numpy as np



@dataclass
class Colony:
    roi: tuple[int, int, int, int]
    roi_global: tuple[int, int, int, int]
    score: float
    label: int = 0


@dataclass
class Tile:
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    image: np.ndarray | None = None
    colonies: list[Colony] = field(default_factory=list)


@dataclass
class Plate:
    sample_id: str
    tiles: dict[tuple[int, int], Tile] = field(default_factory=dict)
    image_path: str | None = None
    image: np.ndarray | None = None          # full-res BGR, from cv2.imread
    crop_bbox: tuple[int, int, int, int] | None = None   # (x1, y1, x2, y2)
    cropped: np.ndarray | None = None        # working image for tiling

    @property
    def colonies(self):
        """Every colony on the plate, flattened across tiles."""
        return [c for t in self.tiles.values() for c in t.colonies]
    
    @property
    def count(self):
        """Number of colonies surviving all filtering."""
        return len(self.colonies)