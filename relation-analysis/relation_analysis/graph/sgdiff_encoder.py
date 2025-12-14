import torch
from pathlib import Path
from typing import Dict, List, Tuple

# Add SGDiff to path (repo root → SGDiff)
SGDIFF_ROOT = Path(__file__).resolve().parents[3] / "SGDiff"
import sys
if str(SGDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(SGDIFF_ROOT))

from ldm.modules.cgip.cgip import CGIPModel
from ldm.modules.cgip.tools import encode_scene_graphs


class SGDiffGraphEncoder:
    """
    Thin wrapper around SGDiff's CGIPModel to produce local/global graph embeddings
    for single-triple graphs.
    """

    def __init__(
        self,
        vocab_path: Path,
        ckpt_path: Path,
        embed_dim: int = 512,
        gnn_width: int = 512,
        gnn_layers: int = 5,
        device: str = "cuda",
    ):
        vocab_path = Path(vocab_path)
        ckpt_path = Path(ckpt_path)
        self.vocab = self._load_vocab(vocab_path)
        self.device = torch.device(device)

        self.model = CGIPModel(
            num_objs=len(self.vocab["object_idx_to_name"]),
            num_preds=len(self.vocab["pred_idx_to_name"]),
            width=gnn_width,
            layers=gnn_layers,
            embed_dim=embed_dim,
            ckpt_path=str(ckpt_path),
            ignore_keys=[],
            max_sample_per_img=1,  # single triple → one local token
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_vocab(self, path: Path) -> Dict:
        import json

        with path.open("r") as f:
            return json.load(f)

    def encode_batch(self, triples: List[Tuple[str, str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of (subject, predicate, object) strings.

        Returns:
            local: (B, 1, 3*embed_dim)
            global: (B, embed_dim)
        """
        scene_graphs = []
        for s, p, o in triples:
            scene_graphs.append({"objects": [s, o], "relationships": [(0, p, 1)]})

        # Track how many relationships each scene graph will yield after SGDiff augmentation
        # (original relationships + one __in_image__ per object).
        rel_counts = [len(sg["relationships"]) + len(sg["objects"]) for sg in scene_graphs]

        objs, rels, obj_to_img, _ = encode_scene_graphs(self.vocab, scene_graphs)
        objs = objs.to(self.device)
        rels = rels.to(self.device)
        obj_to_img = obj_to_img.to(self.device)
        # Rebuild triple_to_img so predicate pooling matches per-image object pooling.
        triple_to_img = torch.cat(
            [torch.full((c,), idx, device=self.device, dtype=torch.int64) for idx, c in enumerate(rel_counts)],
            dim=0,
        )

        # dummy image/boxes (unused in CGIP graph conv)
        bsz = len(triples)
        dummy_imgs = torch.zeros(bsz, 3, 1, 1, device=self.device)
        dummy_boxes = torch.zeros_like(objs, dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, 4)

        graph = (dummy_imgs, objs, dummy_boxes, rels, obj_to_img, triple_to_img)
        local, global_ = self.model.encode_graph_local_global(graph)
        return local, global_
