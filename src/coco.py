"""COCO dataset class names and helpers."""

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# helper to map names to ids
NAME_TO_ID = {name: i for i, name in enumerate(COCO_CLASSES)}


def find_closest_class(query: str, method: str = "fuzzy", topn: int = 3):
    """Return (name, id, score) best matches for a query using fuzzy or semantic method."""
    query = query.lower()
    if method == "semantic":
        try:
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer("all-MiniLM-L6-v2")
            corpus = COCO_CLASSES
            corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
            q_emb = model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(q_emb, corpus_embeddings)[0]
            top_idx = scores.topk(k=topn)
            results = []
            for idx, score in zip(top_idx[1].tolist(), top_idx[0].tolist()):
                name = COCO_CLASSES[idx]
                results.append((name, idx, float(score)))
            return results
        except Exception:
            pass  # move onto fuzzy
    # fuzzy match using difflib
    import difflib

    matches = difflib.get_close_matches(
        query, COCO_CLASSES, n=topn, cutoff=0.0
    )
    out = []
    for m in matches:
        out.append((m, NAME_TO_ID[m], 1.0))
    return out
