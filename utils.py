import re
import cv2
import numpy as np
from pathlib import Path


def euclidean_distance(p, q):
    """Distância euclidiana entre dois pontos (x, y)."""
    return float(np.hypot(q[0] - p[0], q[1] - p[1]))

def parse_annotation(path):
    """Lê o .txt de anotação e devolve (placa, cantos).

    Os cantos seguem a ordem: superior esquerdo, superior direito,
    inferior direito e inferior esquerdo.
    """
    info = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

    # Tolera vírgulas sobrando, como em "x,y x,y, x,y, x,y".
    nums = [int(n) for n in re.findall(r"-?\d+", info["corners"])]
    if len(nums) != 8:
        raise ValueError(f"{path}: esperados 4 cantos, lidos {len(nums) // 2}")
    corners = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
    return info.get("plate", ""), corners


def read_split(split_file):
    """Devolve a lista de pares (caminho_da_imagem, partição)."""
    samples = []
    for line in Path(split_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_path, part = line.rsplit(";", 1)
        samples.append((image_path.strip(), part.strip()))
    return samples


def output_size(corners):
    """Dimensões do retângulo de destino a partir dos lados do quadrilátero."""
    tl, tr, br, bl = corners
    max_w = int(round(max(euclidean_distance(bl, br), euclidean_distance(tl, tr))))
    max_h = int(round(max(euclidean_distance(tr, br), euclidean_distance(tl, bl))))
    return max(max_w, 1), max(max_h, 1)


def transform_image(img, corners):
    """Aplica a transformação perspectiva e devolve a placa retificada."""
    max_w, max_h = output_size(corners)
    src = np.array(corners, dtype="float32")
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, homography, (max_w, max_h), flags=cv2.INTER_LINEAR)


def make_panel(img, corners, rectified, margin=40, height=240):
    """Monta uma figura: região da placa com o quadrilátero e o resultado."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = max(min(xs) - margin, 0), min(max(xs) + margin, img.shape[1])
    y0, y1 = max(min(ys) - margin, 0), min(max(ys) + margin, img.shape[0])
    region = img[y0:y1, x0:x1].copy()
    pts = np.array([(x - x0, y - y0) for x, y in corners], dtype=np.int32)
    cv2.polylines(region, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)

    def fit(im):
        scale = height / im.shape[0]
        return cv2.resize(im, (max(int(im.shape[1] * scale), 1), height))

    gap = np.full((height, 8, 3), 255, dtype=np.uint8)
    return np.hstack([fit(region), gap, fit(rectified)])


def resolve(root, rel):
    """Localiza um arquivo relativo à raiz da base (ou ao diretório atual)."""
    for candidate in (Path(root) / rel, Path(rel)):
        if candidate.exists():
            return candidate
    return None

