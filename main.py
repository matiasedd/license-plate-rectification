import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from utils import read_split, resolve, parse_annotation, transform_image, make_panel

SPLITS = ("training", "validation", "testing")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="RodoSol-ALPR", help="raiz da base de dados")
    ap.add_argument("--split-file", default=None, help="padrão: <root>/split.txt")
    ap.add_argument("--split", default="all", choices=SPLITS + ("all",))
    ap.add_argument("--output", default="output", help="diretório de saída")
    ap.add_argument("--limit", type=int, default=0, help="máximo de amostras (0 = todas)")
    ap.add_argument("--figure", type=int, default=0, help="gera figura com N exemplos")
    args = ap.parse_args()

    root = Path(args.root)
    split_file = Path(args.split_file) if args.split_file else root / "split.txt"
    samples = read_split(split_file)
    if args.split != "all":
        samples = [s for s in samples if s[1] == args.split]
    if args.limit:
        samples = samples[: args.limit]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    done, failed, widths, heights, elapsed = 0, [], [], [], []
    panels = []

    for rel_img, part in samples:
        img_path = resolve(root, rel_img)
        ann_path = resolve(root, str(Path(rel_img).with_suffix(".txt")))
        if img_path is None or ann_path is None:
            failed.append((rel_img, "arquivo não encontrado"))
            continue
        try:
            plate, corners = parse_annotation(ann_path)
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("imagem ilegível")
            t0 = time.perf_counter()
            rectified = transform_image(img, corners)
            elapsed.append(time.perf_counter() - t0)
        except Exception as exc:  # uma amostra ruim não interrompe o lote
            failed.append((rel_img, str(exc)))
            continue

        dest = out / part / f"{img_path.stem}_{plate}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), rectified)

        heights.append(rectified.shape[0])
        widths.append(rectified.shape[1])
        done += 1
        if len(panels) < args.figure:
            panels.append(make_panel(img, corners, rectified))

    if panels:
        width = max(p.shape[1] for p in panels)
        rows = [
            np.hstack([p, np.full((p.shape[0], width - p.shape[1], 3), 255, np.uint8)])
            for p in panels
        ]
        cv2.imwrite(str(out / "figure.png"), np.vstack(rows))

    stats = {
        "split": args.split,
        "amostras": len(samples),
        "processadas": done,
        "falhas": len(failed),
        "largura_media": float(np.mean(widths)) if widths else 0.0,
        "altura_media": float(np.mean(heights)) if heights else 0.0,
        "tempo_medio_ms": 1000 * float(np.mean(elapsed)) if elapsed else 0.0,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()