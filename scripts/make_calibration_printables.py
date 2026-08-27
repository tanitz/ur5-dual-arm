#!/usr/bin/env python3
"""Generate the exact-size ArUco/ChArUco sheets used by this cell.

The output is vector PDF and SVG on A4.  No office application or printer
driver is allowed to decide the geometry: all dimensions are written in mm,
and every sheet carries a 100 mm check line for the printed copy.
"""

from pathlib import Path

import cv2


A4 = (210.0, 297.0)
PT_PER_MM = 72.0 / 25.4
DICTIONARY_NAME = "DICT_4X4_50"
DICTIONARY_ID = cv2.aruco.DICT_4X4_50
OUT = Path(__file__).resolve().parents[1] / "docs" / "calibration"


def marker_bits(marker_id):
    dictionary = cv2.aruco.Dictionary_get(DICTIONARY_ID)
    # 4 data modules plus one black border module on every side.
    image = cv2.aruco.drawMarker(dictionary, marker_id, 6, borderBits=1)
    return [[int(value) == 0 for value in row] for row in image]


def marker_rectangles(marker_id, x, y, size):
    bits = marker_bits(marker_id)
    module = size / len(bits)
    return [(x + col * module, y + row * module, module, module)
            for row, values in enumerate(bits)
            for col, black in enumerate(values) if black]


def charuco_rectangles(x, y, squares_x=6, squares_y=8,
                       square=25.0, marker=18.0):
    rectangles = []
    marker_id = 0
    inset = (square - marker) / 2.0
    for row in range(squares_y):
        for col in range(squares_x):
            cell_x, cell_y = x + col * square, y + row * square
            if (row + col) % 2 == 1:
                # OpenCV ChArUco uses solid black chess squares and puts the
                # ArUco codes in the white squares.
                rectangles.append((cell_x, cell_y, square, square))
                continue
            bits = marker_bits(marker_id)
            module = marker / len(bits)
            for bit_row, values in enumerate(bits):
                for bit_col, black in enumerate(values):
                    if black:
                        rectangles.append((cell_x + inset + bit_col * module,
                                           cell_y + inset + bit_row * module,
                                           module, module))
            marker_id += 1
    return rectangles


def svg(path, title, black_rectangles, white_rectangles=(), guides=()):
    width, height = A4
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="210mm" height="297mm" viewBox="0 0 210 297">',
        '<rect width="210" height="297" fill="white"/>',
        '<g shape-rendering="crispEdges" fill="black">',
    ]
    for x, y, w, h in black_rectangles:
        parts.append(f'<rect x="{x:.6f}" y="{y:.6f}" '
                     f'width="{w:.6f}" height="{h:.6f}"/>')
    parts.append('</g><g shape-rendering="crispEdges" fill="white">')
    for x, y, w, h in white_rectangles:
        parts.append(f'<rect x="{x:.6f}" y="{y:.6f}" '
                     f'width="{w:.6f}" height="{h:.6f}"/>')
    parts.append('</g>')
    for x1, y1, x2, y2 in guides:
        parts.append(f'<path d="M{x1:.6f},{y1:.6f} L{x2:.6f},{y2:.6f}" '
                     'stroke="black" stroke-width="0.25" fill="none"/>')
    parts.extend([
        f'<text x="105" y="12" text-anchor="middle" '
        f'font-family="sans-serif" font-size="4">{title}</text>',
        '<text x="55" y="286" text-anchor="middle" '
        'font-family="sans-serif" font-size="3.2">100 mm scale check</text>',
        '<path d="M5,280 L105,280 M5,278 L5,282 M105,278 L105,282" '
        'stroke="black" stroke-width="0.35" fill="none"/>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf(path, title, black_rectangles, white_rectangles=(), guides=()):
    page_w, page_h = (value * PT_PER_MM for value in A4)

    def rect_command(rectangle, grey):
        x, y, width, height = rectangle
        return (f"{grey} g {x * PT_PER_MM:.4f} "
                f"{page_h - (y + height) * PT_PER_MM:.4f} "
                f"{width * PT_PER_MM:.4f} {height * PT_PER_MM:.4f} re f")

    commands = [rect_command(rectangle, 0) for rectangle in black_rectangles]
    commands.extend(rect_command(rectangle, 1) for rectangle in white_rectangles)
    commands.append("0 g 0.25 w")
    for x1, y1, x2, y2 in guides:
        commands.append(
            f"{x1 * PT_PER_MM:.4f} {page_h - y1 * PT_PER_MM:.4f} m "
            f"{x2 * PT_PER_MM:.4f} {page_h - y2 * PT_PER_MM:.4f} l S")
    commands.extend([
        f"BT /F1 11 Tf 1 0 0 1 {55 * PT_PER_MM:.4f} "
        f"{page_h - 12 * PT_PER_MM:.4f} Tm ({_pdf_escape(title)}) Tj ET",
        f"BT /F1 9 Tf 1 0 0 1 {27 * PT_PER_MM:.4f} "
        f"{page_h - 286 * PT_PER_MM:.4f} Tm (100 mm scale check) Tj ET",
        "0 g 1 w",
        f"{5 * PT_PER_MM:.4f} {page_h - 280 * PT_PER_MM:.4f} m "
        f"{105 * PT_PER_MM:.4f} {page_h - 280 * PT_PER_MM:.4f} l S",
        f"{5 * PT_PER_MM:.4f} {page_h - 278 * PT_PER_MM:.4f} m "
        f"{5 * PT_PER_MM:.4f} {page_h - 282 * PT_PER_MM:.4f} l S",
        f"{105 * PT_PER_MM:.4f} {page_h - 278 * PT_PER_MM:.4f} m "
        f"{105 * PT_PER_MM:.4f} {page_h - 282 * PT_PER_MM:.4f} l S",
    ])
    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} "
         f"{page_h:.4f}] /Resources << /Font << /F1 5 0 R >> >> "
         "/Contents 4 0 R >>").encode("ascii"),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
         f"startxref\n{xref}\n%%EOF\n").encode("ascii"))
    path.write_bytes(output)


def make_marker():
    marker_id, marker_size, quiet = 42, 50.0, 10.0
    card = marker_size + 2 * quiet
    card_x, card_y = (A4[0] - card) / 2, (A4[1] - card) / 2
    marker_x, marker_y = card_x + quiet, card_y + quiet
    black = marker_rectangles(marker_id, marker_x, marker_y, marker_size)
    crop = 3.0
    guides = [
        (card_x - crop, card_y, card_x + crop, card_y),
        (card_x, card_y - crop, card_x, card_y + crop),
        (card_x + card - crop, card_y, card_x + card + crop, card_y),
        (card_x + card, card_y - crop, card_x + card, card_y + crop),
        (card_x - crop, card_y + card, card_x + crop, card_y + card),
        (card_x, card_y + card - crop, card_x, card_y + card + crop),
        (card_x + card - crop, card_y + card,
         card_x + card + crop, card_y + card),
        (card_x + card, card_y + card - crop,
         card_x + card, card_y + card + crop),
    ]
    title = f"ArUco {DICTIONARY_NAME} ID {marker_id} - marker 50 mm"
    svg(OUT / "aruco_4x4_50_id42_50mm_a4.svg", title, black, guides=guides)
    pdf(OUT / "aruco_4x4_50_id42_50mm_a4.pdf", title, black, guides=guides)


def make_board():
    board_w, board_h = 6 * 25.0, 8 * 25.0
    board_x, board_y = (A4[0] - board_w) / 2, (A4[1] - board_h) / 2
    black = charuco_rectangles(board_x, board_y)
    title = "ChArUco 6x8 - square 25 mm - marker 18 mm - DICT_4X4_50"
    svg(OUT / "charuco_6x8_25mm_18mm_4x4_50_a4.svg",
        title, black)
    pdf(OUT / "charuco_6x8_25mm_18mm_4x4_50_a4.pdf",
        title, black)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_marker()
    make_board()
    print("wrote calibration sheets to", OUT)


if __name__ == "__main__":
    main()
