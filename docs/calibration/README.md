# Printable camera-calibration targets

Print the PDF files on A4 at **Actual size / 100%**. Disable Fit, Shrink,
Scale-to-page and borderless enlargement. Measure the line labelled `100 mm
scale check`; do not use a sheet unless that line measures 100.0 mm.

## ChArUco board

- File: `charuco_6x8_25mm_18mm_4x4_50_a4.pdf`
- Chessboard: `6 x 8` squares
- Square length: `25 mm`
- ArUco marker length: `18 mm`
- OpenCV dictionary: `DICT_4X4_50`
- Printed board area: `150 x 200 mm`

Mount the sheet on a flat plate without stretching it. Re-measure several
25 mm squares after mounting; adhesive and paper moisture can distort a good
print.

The Camera panel's `Measure Surface + Box` popup reads this board. It uses the
main GUI's existing camera stream, so no second process competes for the
RealSense.

Two pictures, in this order, with the camera left alone in between:

1. **the box on the table, board nowhere in sight** — press `1 Capture Box`
2. **the board lying where the box stood** — press `2 Capture Surface`
3. press `Apply + Save` to write `vision.surface` and `vision.box_size`

`Reset` starts the two captures again.

Staged rather than both at once, because the two cannot share a picture. The
board is a large bright rectangle lying on exactly the surface being measured,
and no rule of shape can tell it from a box: scaling a rectangle leaves a
rectangle, so the sheet read at the rim's height is a tidy rectangle too — a
bigger one, which is worse, because bigger is how a box gets chosen. Taking
them one at a time removes the question rather than answering it.

What must not move between the two is the camera. What must be shared is the
surface: the board goes *where the box stood*, not beside it.

Nothing is typed. The opening comes from the rim's four corners meeting the
plane; the rim's height above that plane comes from the depth stream along the
edge of the picture the box was captured in, so a box that is not the height
the config claims is measured at the height it really is. `--height` exists
only for a sensor with nothing to say there.

Lay it flat on the surface the boxes stand on, beside the box and not under
it, with the whole sheet in view. What it is for is the one error that has no
symptom: the detector places a box at whatever depth makes an opening of the
*configured* size fit the rim it found, so a size typed 5% large reads 5% too
far away and moves the arms 5% too far sideways, with every direction still
correct. The board's squares are 25 mm because they were drawn 25 mm, so it
can contradict that; `--box` meets the four rim rays with the plane the rim
stands on and measures the opening outright.

If the check line did not come out at 100 mm, reprint it — a printer left on
"fit to page" typically shrinks a sheet to about 95%. When reprinting is not
possible, measure the line as precisely as you can and say so:

The integrated workflow expects an actual-size print. Reprint a scaled sheet;
do not enter a correction factor into the production GUI.

A rescaled sheet is not a damaged one: detection reads patterns and cares
nothing for size, and only the pose carries metres. But nothing downstream can
check that number. A board declared 5% large fits its own corners exactly as
well and simply sits 5% further away — measured, 625 mm instead of 594 mm, at
0.24 px reprojection either way. It is the same error the board was printed to
expose, moved one step back, so measure over the longest baseline available:
the printed board is 150 mm across, and `100 x measured / 150` is a better
scale figure than a ruler laid along the 100 mm line.

Measured against a rendered board, an A4 sheet at this cell's 700 mm working
distance places the surface to within 0.5 mm and 0.2 deg. Past about a metre
the 18 mm codes run out of pixels at 640x480 and the answer degrades to
several millimetres — print the board larger, or bring it nearer, rather than
believing it out there.

The aligned RealSense colour stream supplies lens intrinsics from the device
already, so nothing here refits them; the board is used as a known plane, not
as a lens target.

Regenerate both PDF and SVG files with:

```bash
python scripts/make_calibration_printables.py
```
