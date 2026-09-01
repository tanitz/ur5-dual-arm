"""
The Communication tab, and the two steps that read off it.

test_comms.py checks what a link *is* and what it *does* on the wire. This
checks the third thing, which is the one an operator actually touches: that
the tables show what the cell is connected to, that the dialogs refuse what
the wire would refuse, and that a program line offers exactly the names the
tabs set up — no more, because a picker offering a name the checker will then
reject is a picker that lets somebody finish a program that cannot start.

No machine and no robot. The panels are built against a simulated cell.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell                             # noqa: E402
from ur5dual.comms import KINDS, make_item, make_link     # noqa: E402
from ur5dual.config import CellConfig                     # noqa: E402
from ur5dual.gui.app import MainWindow                    # noqa: E402
from ur5dual.gui.panels.network import (                  # noqa: E402
    ItemDialog, LinkDialog, NetworkPanel,
)
from ur5dual.gui.panels.program import INSERTS            # noqa: E402
from ur5dual.gui.panels.step_edit import StepEditDialog   # noqa: E402
from ur5dual.gui.widgets.rail import RAIL_ITEMS            # noqa: E402
from ur5dual.program.steps import Step                    # noqa: E402

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


app = QApplication.instance() or QApplication([])
cfg = CellConfig.load()
cfg.arms["B"].enabled = True
window = MainWindow(cell=Cell(cfg, simulated=True), connect_on_start=False)
window.resize(1280, 800)
app.processEvents()

net = window.panels["net"]
program = window.panels["program"]

print("one tab holds every machine, whatever it speaks")
check("there is one Communication tab and no second protocol tab",
      isinstance(net, NetworkPanel) and "modbus" not in window.panels,
      str(sorted(window.panels)))
check("and it filters nothing out", net.KINDS is None, str(net.KINDS))
check("it edits the library the executor reads",
      net.library is window.link_library())
# The rail is the tab bar, so an icon on it that the sidebar's stack does not
# hold is a button that lights up and shows whatever was already open.
check("every panel the rail names is actually in the sidebar",
      all(window.sidebar.indexOf(window.panels[pid]) >= 0
          for pid, _g, _l, _t in RAIL_ITEMS),
      str([pid for pid, _g, _l, _t in RAIL_ITEMS
           if window.sidebar.indexOf(window.panels[pid]) < 0]))
window.rail.buttons["net"].click()
app.processEvents()
check("pressing it on the rail actually shows it",
      window.sidebar.currentWidget() is net,
      type(window.sidebar.currentWidget()).__name__)

print("\nthe tables show what the cell is connected to")
library = window.link_library()
library.add(make_link("MC1", "tcp", "10.1.68.200", 2000,
                      data=[make_item("START", "send", "ST"),
                            make_item("DONE", "recv", "DN")]))
library.add(make_link("SCAN1", "udp", "10.1.68.201", 5000,
                      data=[make_item("BARCODE", "recv", "",
                                      match="prefix")]))
library.add(make_link("PLC1", "modbus", "10.1.68.210", 502, unit=3,
                      data=[make_item("RUN", "send", 1, area="coil",
                                      register=10),
                            make_item("READY", "recv", 1, area="discrete",
                                      register=4),
                            make_item("SPEED", "send", 250, area="holding",
                                      register=20)]))
window.links_changed()
net.refresh()
app.processEvents()


def rows(column):
    return [net.links_table.item(r, column).text()
            for r in range(net.links_table.rowCount())]


check("all three are on the one list, in the order they were set up",
      rows(0) == ["MC1", "SCAN1", "PLC1"], str(rows(0)))
check("each row names its own protocol rather than making it be guessed",
      rows(1) == ["TCP", "UDP", "Modbus TCP"], str(rows(1)))
check("and the address it will dial",
      net.links_table.item(0, 2).text() == "10.1.68.200:2000",
      net.links_table.item(0, 2).text())
check("a Modbus row carries the device id, because two devices share a port",
      net.links_table.item(2, 2).text() == "10.1.68.210:502  #3",
      net.links_table.item(2, 2).text())

net.links_table.selectRow(0)
app.processEvents()
check("selecting a machine shows what it carries",
      [net.items_table.item(r, 0).text()
       for r in range(net.items_table.rowCount())] == ["START", "DONE"],
      str([net.items_table.item(r, 0).text()
           for r in range(net.items_table.rowCount())]))
check("and which way each goes",
      net.items_table.item(0, 1).text() == "→ send"
      and net.items_table.item(1, 1).text() == "← recv")
check("the strip says whose data is on the table",
      net.items_strip.text() == "Data on MC1", net.items_strip.text())
net.links_table.selectRow(1)
app.processEvents()
check("and changing machine changes it",
      net.items_strip.text() == "Data on SCAN1"
      and net.items_table.rowCount() == 1)
net.links_table.selectRow(2)
app.processEvents()
check("a Modbus machine's data reads as registers on the same table",
      [net.items_table.item(r, 2).text()
       for r in range(net.items_table.rowCount())]
      == ["coil 10 = 1", "discrete 4 = 1", "holding 20 = 250"],
      str([net.items_table.item(r, 2).text()
           for r in range(net.items_table.rowCount())]))

print("\nthe dialogs refuse what the wire would refuse")


def refuses(dialog, fragment):
    problems = dialog.check()
    return any(fragment in p for p in problems), str(problems)


picker = LinkDialog(None, make_link("", "tcp", "", 2000), KINDS)
check("the protocol is one dropdown carrying all three",
      [picker.kind.itemData(i) for i in range(picker.kind.count())]
      == ["tcp", "udp", "modbus"],
      str([picker.kind.itemText(i) for i in range(picker.kind.count())]))
check("a machine with no name is refused — the name is what a line says",
      *refuses(picker, "needs a name"))
check("and one whose name is taken",
      *refuses(LinkDialog(None, make_link("MC1", "tcp", "1.2.3.4", 2000),
                          KINDS, taken=["MC1"]), "already a machine"))

typed = LinkDialog(None, make_link("MC2", "tcp", "10.1.68.202", 2000), KINDS)
check("a link typed in comes back out as what was typed",
      typed.result_value()["host"] == "10.1.68.202"
      and typed.result_value()["port"] == 2000
      and typed.check() == [], str(typed.check()))
check("its terminator is shown the way it is stored, not escaped twice",
      typed.terminator.text() == "\\r\\n", repr(typed.terminator.text()))
check("and comes back out of the dialog unchanged by having been looked at",
      typed.result_value()["terminator"] == "\\r\\n",
      repr(typed.result_value()["terminator"]))

carrier = ItemDialog(None, window.link_library().get("MC1"),
                     make_item("SPLIT", "both", "A\\tB"))
check("a payload with an escape in it survives the round trip too",
      carrier.value.text() == "A\\tB"
      and carrier.result_value()["value"] == "A\\tB",
      repr(carrier.value.text()))

modbus_dialog = LinkDialog(None, make_link("PLC2", "modbus", "10.1.68.211",
                                           502), KINDS)
check("a Modbus machine shows its device id and hides the terminator",
      modbus_dialog.unit_row.isVisibleTo(modbus_dialog)
      and not modbus_dialog.terminator_row.isVisibleTo(modbus_dialog))
check("while a socket does the opposite",
      typed.terminator_row.isVisibleTo(typed)
      and not typed.unit_row.isVisibleTo(typed))

# The whole point of one tab: the fields follow the dropdown rather than the
# operator having had to pick the right page before they could start.
swapping = LinkDialog(None, make_link("MC3", "tcp", "10.1.68.203", 2000), KINDS)
swapping.kind.setCurrentIndex(swapping.kind.findData("modbus"))
app.processEvents()
check("moving the dropdown to Modbus swaps the fields under it",
      swapping.unit_row.isVisibleTo(swapping)
      and not swapping.terminator_row.isVisibleTo(swapping))
check("and offers that protocol's port instead of the socket one",
      swapping.port.value() == 502, str(swapping.port.value()))
swapping.kind.setCurrentIndex(swapping.kind.findData("udp"))
app.processEvents()
check("moving it back swaps them back",
      swapping.terminator_row.isVisibleTo(swapping)
      and not swapping.unit_row.isVisibleTo(swapping)
      and swapping.port.value() == 2000)
check("and what comes out is the protocol that was left showing",
      swapping.result_value()["kind"] == "udp",
      swapping.result_value()["kind"])

link = window.link_library().get("MC1")
plc = window.link_library().get("PLC1")
check("a datum with no name is refused",
      *refuses(ItemDialog(None, link, make_item("", "send", "ST")),
               "needs a name"))
check("a Modbus input register may not be written, in the dialog that made it",
      *refuses(ItemDialog(None, plc, make_item("R", "send", 1, area="input",
                                               register=5)), "read-only"))
check("and a coil holding 7 is caught before it is saved",
      *refuses(ItemDialog(None, plc, make_item("R", "send", 7, area="coil",
                                               register=5)), "0 or 1"))

send_only = ItemDialog(None, link, make_item("GO", "send", "GO"))
recv_item = ItemDialog(None, link, make_item("BACK", "recv", "OK"))
check("how a message is matched is a question about receiving only",
      not send_only.match_row.isVisibleTo(send_only)
      and recv_item.match_row.isVisibleTo(recv_item))

print("\nthe program line offers the tabs' names, and only those")
check("the insert list carries both steps",
      [n for n, _mk in INSERTS if n in ("SEND DATA", "RECV DATA")]
      == ["SEND DATA", "RECV DATA"])


def editor(step):
    return StepEditDialog(step, window.points, cell=window.cell,
                          links=window.link_library())


sending = editor(Step("SEND", link="", item="", text=""))
check("every machine is offered, whichever protocol it speaks",
      [sending.link_pick.itemText(i) for i in range(sending.link_pick.count())]
      == ["", "MC1", "SCAN1", "PLC1"],
      str([sending.link_pick.itemText(i)
           for i in range(sending.link_pick.count())]))
sending.link_pick.setCurrentText("MC1")
app.processEvents()
check("a SEND offers only what this cell may send",
      [sending.item_pick.itemText(i)
       for i in range(sending.item_pick.count())] == ["", "START"],
      str([sending.item_pick.itemText(i)
           for i in range(sending.item_pick.count())]))
check("and says what the chosen datum carries, so it need not be remembered",
      "TCP" in sending.comms_hint.text()
      and "10.1.68.200" in sending.comms_hint.text(),
      sending.comms_hint.text())

sending.item_pick.setCurrentText("START")
built = sending.get_step()
check("what comes back is the line that was picked",
      built.kind == "SEND" and built.get("link") == "MC1"
      and built.get("item") == "START", str(built.fields))
check("and it reads as two names rather than an address",
      built.describe() == "send START to MC1", built.describe())

waiting = editor(Step("RECV", link="MC1", item="", into="", timeout=0.0))
check("a RECV offers only what this cell may wait for",
      [waiting.item_pick.itemText(i)
       for i in range(waiting.item_pick.count())] == ["", "DONE"],
      str([waiting.item_pick.itemText(i)
           for i in range(waiting.item_pick.count())]))
waiting.item_pick.setCurrentText("DONE")
waiting.timeout.setValue(30)
waiting.into_edit.setText("reply")
built = waiting.get_step()
check("and it comes back with the wait on it",
      built.get("timeout") == 30.0 and built.get("into") == "reply")
check("which is what the line says",
      built.describe() == "wait for DONE from MC1 -> reply  (up to 30 s)",
      built.describe())

on_modbus = editor(Step("SEND", link="PLC1", item="", text=""))
check("Modbus has no free text to fall back on, so it is not offered",
      not on_modbus.text_row.isVisibleTo(on_modbus))
check("while a socket keeps the escape hatch",
      sending.text_row.isVisibleTo(sending))

print("\na name the tabs no longer have is shown, not silently dropped")
elsewhere = editor(Step("SEND", link="OTHERCELL", item="FIRE"))
check("the machine survives being opened on a cell that never had it",
      elsewhere.link_pick.currentText() == "OTHERCELL",
      elsewhere.link_pick.currentText())
check("and so does the datum",
      elsewhere.get_step().get("item") == "FIRE",
      str(elsewhere.get_step().fields))

print("\nthe step list shows the line")
program.program.steps.clear()
program.program.add(Step("SEND", link="MC1", item="START"))
program.program.add(Step("RECV", link="MC1", item="DONE", timeout=30.0))
program.refresh()
app.processEvents()
check("both rows are named by what they do",
      [program.table.item(r, 1).text() for r in range(2)]
      == ["SEND DATA", "RECV DATA"],
      str([program.table.item(r, 1).text() for r in range(2)]))
check("and the checker passes a line that names what the cell has",
      program.problems.text() == "", program.problems.text())

window.link_library().remove("MC1")
window.links_changed()
program.refresh()
app.processEvents()
check("deleting the machine off the tab turns both lines red, on paper",
      "no machine called 'MC1'" in program.problems.text(),
      program.problems.text()[:90])

window.close()
print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
