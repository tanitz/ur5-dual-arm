"""The machines this cell stands next to, and the names a program calls them.

One list of machines, whatever they speak, with a Test button that proves the
address from the tab that set it up rather than from three lines into a
program with an arm already somewhere.

The protocol is a field on the machine — a dropdown in the dialog that sets it
up — rather than a tab the operator has to have picked correctly before they
can look for one. A cell has *machines*; which of them happens to answer
Modbus is a detail of how it is reached, not a different kind of thing to keep
somewhere else. Split across two tabs, a cell with one PLC and one labeller
had half its equipment on each, and the question an operator actually asks —
what is this cell wired to — could not be answered from either.

What the protocol still decides is what the dialogs *show*. A raw TCP or UDP
link carries whatever bytes an item names and is *heard from* — messages
arrive when the machine decides to send them, split on a terminator. A Modbus
link carries numbers at register addresses and is *asked* — nothing arrives at
all, and waiting means reading the same register until it says what the step
wanted. So the fields swap when the dropdown moves, rather than half of them
sitting greyed out whichever protocol is chosen.

The two tables are stacked rather than side by side, so each keeps the full
sidebar width and leaves enough room to read its names and values.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...comms.links import (
    AREA_LABEL, AREAS, DIRECTIONS, FORMATS, KIND_LABEL, KINDS, MATCHES,
    MODBUS_KINDS, REGISTER_MAX, REGISTER_VALUE_MAX, address, check_item,
    check_link, describe_item, is_modbus, make_item, make_link,
)
from .. import style as S

# What a direction is called on the row. The stored value never changes with
# it — a link file written before a rename still loads.
DIRECTION_LABEL = {"send": "→ send", "recv": "← recv", "both": "↔ both"}


def _table(columns, widths):
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.verticalHeader().hide()
    table.setStyleSheet(f"font-size:{S.fpx(12)}px;")
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    header = table.horizontalHeader()
    for i, mode in enumerate(widths):
        header.setSectionResizeMode(i, mode)
    return table


def _cell(text, tip=None):
    item = QTableWidgetItem(str(text))
    if tip:
        item.setToolTip(tip)
    return item


class NetworkPanel(QWidget):
    """Every machine the cell talks to, and the data items on each.

    `KINDS` of None is "all of them", which is what `of_kind` already means by
    it. It stays a field rather than becoming a hard-coded call because it is
    the one thing that would have to change to split this page again, and a
    named reason is easier to find later than a filter that was deleted.
    """

    KINDS = None
    TITLE = "Communication — machines"
    BLURB = ("One list, whatever they speak. The protocol is set per machine: "
             "TCP or UDP for a socket carrying strings or byte frames, Modbus "
             "TCP for registers and coils at an address.")

    def __init__(self, app):
        super().__init__()
        self.app = app

        body = QVBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(6))
        body.addWidget(self._build_links(), 1)
        body.addWidget(self._build_items(), 1)
        body.addWidget(self._build_actions(), 0)

    # ---- the library this tab is editing ---------------------------------
    @property
    def library(self):
        """The one the app holds, not a copy.

        Edits land in the config's block immediately and are written to disk
        by Save. A tab that kept its own copy would let two tabs disagree
        about what this cell is connected to, and the program would run
        against whichever one saved last.
        """
        return self.app.link_library()

    def _selected_link(self):
        row = self.links_table.currentRow()
        links = self.library.of_kind(self.KINDS)
        return links[row] if 0 <= row < len(links) else None

    def _selected_item(self):
        link = self._selected_link()
        if link is None:
            return None
        row = self.items_table.currentRow()
        items = list(link.get("data") or [])
        return items[row] if 0 <= row < len(items) else None

    # ---- layout ----------------------------------------------------------
    def _build_links(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(4))
        v.addWidget(S.strip(self.TITLE))
        if self.BLURB:
            blurb = QLabel(self.BLURB)
            blurb.setWordWrap(True)
            blurb.setStyleSheet(f"font-size:{S.fpx(11)}px;color:#555555;")
            v.addWidget(blurb)

        self.links_table = _table(
            ["name", "protocol", "address"],
            [QHeaderView.ResizeToContents, QHeaderView.ResizeToContents,
             QHeaderView.Stretch])
        self.links_table.currentCellChanged.connect(
            lambda *_a: self._show_items())
        self.links_table.doubleClicked.connect(lambda *_a: self._edit_link())
        v.addWidget(self.links_table, 1)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        for label, tip, callback, colour in (
                ("＋ Add", "set up another machine", self._add_link, S.GREEN),
                ("✏ Edit", "change this machine's address",
                 self._edit_link, None),
                ("✕ Del", "forget this machine", self._delete_link, S.RED)):
            button = S.touch_button(label, colour, height=38, font_px=12)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            row.addWidget(button, 1)
        v.addLayout(row)
        return page

    def _build_items(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(4))
        self.items_strip = S.strip("Data", "#e8eef8")
        v.addWidget(self.items_strip)

        self.items_table = _table(
            ["name", "way", "what"],
            [QHeaderView.ResizeToContents, QHeaderView.ResizeToContents,
             QHeaderView.Stretch])
        self.items_table.doubleClicked.connect(lambda *_a: self._edit_item())
        v.addWidget(self.items_table, 1)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        for label, tip, callback, colour in (
                ("＋ Add", "name another thing this machine sends or is sent",
                 self._add_item, S.GREEN),
                ("✏ Edit", "change what this datum carries",
                 self._edit_item, None),
                ("✕ Del", "forget this datum", self._delete_item, S.RED)):
            button = S.touch_button(label, colour, height=38, font_px=12)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            row.addWidget(button, 1)
        v.addLayout(row)
        return page

    def _build_actions(self):
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(S.sx(4))

        test = S.touch_button("⇄ Test", S.BLUE, height=40, font_px=13)
        test.setToolTip("open this machine now and say whether it answered")
        test.clicked.connect(self._test)
        row.addWidget(test, 1)

        save = S.touch_button("💾 Save", S.PURPLE, height=40, font_px=13)
        save.setToolTip("write these machines into cell.yaml")
        save.clicked.connect(self._save)
        row.addWidget(save, 1)

        self.status = S.caption("")
        self.status.setAlignment(Qt.AlignCenter)
        row.addWidget(self.status, 1)
        return page

    # ---- showing what is there -------------------------------------------
    def refresh(self):
        links = self.library.of_kind(self.KINDS)
        keep = self.links_table.currentRow()
        self.links_table.setRowCount(len(links))
        for row, link in enumerate(links):
            self.links_table.setItem(row, 0, _cell(link.get("name", "")))
            self.links_table.setItem(
                row, 1, _cell(KIND_LABEL.get(link.get("kind"), "?")))
            self.links_table.setItem(
                row, 2, _cell(address(link),
                              "%d data item(s)" % len(link.get("data") or [])))
        if links and not 0 <= keep < len(links):
            self.links_table.selectRow(0)
        elif 0 <= keep < len(links):
            self.links_table.selectRow(keep)
        self._show_items()

    def _show_items(self):
        link = self._selected_link()
        self.items_strip.setText(
            "Data on %s" % link["name"] if link else "Data")
        items = list((link or {}).get("data") or [])
        self.items_table.setRowCount(len(items))
        for row, item in enumerate(items):
            self.items_table.setItem(row, 0, _cell(item.get("name", "")))
            self.items_table.setItem(
                row, 1, _cell(DIRECTION_LABEL.get(item.get("direction"), "?")))
            self.items_table.setItem(row, 2, _cell(describe_item(link, item)))

    def _say(self, text):
        self.status.setText(text)
        self.app.log(text)

    # ---- links -----------------------------------------------------------
    def _add_link(self):
        seed = make_link("", "tcp", "", None)
        link = LinkDialog(self, seed, KINDS, taken=self._taken()).ask()
        if link is None:
            return
        self.library.add(link)
        self.app.links_changed()
        self.refresh()
        self._say("added %s" % link["name"])

    def _edit_link(self):
        current = self._selected_link()
        if current is None:
            return
        edited = LinkDialog(self, current, KINDS,
                            taken=self._taken(current["name"])).ask()
        if edited is None:
            return
        # the items belong to the machine, not to the dialog that never showed
        # them: an address corrected must not empty the list of what it carries
        edited["data"] = list(current.get("data") or [])
        self.library.replace(current["name"], edited)
        self.app.links_changed()
        self.refresh()
        # Switching protocol keeps the data items, which is right — they are
        # what this machine carries. But a payload of "ST" is not a register
        # value and register 10 is not a string, so it is said here, beside
        # the list that has to be gone through, rather than only by Save.
        if edited["kind"] != current["kind"] and edited["data"]:
            self._say("%s now speaks %s — check its %d data item(s), a "
                      "payload is not a register"
                      % (edited["name"], KIND_LABEL[edited["kind"]],
                         len(edited["data"])))
        else:
            self._say("changed %s" % edited["name"])

    def _delete_link(self):
        link = self._selected_link()
        if link is None:
            return
        self.library.remove(link["name"])
        self.app.links_changed()
        self.refresh()
        self._say("forgot %s — any program line naming it will now say so"
                  % link["name"])

    def _taken(self, allow=None):
        return [n for n in self.library.names() if n != allow]

    # ---- data items ------------------------------------------------------
    def _add_item(self):
        link = self._selected_link()
        if link is None:
            self._say("set a machine up first")
            return
        seed = make_item("", "both", 1 if is_modbus(link) else "")
        item = ItemDialog(self, link, seed, taken=self._items_taken(link)).ask()
        if item is None:
            return
        link.setdefault("data", []).append(item)
        self.app.links_changed()
        self._show_items()
        self._say("%s now carries %s" % (link["name"], item["name"]))

    def _edit_item(self):
        link, current = self._selected_link(), self._selected_item()
        if link is None or current is None:
            return
        edited = ItemDialog(self, link, current,
                            taken=self._items_taken(link, current["name"])).ask()
        if edited is None:
            return
        link["data"][self.items_table.currentRow()] = edited
        self.app.links_changed()
        self._show_items()
        self._say("changed %s %s" % (link["name"], edited["name"]))

    def _delete_item(self):
        link, current = self._selected_link(), self._selected_item()
        if link is None or current is None:
            return
        link["data"].pop(self.items_table.currentRow())
        self.app.links_changed()
        self._show_items()
        self._say("%s no longer carries %s" % (link["name"], current["name"]))

    @staticmethod
    def _items_taken(link, allow=None):
        return [i.get("name") for i in (link.get("data") or [])
                if i.get("name") != allow]

    # ---- proving it ------------------------------------------------------
    def _test(self):
        link = self._selected_link()
        if link is None:
            self._say("nothing selected to test")
            return
        self._say(self.app.links.probe(link["name"]))

    def _save(self):
        problems = self.library.check()
        if problems:
            self._say(problems[0])
            for text in problems[1:]:
                self.app.log(text)
            return
        self.app.save_links()
        self._say("saved %d machine(s)" % len(self.library.links))


# ── the two dialogs ───────────────────────────────────────────────────────
class _Dialog(QDialog):
    """Shared plumbing: the OK/Cancel bar, and refusing to close on nonsense.

    Checked here rather than on Save, because the field that is wrong is on
    screen with a cursor in it now and will not be in ten minutes.
    """

    def __init__(self, parent, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(S.sx(380))
        self.form = QVBoxLayout(self)
        self.form.setContentsMargins(S.sx(10), S.sx(10), S.sx(10), S.sx(10))
        self.form.setSpacing(S.sx(6))
        self.problem = QLabel("")
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet(f"color:{S.RED};font-size:{S.fpx(12)}px;")

    def _finish(self):
        self.form.addWidget(self.problem)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        self.form.addWidget(buttons)

    def _row(self, label, widget, tip=None):
        row = QHBoxLayout()
        row.setSpacing(S.sx(6))
        caption = S.caption(label)
        caption.setFixedWidth(S.sx(86))
        row.addWidget(caption)
        row.addWidget(widget, 1)
        if tip:
            widget.setToolTip(tip)
        holder = QWidget()
        holder.setLayout(row)
        self.form.addWidget(holder)
        return holder

    def _edit(self, text=""):
        field = QLineEdit(str(text))
        field.setMinimumHeight(S.sx(30))
        field.setStyleSheet(S.field())
        return field

    def _pick(self, items, current=None, labels=None):
        combo = QComboBox()
        for value in items:
            combo.addItem((labels or {}).get(value, str(value)), value)
        if current is not None:
            index = combo.findData(current)
            combo.setCurrentIndex(max(0, index))
        combo.setMinimumHeight(S.sx(30))
        combo.setStyleSheet(S.combo())
        return combo

    def _int(self, low, high, value):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(int(value))
        spin.setMinimumHeight(S.sx(30))
        spin.setStyleSheet(S.field())
        return spin

    def _try_accept(self):
        problems = self.check()
        if problems:
            self.problem.setText("\n".join(problems))
            return
        self.accept()

    def ask(self):
        return self.result_value() if self.exec_() == QDialog.Accepted else None


class LinkDialog(_Dialog):
    """One machine: what it is called, and how to reach it."""

    def __init__(self, parent, link, kinds, taken=()):
        super().__init__(parent, "Machine")
        self.taken = set(taken)
        self.kinds = tuple(kinds)

        self.name = self._edit(link.get("name", ""))
        self._row("name", self.name,
                  "what a program line calls it — MC1, PLC1, labeller")
        self.kind = self._pick(self.kinds, link.get("kind"), KIND_LABEL)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._row("protocol", self.kind)
        self.host = self._edit(link.get("host", ""))
        self._row("address", self.host, "10.1.68.200")
        self.port = self._int(1, 65535, link.get("port", 2000) or 2000)
        self._row("port", self.port)
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(0.1, 120.0)
        self.timeout.setDecimals(1)
        self.timeout.setSuffix(" s")
        self.timeout.setValue(float(link.get("timeout", 3.0) or 3.0))
        self.timeout.setMinimumHeight(S.sx(30))
        self.timeout.setStyleSheet(S.field())
        self._row("connect in", self.timeout,
                  "how long dialling may take before it is called unreachable\n"
                  "This is not how long a RECV waits — that is on the step.")

        # raw only
        self.format = self._pick(FORMATS, link.get("format", "string"))
        self.format_row = self._row(
            "data", self.format,
            "string  the machine speaks text\n"
            "hex     it speaks a byte frame, written 02 41 03")
        self.encoding = self._edit(link.get("encoding", "utf-8"))
        self.encoding_row = self._row("encoding", self.encoding, "utf-8, ascii")
        # Shown as stored: `terminator` is already in escaped notation, and
        # escaping it again is what put `\\\\r\\\\n` in this field.
        self.terminator = self._edit(link.get("terminator", "\\r\\n"))
        self.terminator_row = self._row(
            "ends with", self.terminator,
            "what marks the end of one message, typed \\r\\n\n"
            "It is added to everything sent, and is what incoming bytes are "
            "split on.\nLeave it empty for a machine that frames its own "
            "packets.")

        # modbus only
        self.unit = self._int(0, 255, link.get("unit", 1) or 1)
        self.unit_row = self._row("device id", self.unit,
                                  "the Modbus unit/slave id, usually 1")

        self._kind_changed()
        self._finish()

    def _kind_changed(self):
        modbus = self.kind.currentData() in MODBUS_KINDS
        for row in (self.format_row, self.encoding_row, self.terminator_row):
            row.setVisible(not modbus)
        self.unit_row.setVisible(modbus)
        # A port typed for the other protocol is the wrong port far more often
        # than it is a deliberate choice, so switching offers that protocol's
        # own default — but only from the other default, never over a number
        # somebody actually entered.
        if modbus and self.port.value() == 2000:
            self.port.setValue(502)
        elif not modbus and self.port.value() == 502:
            self.port.setValue(2000)

    def result_value(self):
        return make_link(
            name=self.name.text().strip(), kind=self.kind.currentData(),
            host=self.host.text().strip(), port=int(self.port.value()),
            format=self.format.currentData(),
            encoding=self.encoding.text().strip() or "utf-8",
            terminator=self.terminator.text(),
            timeout=float(self.timeout.value()), unit=int(self.unit.value()))

    def check(self):
        name = self.name.text().strip()
        if not name:
            return ["a machine needs a name — it is what a program line says"]
        if name in self.taken:
            return ["there is already a machine called %r" % name]
        return check_link(self.result_value())


class ItemDialog(_Dialog):
    """One named datum on one machine.

    The fields shown are the protocol's, not a superset with half of them
    greyed out: a Modbus datum is an address and a number, and a TCP datum is
    a string and a way of matching it, and neither has anything to say about
    the other.
    """

    def __init__(self, parent, link, item, taken=()):
        super().__init__(parent, "Data on %s" % link.get("name", "?"))
        self.link = link
        self.taken = set(taken)
        self.modbus = is_modbus(link)

        self.name = self._edit(item.get("name", ""))
        self._row("name", self.name,
                  "what the program line picks off the list — START, DONE, "
                  "READY")
        self.direction = self._pick(DIRECTIONS, item.get("direction", "both"),
                                    DIRECTION_LABEL)
        self.direction.currentIndexChanged.connect(self._direction_changed)
        self._row("way", self.direction,
                  "send  this cell may put it on the wire\n"
                  "recv  this cell may wait for it\n"
                  "both  either")

        if self.modbus:
            self.area = self._pick(AREAS, item.get("area", "holding"),
                                   AREA_LABEL)
            self._row("table", self.area)
            self.register = self._int(0, REGISTER_MAX,
                                      item.get("register", 0) or 0)
            self._row("register", self.register)
            self.value = self._int(0, REGISTER_VALUE_MAX,
                                   _as_int(item.get("value", 0)))
            self._row("value", self.value,
                      "what a SEND writes here, and what a RECV waits for it "
                      "to read")
        else:
            self.value = self._edit(str(item.get("value", "")))
            self._row("payload", self.value,
                      "the text this carries. \\r \\n \\t are typed as they "
                      "read\nOn a hex link it is bytes, written 02 41 03.")
            self.match = self._pick(MATCHES, item.get("match", "exact"))
            self.match_row = self._row(
                "matches", self.match,
                "exact     the whole message is this\n"
                "contains  this appears somewhere in it\n"
                "prefix    the message starts with this\n\n"
                "For machines that append a sequence number nobody asked for.")

        self._direction_changed()
        self._finish()

    def _direction_changed(self):
        if self.modbus:
            return
        # How a message is matched is a question about receiving. A send-only
        # datum showing it would be offering a setting that does nothing.
        self.match_row.setVisible(self.direction.currentData() != "send")

    def result_value(self):
        if self.modbus:
            return make_item(name=self.name.text().strip(),
                             direction=self.direction.currentData(),
                             value=int(self.value.value()),
                             area=self.area.currentData(),
                             register=int(self.register.value()))
        return make_item(name=self.name.text().strip(),
                         direction=self.direction.currentData(),
                         value=self.value.text(),
                         match=self.match.currentData())

    def check(self):
        name = self.name.text().strip()
        if not name:
            return ["a datum needs a name — it is what the program line picks"]
        if name in self.taken:
            return ["%s already carries something called %r"
                    % (self.link.get("name"), name)]
        return check_item(self.link, self.result_value())


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
