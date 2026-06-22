#!/usr/bin/env python3
"""Write a styled XLSX workbook through LibreOffice UNO.

Run this file with /usr/bin/python3 because the system Python provides the
LibreOffice UNO bridge on this machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


BLUE = 0x1F4E78
LIGHT_BLUE = 0xDCE6F1
PALE_BLUE = 0xF4F8FC
GREEN = 0xE2F0D9
YELLOW = 0xFFF2CC
GRAY = 0xE7E6E6
RED = 0xFCE4D6
WHITE = 0xFFFFFF
DARK = 0x203864


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def connect():
    process = subprocess.Popen(
        [
            "libreoffice",
            "--headless",
            "--accept=socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext",
            "--nologo",
            "--nodefault",
            "--nofirststartwizard",
            "--norestore",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    for _ in range(50):
        try:
            ctx = resolver.resolve(
                "uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext"
            )
            return process, ctx
        except Exception:
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("Could not connect to LibreOffice UNO listener")


def clean_sheet_name(value):
    return "".join("_" if ch in r'[]:*?/\\' else ch for ch in value)[:31]


def write_sheet(doc, sheet, spec, index):
    title = spec["title"]
    subtitle = spec.get("subtitle", "")
    headers = spec["headers"]
    rows = spec["rows"]
    widths = spec.get("widths", {})
    sheet.Name = clean_sheet_name(spec["name"])
    sheet.TabColor = spec.get("tab_color", BLUE)

    last_col = max(len(headers) - 1, 0)
    last_row = 4 + len(rows)
    sheet.getCellRangeByPosition(0, 0, last_col, 0).merge(True)
    sheet.getCellByPosition(0, 0).String = title
    sheet.getCellRangeByPosition(0, 1, last_col, 1).merge(True)
    sheet.getCellByPosition(0, 1).String = subtitle

    title_range = sheet.getCellRangeByPosition(0, 0, last_col, 0)
    title_range.CharWeight = 150.0
    title_range.CharHeight = 15.0
    title_range.CharColor = DARK
    subtitle_range = sheet.getCellRangeByPosition(0, 1, last_col, 1)
    subtitle_range.CharColor = 0x666666
    subtitle_range.CharHeight = 10.0

    for col, header in enumerate(headers):
        cell = sheet.getCellByPosition(col, 3)
        cell.String = str(header)
    header_range = sheet.getCellRangeByPosition(0, 3, last_col, 3)
    header_range.CellBackColor = BLUE
    header_range.CharColor = WHITE
    header_range.CharWeight = 150.0
    header_range.IsTextWrapped = True

    for row_offset, values in enumerate(rows, 4):
        for col, value in enumerate(values):
            cell = sheet.getCellByPosition(col, row_offset)
            if isinstance(value, bool):
                cell.Value = int(value)
            elif isinstance(value, (int, float)):
                cell.Value = float(value)
            else:
                cell.String = "" if value is None else str(value)
        body = sheet.getCellRangeByPosition(0, row_offset, last_col, row_offset)
        body.CellBackColor = WHITE if row_offset % 2 == 0 else PALE_BLUE
        body.IsTextWrapped = True
        sheet.getRows().getByIndex(row_offset).Height = spec.get("row_height", 620)

    status_col = headers.index("状态") if "状态" in headers else -1
    priority_col = headers.index("优先级") if "优先级" in headers else -1
    rank_col = headers.index("排名") if "排名" in headers else -1
    for row_offset in range(4, last_row):
        if status_col >= 0:
            cell = sheet.getCellByPosition(status_col, row_offset)
            text = cell.String
            if "已完成" in text:
                cell.CellBackColor = GREEN
            elif "训练中" in text:
                cell.CellBackColor = YELLOW
            elif "未完成" in text:
                cell.CellBackColor = RED
            elif "历史" in text or "归档" in text:
                cell.CellBackColor = GRAY
        if priority_col >= 0:
            cell = sheet.getCellByPosition(priority_col, row_offset)
            cell.CellBackColor = RED if cell.String == "P0" else YELLOW
            cell.CharWeight = 150.0
        if rank_col >= 0 and sheet.getCellByPosition(rank_col, row_offset).Value <= 3:
            sheet.getCellByPosition(rank_col, row_offset).CellBackColor = GREEN
            sheet.getCellByPosition(rank_col, row_offset).CharWeight = 150.0

    for col, header in enumerate(headers):
        width = widths.get(header, 14)
        sheet.getColumns().getByIndex(col).Width = int(max(8, min(width, 60)) * 250)
    sheet.getRows().getByIndex(0).Height = 750
    sheet.getRows().getByIndex(1).Height = 500
    sheet.getRows().getByIndex(3).Height = 750

    controller = doc.getCurrentController()
    controller.setActiveSheet(sheet)
    controller.freezeAtPosition(min(2, len(headers)), 4)

    if rows:
        ranges = doc.DatabaseRanges
        database_name = f"filter_{index}"
        ranges.addNewByName(
            database_name, sheet.getCellRangeByPosition(0, 3, last_col, last_row - 1).getRangeAddress()
        )
        ranges.getByName(database_name).AutoFilter = True


def main():
    payload_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    process, ctx = connect()
    try:
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL("private:factory/scalc", "_blank", 0, ())
        sheets = doc.getSheets()
        for index, spec in enumerate(payload["sheets"]):
            if index == 0:
                sheet = sheets.getByIndex(0)
            else:
                sheets.insertNewByName(f"Sheet{index + 1}", index)
                sheet = sheets.getByIndex(index)
            write_sheet(doc, sheet, spec, index)
        doc.getCurrentController().setActiveSheet(sheets.getByIndex(0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        doc.storeAsURL(
            output_path.as_uri(),
            (
                prop("FilterName", "Calc MS Excel 2007 XML"),
                prop("Overwrite", True),
            ),
        )
        doc.close(True)
    finally:
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
