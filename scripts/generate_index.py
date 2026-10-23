from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT_FOLDER = Path(__file__).resolve().parent.parent
GPX_FOLDER = ROOT_FOLDER / "gpx"
OUTPUT_FILE = ROOT_FOLDER / "hikes.json"


def clean_text(value: Optional[str]) -> Optional[str]:
    """
    Remove surrounding whitespace and return None for empty values.
    """

    if value is None:
        return None

    value = value.strip()

    return value if value else None


def find_first_text(
    root: ET.Element,
    xpath: str
) -> Optional[str]:
    """
    Find the first matching XML element while ignoring GPX namespaces.

    Example:
        find_first_text(root, ".//{*}trk/{*}name")
    """

    element = root.find(xpath)

    if element is None:
        return None

    return clean_text(element.text)


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Parse a GPX ISO 8601 timestamp.

    Supports timestamps such as:
        2026-08-08T06:51:24.000Z
        2026-08-08T06:51:24Z
        2026-08-08T06:51:24+00:00
    """

    if not value:
        return None

    normalized_value = value.strip()

    if normalized_value.endswith("Z"):
        normalized_value = normalized_value[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized_value)
    except ValueError:
        return None


def get_activity_datetime(
    root: ET.Element
) -> Optional[datetime]:
    """
    Determine the activity date using the following priority:

    1. GPX metadata/time
    2. Track-level time, if present
    3. First track point time
    4. First route point time
    5. First waypoint time
    """

    possible_time_paths = [
        "./{*}metadata/{*}time",
        "./{*}trk/{*}time",
        ".//{*}trkpt/{*}time",
        ".//{*}rtept/{*}time",
        ".//{*}wpt/{*}time"
    ]

    for xpath in possible_time_paths:

        time_text = find_first_text(
            root,
            xpath
        )

        parsed_datetime = parse_datetime(
            time_text
        )

        if parsed_datetime is not None:
            return parsed_datetime

    return None


def get_activity_name(
    root: ET.Element,
    file: Path
) -> str:
    """
    Determine activity name using the following priority:

    1. Track name
    2. Route name
    3. GPX metadata name
    4. Filename without extension
    """

    possible_name_paths = [
        "./{*}trk/{*}name",
        "./{*}rte/{*}name",
        "./{*}metadata/{*}name"
    ]

    for xpath in possible_name_paths:

        name = find_first_text(
            root,
            xpath
        )

        if name:
            return name

    return file.stem


def get_activity_type(
    root: ET.Element
) -> Optional[str]:
    """
    Read the activity type from the GPX track.
    """

    return find_first_text(
        root,
        "./{*}trk/{*}type"
    )


def parse_gpx_file(
    file: Path
) -> dict:
    """
    Parse one GPX file and return normalized activity metadata.
    """

    xml_tree = ET.parse(file)
    root = xml_tree.getroot()

    activity_name = get_activity_name(
        root,
        file
    )

    activity_datetime = get_activity_datetime(
        root
    )

    activity_type = get_activity_type(
        root
    )

    if activity_datetime is not None:

        activity_date = activity_datetime.date().isoformat()
        activity_year = activity_datetime.year
        activity_timestamp = activity_datetime.isoformat()

    else:

        activity_date = None
        activity_year = None
        activity_timestamp = None

    return {
        "file": file.name,
        "name": activity_name,
        "date": activity_date,
        "year": activity_year,
        "timestamp": activity_timestamp,
        "type": activity_type
    }


def main() -> None:

    if not GPX_FOLDER.exists():

        raise FileNotFoundError(
            f"GPX folder does not exist: {GPX_FOLDER}"
        )

    gpx_files = sorted(
        GPX_FOLDER.glob("*.gpx")
    )

    activities = []
    errors = []

    for file in gpx_files:

        try:

            activity = parse_gpx_file(
                file
            )

            activities.append(
                activity
            )

            print(
                f"Processed: {file.name} "
                f"-> {activity['name']} "
                f"({activity['date']})"
            )

        except ET.ParseError as error:

            errors.append(
                {
                    "file": file.name,
                    "error": f"Invalid XML: {error}"
                }
            )

            print(
                f"ERROR: Invalid GPX XML in {file.name}: {error}"
            )

        except Exception as error:

            errors.append(
                {
                    "file": file.name,
                    "error": str(error)
                }
            )

            print(
                f"ERROR: Could not process {file.name}: {error}"
            )

    # Newest activities first.
    # Activities without a date are placed last.

    activities.sort(
        key=lambda item: (
            item["date"] is not None,
            item["date"] or "",
            item["name"].lower()
        ),
        reverse=True
    )

    output_data = {
        "activities": activities,
        "years": sorted(
            {
                activity["year"]
                for activity in activities
                if activity["year"] is not None
            },
            reverse=True
        ),
        "errors": errors
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        json.dump(
            output_data,
            output,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"Generated: {OUTPUT_FILE}")
    print(f"Activities: {len(activities)}")
    print(f"Years: {output_data['years']}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
