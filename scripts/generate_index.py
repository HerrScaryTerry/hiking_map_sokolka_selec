from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# Configuration
# ============================================================

ROOT_FOLDER = Path(__file__).resolve().parent.parent
GPX_FOLDER = ROOT_FOLDER / "gpx"
OUTPUT_FILE = ROOT_FOLDER / "hikes.json"

# Route simplification tolerance in metres.
#
# Statistics are calculated from all original GPX points.
# Only coordinates displayed on the map are simplified.
#
# Recommended:
#   0  = no simplification
#   5  = very detailed
#   8  = recommended for hiking routes
#   15 = smaller JSON, less route detail
SIMPLIFICATION_TOLERANCE_METERS = 8.0

# Coordinates in JSON are rounded to this number of decimals.
# Five decimals represent approximately one metre.
COORDINATE_DECIMALS = 5

EARTH_RADIUS_METERS = 6_371_000.0


# ============================================================
# General helpers
# ============================================================

def clean_text(value: Optional[str]) -> Optional[str]:
    """
    Trim text and convert empty strings to None.
    """

    if value is None:
        return None

    cleaned_value = value.strip()

    return cleaned_value if cleaned_value else None


def find_first_text(
    root: ET.Element,
    paths: list[str]
) -> Optional[str]:
    """
    Return text from the first matching XML element.

    The {*} syntax ignores the concrete XML namespace.
    """

    for path in paths:

        element = root.find(path)

        if element is None:
            continue

        value = clean_text(element.text)

        if value:
            return value

    return None


def parse_datetime(
    value: Optional[str]
) -> Optional[datetime]:
    """
    Parse an ISO 8601 timestamp from Garmin GPX.

    Examples:
        2026-08-08T06:51:24.000Z
        2026-08-08T06:51:24Z
        2026-08-08T06:51:24+00:00
    """

    if not value:
        return None

    normalized_value = value.strip()

    if normalized_value.endswith("Z"):
        normalized_value = (
            normalized_value[:-1]
            + "+00:00"
        )

    try:
        return datetime.fromisoformat(
            normalized_value
        )

    except ValueError:
        return None


# ============================================================
# Distance calculation
# ============================================================

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Returns metres.
    """

    latitude_1 = math.radians(lat1)
    latitude_2 = math.radians(lat2)

    latitude_difference = math.radians(
        lat2 - lat1
    )

    longitude_difference = math.radians(
        lon2 - lon1
    )

    haversine_value = (
        math.sin(latitude_difference / 2) ** 2
        +
        math.cos(latitude_1)
        *
        math.cos(latitude_2)
        *
        math.sin(longitude_difference / 2) ** 2
    )

    angular_distance = 2 * math.atan2(
        math.sqrt(haversine_value),
        math.sqrt(1 - haversine_value)
    )

    return (
        EARTH_RADIUS_METERS
        *
        angular_distance
    )


# ============================================================
# Route simplification
# ============================================================

def convert_to_local_metres(
    latitude: float,
    longitude: float,
    reference_latitude: float
) -> tuple[float, float]:
    """
    Convert latitude and longitude approximately to local metres.

    This is sufficient for route simplification over hiking
    distances.
    """

    x = (
        math.radians(longitude)
        *
        EARTH_RADIUS_METERS
        *
        math.cos(
            math.radians(reference_latitude)
        )
    )

    y = (
        math.radians(latitude)
        *
        EARTH_RADIUS_METERS
    )

    return x, y


def perpendicular_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float]
) -> float:
    """
    Calculate perpendicular distance from a coordinate to a line.

    Returns metres.
    """

    reference_latitude = (
        point[0]
        +
        line_start[0]
        +
        line_end[0]
    ) / 3

    point_x, point_y = convert_to_local_metres(
        point[0],
        point[1],
        reference_latitude
    )

    start_x, start_y = convert_to_local_metres(
        line_start[0],
        line_start[1],
        reference_latitude
    )

    end_x, end_y = convert_to_local_metres(
        line_end[0],
        line_end[1],
        reference_latitude
    )

    line_dx = end_x - start_x
    line_dy = end_y - start_y

    line_length_squared = (
        line_dx ** 2
        +
        line_dy ** 2
    )

    if line_length_squared == 0:

        return math.hypot(
            point_x - start_x,
            point_y - start_y
        )

    projection = (
        (
            (point_x - start_x) * line_dx
            +
            (point_y - start_y) * line_dy
        )
        /
        line_length_squared
    )

    projection = max(
        0.0,
        min(1.0, projection)
    )

    closest_x = (
        start_x
        +
        projection * line_dx
    )

    closest_y = (
        start_y
        +
        projection * line_dy
    )

    return math.hypot(
        point_x - closest_x,
        point_y - closest_y
    )


def simplify_coordinates(
    coordinates: list[tuple[float, float]],
    tolerance_metres: float
) -> list[tuple[float, float]]:
    """
    Simplify a route using the Douglas-Peucker algorithm.
    """

    if (
        len(coordinates) <= 2
        or tolerance_metres <= 0
    ):
        return coordinates

    first_point = coordinates[0]
    last_point = coordinates[-1]

    maximum_distance = 0.0
    maximum_index = 0

    for index in range(
        1,
        len(coordinates) - 1
    ):

        distance = perpendicular_distance(
            coordinates[index],
            first_point,
            last_point
        )

        if distance > maximum_distance:

            maximum_distance = distance
            maximum_index = index

    if maximum_distance > tolerance_metres:

        first_section = simplify_coordinates(
            coordinates[
                :maximum_index + 1
            ],
            tolerance_metres
        )

        second_section = simplify_coordinates(
            coordinates[
                maximum_index:
            ],
            tolerance_metres
        )

        return (
            first_section[:-1]
            +
            second_section
        )

    return [
        first_point,
        last_point
    ]


# ============================================================
# GPX metadata
# ============================================================

def get_activity_name(
    root: ET.Element,
    file_path: Path
) -> str:
    """
    Determine the activity name.

    Priority:
        1. Track name
        2. Route name
        3. Metadata name
        4. Filename
    """

    name = find_first_text(
        root,
        [
            "./{*}trk/{*}name",
            "./{*}rte/{*}name",
            "./{*}metadata/{*}name"
        ]
    )

    return name or file_path.stem


def get_activity_type(
    root: ET.Element
) -> Optional[str]:
    """
    Read the activity type from the GPX track.
    """

    return find_first_text(
        root,
        [
            "./{*}trk/{*}type",
            "./{*}rte/{*}type"
        ]
    )


def get_metadata_datetime(
    root: ET.Element
) -> Optional[datetime]:
    """
    Read the first available activity timestamp.

    Priority:
        1. Metadata time
        2. Track time
        3. First trackpoint time
        4. First route-point time
    """

    timestamp_text = find_first_text(
        root,
        [
            "./{*}metadata/{*}time",
            "./{*}trk/{*}time",
            ".//{*}trkpt/{*}time",
            ".//{*}rtept/{*}time"
        ]
    )

    return parse_datetime(timestamp_text)


# ============================================================
# GPX trackpoint extraction
# ============================================================

def read_trackpoints(
    root: ET.Element
) -> list[dict]:
    """
    Read trackpoints from all Garmin GPX track segments.
    """

    points: list[dict] = []

    trackpoint_elements = root.findall(
        ".//{*}trkpt"
    )

    if not trackpoint_elements:

        trackpoint_elements = root.findall(
            ".//{*}rtept"
        )

    for element in trackpoint_elements:

        latitude_text = element.get("lat")
        longitude_text = element.get("lon")

        if (
            latitude_text is None
            or longitude_text is None
        ):
            continue

        try:

            latitude = float(latitude_text)
            longitude = float(longitude_text)

        except ValueError:
            continue

        elevation_element = element.find(
            "./{*}ele"
        )

        time_element = element.find(
            "./{*}time"
        )

        elevation = None
        timestamp = None

        if (
            elevation_element is not None
            and elevation_element.text
        ):

            try:
                elevation = float(
                    elevation_element.text
                )

            except ValueError:
                elevation = None

        if (
            time_element is not None
            and time_element.text
        ):

            timestamp = parse_datetime(
                time_element.text
            )

        points.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "elevation": elevation,
                "timestamp": timestamp
            }
        )

    return points


# ============================================================
# Activity statistics
# ============================================================

def calculate_statistics(
    points: list[dict]
) -> dict:
    """
    Calculate statistics from the original GPX points.

    Distance:
        Haversine distance between consecutive points.

    Elevation:
        Sum of positive/negative differences between consecutive
        points with elevation data.

    Duration:
        Difference between the first and final valid trackpoint
        timestamp. This represents elapsed activity time.
    """

    total_distance_metres = 0.0
    total_elevation_gain_metres = 0.0
    total_elevation_loss_metres = 0.0

    for index in range(
        1,
        len(points)
    ):

        previous_point = points[index - 1]
        current_point = points[index]

        total_distance_metres += haversine_distance(
            previous_point["latitude"],
            previous_point["longitude"],
            current_point["latitude"],
            current_point["longitude"]
        )

        previous_elevation = (
            previous_point["elevation"]
        )

        current_elevation = (
            current_point["elevation"]
        )

        if (
            previous_elevation is not None
            and current_elevation is not None
        ):

            elevation_difference = (
                current_elevation
                -
                previous_elevation
            )

            if elevation_difference > 0:

                total_elevation_gain_metres += (
                    elevation_difference
                )

            elif elevation_difference < 0:

                total_elevation_loss_metres += abs(
                    elevation_difference
                )

    timestamps = [
        point["timestamp"]
        for point in points
        if point["timestamp"] is not None
    ]

    duration_seconds = 0

    if len(timestamps) >= 2:

        duration_seconds = max(
            0,
            round(
                (
                    timestamps[-1]
                    -
                    timestamps[0]
                ).total_seconds()
            )
        )

    elevations = [
        point["elevation"]
        for point in points
        if point["elevation"] is not None
    ]

    minimum_elevation = (
        min(elevations)
        if elevations
        else None
    )

    maximum_elevation = (
        max(elevations)
        if elevations
        else None
    )

    return {
        "distanceKilometers": round(
            total_distance_metres / 1000,
            2
        ),
        "elevationGainMeters": round(
            total_elevation_gain_metres
        ),
        "elevationLossMeters": round(
            total_elevation_loss_metres
        ),
        "durationSeconds": duration_seconds,
        "minimumElevationMeters": (
            round(minimum_elevation)
            if minimum_elevation is not None
            else None
        ),
        "maximumElevationMeters": (
            round(maximum_elevation)
            if maximum_elevation is not None
            else None
        )
    }


# ============================================================
# Parse one GPX file
# ============================================================

def parse_gpx_file(
    file_path: Path
) -> dict:
    """
    Parse one GPX file into the final website data structure.
    """

    xml_tree = ET.parse(file_path)
    root = xml_tree.getroot()

    trackpoints = read_trackpoints(root)

    if len(trackpoints) < 2:

        raise ValueError(
            "GPX does not contain at least two valid trackpoints."
        )

    activity_datetime = get_metadata_datetime(
        root
    )

    if activity_datetime is None:

        timestamps = [
            point["timestamp"]
            for point in trackpoints
            if point["timestamp"] is not None
        ]

        if timestamps:
            activity_datetime = timestamps[0]

    activity_name = get_activity_name(
        root,
        file_path
    )

    activity_type = get_activity_type(
        root
    )

    statistics = calculate_statistics(
        trackpoints
    )

    original_coordinates = [
        (
            point["latitude"],
            point["longitude"]
        )
        for point in trackpoints
    ]

    simplified_coordinates = simplify_coordinates(
        original_coordinates,
        SIMPLIFICATION_TOLERANCE_METERS
    )

    output_coordinates = [
        [
            round(
                latitude,
                COORDINATE_DECIMALS
            ),
            round(
                longitude,
                COORDINATE_DECIMALS
            )
        ]
        for latitude, longitude
        in simplified_coordinates
    ]

    if activity_datetime is not None:

        activity_date = (
            activity_datetime
            .date()
            .isoformat()
        )

        activity_year = (
            activity_datetime.year
        )

        activity_timestamp = (
            activity_datetime.isoformat()
        )

    else:

        activity_date = None
        activity_year = None
        activity_timestamp = None

    return {
        "file": file_path.name,
        "name": activity_name,
        "date": activity_date,
        "year": activity_year,
        "timestamp": activity_timestamp,
        "type": activity_type,
        "distanceKilometers":
            statistics["distanceKilometers"],
        "elevationGainMeters":
            statistics["elevationGainMeters"],
        "elevationLossMeters":
            statistics["elevationLossMeters"],
        "durationSeconds":
            statistics["durationSeconds"],
        "minimumElevationMeters":
            statistics["minimumElevationMeters"],
        "maximumElevationMeters":
            statistics["maximumElevationMeters"],
        "originalPointCount":
            len(original_coordinates),
        "displayPointCount":
            len(output_coordinates),
        "coordinates":
            output_coordinates
    }


# ============================================================
# Main process
# ============================================================

def main() -> None:
    """
    Parse all GPX files and generate hikes.json.
    """

    if not GPX_FOLDER.exists():

        raise FileNotFoundError(
            f"GPX folder does not exist: {GPX_FOLDER}"
        )

    gpx_files = sorted(
        GPX_FOLDER.glob("*.gpx")
    )

    activities: list[dict] = []
    errors: list[dict] = []

    print(
        f"Found {len(gpx_files)} GPX file(s)."
    )

    for file_path in gpx_files:

        try:

            activity = parse_gpx_file(
                file_path
            )

            activities.append(activity)

            print(
                "Processed: "
                f"{file_path.name} -> "
                f"{activity['name']} | "
                f"{activity['date']} | "
                f"{activity['distanceKilometers']} km | "
                f"{activity['originalPointCount']} -> "
                f"{activity['displayPointCount']} points"
            )

        except ET.ParseError as error:

            errors.append(
                {
                    "file": file_path.name,
                    "error": (
                        "Invalid GPX XML: "
                        f"{error}"
                    )
                }
            )

            print(
                f"ERROR: Invalid XML in "
                f"{file_path.name}: {error}"
            )

        except Exception as error:

            errors.append(
                {
                    "file": file_path.name,
                    "error": str(error)
                }
            )

            print(
                f"ERROR: Could not process "
                f"{file_path.name}: {error}"
            )

    activities.sort(
        key=lambda activity: (
            activity["date"] is not None,
            activity["date"] or "",
            activity["name"].lower()
        ),
        reverse=True
    )

    years = sorted(
        {
            activity["year"]
            for activity in activities
            if activity["year"] is not None
        },
        reverse=True
    )

    output_data = {
        "activities": activities,
        "years": years,
        "errors": errors
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        json.dump(
            output_data,
            output_file,
            ensure_ascii=False,
            separators=(",", ":")
        )

    file_size_megabytes = (
        OUTPUT_FILE.stat().st_size
        /
        1024
        /
        1024
    )

    print()
    print(f"Generated: {OUTPUT_FILE}")
    print(f"Activities: {len(activities)}")
    print(f"Years: {years}")
    print(f"Errors: {len(errors)}")
    print(
        "hikes.json size: "
        f"{file_size_megabytes:.2f} MB"
    )


if __name__ == "__main__":
    main()
