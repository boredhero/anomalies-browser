"""Coordinate reference system helpers."""


def utm_zone_from_lon(lon: float) -> int:
    """Get UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1


def epsg_from_lonlat(lon: float, lat: float) -> int:
    """Get EPSG code for UTM zone at a given lon/lat.

    Returns EPSG for WGS84 UTM North (326xx) or South (327xx).
    """
    zone = utm_zone_from_lon(lon)
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone
