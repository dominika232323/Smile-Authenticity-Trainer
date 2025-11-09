import pytest

from ai.data_preprocessing.normalize_face import calculate_distance_between_points


@pytest.mark.parametrize(
    ("point1", "point2", "expected"),
    [
        ((0.0, 0.0), (1.0, 0.0), 1.0),
        ((0.0, 0.0), (0.0, 1.0), 1.0),
        ((1.0, 1.0), (4.0, 5.0), 5.0),
        ((-1.0, -1.0), (2.0, 3.0), 5.0),
        ((1.5, 2.5), (4.5, 6.5), 5.0),
        ((1000.0, 2000.0), (1003.0, 2004.0), 5.0),
    ],
)
def test_distance(point1, point2, expected):
    result = calculate_distance_between_points(point1, point2)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "point",
    [(1.0, 1.0), (-4.2, -3.2), (-1.0, -1.5), (3.0, 4.0), (2.54, 6.0), (16.44, 33.81)],
)
def test_distance_between_same_points(point):
    result = calculate_distance_between_points(point, point)

    assert result == 0


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((1.0, 1.0), 1.41421356237),
        ((-4.2, -3.2), 5.28015151298),
        ((3.0, 4.0), 5.0),
        ((8.0, 6.0), 10.0),
        ((5.0, 12.0), 13.0),
    ],
)
def test_distance_between_origin_and_point(point, expected):
    origin = (0.0, 0.0)
    result = calculate_distance_between_points(point, origin)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("point1", "point2", "expected"),
    [
        ((1.0, 5.0), (6.0, 5.0), 5.0),
        ((0.0, 1.0), (1.0, 1.0), 1.0),
        ((1.0, 1.0), (4.0, 1.0), 3.0),
        ((-1.0, -1.0), (2.0, -1.0), 3.0),
        ((-1000.0, 2000.0), (1003.0, 2000.0), 2003.0),
    ],
)
def test_distance_horizontal_points(point1, point2, expected):
    result = calculate_distance_between_points(point1, point2)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("point1", "point2", "expected"),
    [
        ((3.0, 2.0), (3.0, 8.0), 6.0),
        ((0.0, 0.0), (0.0, 1.0), 1.0),
        ((4.0, 1.0), (4.0, 5.0), 4.0),
        ((-1.0, -1.0), (-1.0, 4.0), 5.0),
        ((1000.0, -2000.0), (1000.0, 2004.0), 4004.0),
    ],
)
def test_distance_vertical_points(point1, point2, expected):
    result = calculate_distance_between_points(point1, point2)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("point1", "point2"),
    [
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (0.0, 1.0)),
        ((1.0, 1.0), (4.0, 5.0)),
        ((-1.0, -1.0), (2.0, 3.0)),
        ((1.5, 2.5), (4.5, 6.5)),
        ((1000.0, 2000.0), (1003.0, 2004.0)),
    ],
)
def test_distance_commutative_property(point1, point2):
    distance1 = calculate_distance_between_points(point1, point2)
    distance2 = calculate_distance_between_points(point2, point1)

    assert distance1 == pytest.approx(distance2)
