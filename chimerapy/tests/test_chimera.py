import datetime
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sunpy.map import Map, all_coordinates_from_map
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from chimerapy.chimera import calculate_area_map, filter_ch, generate_candidate_mask
from chimerapy.chimera import chimera as chimera_new
import chimerapy.chimera_original as chimera_old
import warnings
import time
import matplotlib.pyplot as plt
import urllib.request
import os
import gzip
import traceback
import sunpy.map
import sunpy.sun.constants as constants


def download_file(url, data_dir="data"):
    """Downloads a file from the given URL."""
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, os.path.basename(url))
    try:
        print(f"Downloading {url} to {filename}")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully.")
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


@pytest.fixture()
def p171():
    base_url = "https://jsoc1.stanford.edu/data/aia/synoptic/{year}/{month:02d}/{day:02d}/H0200/AIA{year}{month:02d}{day:02d}_0232_0171.fits"
    start_date = datetime.date(2016, 9, 22)
    file_paths = []
    for i in range(1):
        current_date = start_date + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        url = base_url.format(year=current_date.year, month=current_date.month, day=current_date.day, date=date_str)
        filepath = download_file(url)
        if filepath:
            file_paths.append(filepath)
    return file_paths


@pytest.fixture()
def p193():
    base_url = "https://jsoc1.stanford.edu/data/aia/synoptic/{year}/{month:02d}/{day:02d}/H0200/AIA{year}{month:02d}{day:02d}_0232_0193.fits"
    start_date = datetime.date(2016, 9, 22)
    file_paths = []
    for i in range(1):
        current_date = start_date + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        url = base_url.format(year=current_date.year, month=current_date.month, day=current_date.day, date=date_str)
        filepath = download_file(url)
        if filepath:
            file_paths.append(filepath)
    return file_paths


@pytest.fixture()
def p211():
    base_url = "https://jsoc1.stanford.edu/data/aia/synoptic/{year}/{month:02d}/{day:02d}/H0200/AIA{year}{month:02d}{day:02d}_0232_0211.fits"
    start_date = datetime.date(2016, 9, 22)
    file_paths = []
    for i in range(1):
        current_date = start_date + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        url = base_url.format(year=current_date.year, month=current_date.month, day=current_date.day, date=date_str)
        filepath = download_file(url)
        if filepath:
            file_paths.append(filepath)
    return file_paths


@pytest.fixture(scope="module")
def m171():
    return Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")


@pytest.fixture(scope="module")
def m193():
    return Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")


@pytest.fixture(scope="module")
def m211():
    return Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")


@pytest.fixture(scope="module")
def dummy_hmi(m171):
    """Create a dummy HMI map for the original CHIMERA code."""
    dummy_hmi = Map(np.zeros_like(m171.data), m171.meta)
    return dummy_hmi

def extract_coronal_hole_properties(coronal_holes):
    """Extract relevant properties from the coronal hole dictionaries."""
    properties = []
    for ch in coronal_holes:
        properties.append({
            "area_meters2": ch["area_meters2"].to_value(u.Mm**2),
            "centroid_lon": ch["centroid_heliographic"].lon.to_value(u.deg),
            "centroid_lat": ch["centroid_heliographic"].lat.to_value(u.deg),
            "extent_lon": ch["extent_lon"].to_value(u.deg),
            "extent_lat": ch["extent_lat"].to_value(u.deg),
        })
    return properties


def test_generate_candidate_mask(m171, m193, m211):
    from examples.paper_figures import mask_map

    result_mask = generate_candidate_mask(m171, m193, m211)

    expected_shape = m171.data.shape
    assert result_mask.shape == expected_shape, "Mask shape does not match expected shape."

    expected_mask = mask_map.data.astype(bool)
    np.testing.assert_allclose(result_mask, expected_mask)


def test_calculate_area_map(m171):
    area_map, disk_mask = calculate_area_map(m171)
    total_area = area_map.sum()
    hemi_sphere_area = 2 * np.pi * m171.rsun_meters**2
    assert_allclose(total_area, hemi_sphere_area, rtol=5e-4)  # 0.05% seems pretty ok?


@pytest.mark.parametrize("theta", ([5, 10, 15, 45, 75, 90] * u.deg))
def test_filter_by_area_size(m171, theta):
    hpc_coords = all_coordinates_from_map(m171)
    hgs_coords = hpc_coords.transform_to("heliographic_stonyhurst")
    ref = m171.reference_coordinate
    radial_angle = hgs_coords.separation(ref)
    data = np.zeros_like(m171.data)
    data[radial_angle <= theta] = 1
    m171.data[:, :] = data
    label_mask, regions = filter_ch(data, m171, min_area=0 * u.m**2)
    expected_area = 2 * np.pi * (1 - np.cos(theta)) * m171.rsun_meters**2
    assert_allclose(regions[0].surface_area, expected_area, rtol=0.01)


@pytest.mark.parametrize("pos", ([5, 10, 15, 45, 75, 90] * u.deg))
def test_filter_by_area_position(m171, pos):
    rtol = 0.01
    theta = 10 * u.deg
    hpc_coords = all_coordinates_from_map(m171)
    hgs_coords = hpc_coords.transform_to("heliographic_stonyhurst")
    ref = m171.reference_coordinate
    center = ref.transform_to("heliographic_stonyhurst").spherical_offsets_by(0 * u.deg, pos)
    radial_angle = hgs_coords.separation(center)
    data = np.zeros_like(m171.data)
    data[radial_angle <= theta] = 1
    m171.data[:, :] = data
    label_mask, regions = filter_ch(data, m171, min_area=0 * u.m**2)
    expected_area = 2 * np.pi * (1 - np.cos(theta)) * m171.rsun_meters**2
    if u.allclose(pos, 90 * u.deg):
        expected_area *= 0.5  # half behind the limb
        rtol = 0.10  # more error as area per pixel is huge
    assert_quantity_allclose(regions[0].surface_area, expected_area, rtol=rtol)


def chimera_over_time(m171, m193, m211, dummy_hmi):
    all_results = None
    print("Starting chimera_over_time")

    try:
        print("Running chimera_new and chimera_old")
        print(f"Type of dummy_hmi in chimera_over_time: {type(dummy_hmi)}")
        _, mask_new, coronal_holes_new = chimera_new(m171, m193, m211)
        _, mask_old, coronal_holes_old = chimera_old.chimera(m171, m193, m211, dummy_hmi)

        properties_new = extract_coronal_hole_properties(coronal_holes_new)
        properties_old = extract_coronal_hole_properties(coronal_holes_old)

        all_results = {
            "mask_new": mask_new,
            "mask_old": mask_old,
            "properties_new": properties_new,
            "properties_old": properties_old,
            "files": [m171.meta['filename'], m193.meta['filename'], m211.meta['filename']]
        }
        print("CHIMERA analysis completed successfully")
    except Exception as e:
        print(f"Error running CHIMERA analysis: {e}")
        traceback.print_exc()

    return all_results


def plot_chimera_comparison(all_results):
    """Generates plots to compare CHIMERA versions."""
    for i in range(len(all_results)):
        date = all_results[i]["date"]
        mask_new = all_results[i]["mask_new"]
        mask_old = all_results[i]["mask_old"]
        files = all_results[i]["files"]

        if mask_new is None or mask_old is None:
            print(f"No plots possible for {date} because of failed download.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mask_new, origin="lower", cmap="gray")
        axes[0].set_title(f"New CHIMERA - {date}")
        axes[1].imshow(mask_old, origin="lower", cmap="gray")
        axes[1].set_title(f"Old CHIMERA - {date}")
        fig.suptitle(f"CHIMERA Comparison - {date}")
        plt.show()

        diff_mask = mask_new.astype(int) - mask_old.astype(int)
        plt.imshow(diff_mask, origin="lower", cmap="RdBu", vmin=-1, vmax=1)
        plt.title(f"Difference (New - Old) - {date}")
        plt.colorbar()
        plt.show()

        properties_new = all_results[i]["properties_new"]
        properties_old = all_results[i]["properties_old"]

        if properties_new and properties_old:
            areas_new = [ch["area_meters2"] for ch in properties_new]
            areas_old = [ch["area_meters2"] for ch in properties_old]
            plt.plot(areas_new, label="New CHIMERA")
            plt.plot(areas_old, label="Old CHIMERA")
            plt.xlabel("Coronal Holes")
            plt.ylabel("Area (m^2)")
            plt.title(f"Area Comparison - {date}")
            plt.legend()
            plt.show()
        else:
            print(f"Skipping area comparison plot for {date} because coronal hole properties are missing.")

def test_chimera_versions(m171, m193, m211, dummy_hmi):
    try:
        print(f"Type of dummy_hmi: {type(dummy_hmi)}")
        all_results = chimera_over_time(m171, m193, m211, dummy_hmi)
        assert all_results is not None
    except Exception as e:
        print(f"Test failed with error: {e}")
        traceback.print_exc()
        raise