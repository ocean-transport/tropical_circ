import ast
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm


NOTEBOOK = Path(__file__).resolve().parents[1] / "u_model_comp.ipynb"
CELLS = {
    cell["id"]: "".join(cell["source"])
    for cell in json.loads(NOTEBOOK.read_text())["cells"]
}


def notebook_namespace():
    namespace = {"np": np, "xr": xr}
    exec(CELLS["752b2dfa"], namespace)
    for cell_id in ("706f468b", "aaaa68db"):
        functions = [
            node
            for node in ast.parse(CELLS[cell_id]).body
            if isinstance(node, ast.FunctionDef)
        ]
        exec(compile(ast.Module(body=functions, type_ignores=[]), cell_id, "exec"), namespace)
    return namespace


def sample_dataset(curvilinear=False, descending=False, calendar="standard"):
    latitude = np.array([-40, -20, 0, 9, 10, 20, 40], dtype=float)
    if descending:
        latitude = latitude[::-1]
    longitude = np.array([-180, -159, -120, -104.25, -80, -40], dtype=float)
    time = xr.date_range(
        "1999-01-01", periods=14, freq="YS", calendar=calendar,
        use_cftime=calendar != "standard",
    )
    shape = (time.size, 3, latitude.size, longitude.size)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) / 1000
    values[2, 1, :, 2] = np.nan
    coords = {"time": time, "lev": [5000.0, 100000.0, 400000.0]}
    if curvilinear:
        lon, lat = np.meshgrid(longitude, latitude)
        lat += np.array([0, 0.1, -0.1, 0, 0.1, 0])
        coords.update(lat=(("y", "x"), lat), lon=(("y", "x"), lon))
    else:
        coords.update(lat=("y", latitude), lon=("x", longitude))
    dataset = xr.Dataset(
        {"uo": (("time", "lev", "y", "x"), da.from_array(values, chunks=(2, 1, 1, 2)))},
        coords=coords,
    )
    dataset.lev.attrs["units"] = "cm"
    dataset.uo.attrs["units"] = "m s-1"
    return dataset


def original_section(data, namespace):
    latitude = data.lat
    longitude = data.lon % 360
    region = (
        (latitude >= namespace["LATITUDE_MIN"])
        & (latitude <= namespace["LATITUDE_MAX"])
        & (longitude >= namespace["LONGITUDE_MIN_360"])
        & (longitude <= namespace["LONGITUDE_MAX_360"])
    )
    data = data.where(region.compute(), drop=True)
    latitude = data.lat
    if latitude.ndim == 1:
        target = float(latitude.where(latitude <= 9.5, drop=True).max().item())
        index = int(np.argmin(np.abs(latitude.values - target)))
    else:
        row_mean = latitude.mean("x", skipna=True)
        target = float(row_mean.where(row_mean <= 9.5, drop=True).max().item())
        index = int(np.argmin(np.abs(row_mean.values - target)))
    section = data.isel(y=index)
    longitude = section.lon % 360
    return section.where((longitude >= 180) & (longitude <= 280), drop=True).sortby("lon")


class NotebookSelectionTests(unittest.TestCase):
    def setUp(self):
        self.namespace = notebook_namespace()

    def test_selection_preserves_results_and_stays_lazy(self):
        for curved, descending, calendar in (
            (False, False, "standard"),
            (False, True, "standard"),
            (True, False, "standard"),
            (True, True, "noleap"),
        ):
            with self.subTest(curved=curved, descending=descending, calendar=calendar):
                data = sample_dataset(curved, descending, calendar)
                tasks = []
                with Callback(pretask=lambda key, graph, state: tasks.append(key)):
                    regional = self.namespace["subset_region"](data)
                    actual = self.namespace["extract_section_9_5N"](regional)
                self.assertEqual(tasks, [])
                self.assertIsNotNone(actual.uo.chunks)
                self.assertEqual(actual.sizes["time"], 11)
                self.assertNotIn("y", actual.uo.dims)
                years = data.time.dt.year
                expected = original_section(data, self.namespace).isel(
                    time=np.flatnonzero((years >= 2000) & (years <= 2010))
                )
                xr.testing.assert_allclose(actual.compute(), expected.compute())

    def test_curvilinear_out_of_region_cells_remain_masked(self):
        data = sample_dataset(curvilinear=True)
        latitude = data.lat.values.copy()
        latitude[3, 2] = 25
        latitude[3, 3] = -15
        data = data.assign_coords(lat=(("y", "x"), latitude))
        actual = self.namespace["extract_section_9_5N"](
            self.namespace["subset_region"](data)
        )
        expected = original_section(data, self.namespace).isel(time=slice(1, 12))
        xr.testing.assert_allclose(actual.compute(), expected.compute())
        self.assertTrue(bool(actual.uo.sel(x=2).isnull().all().compute()))

    def test_only_intersecting_source_chunks_are_read(self):
        reads = []
        time = xr.date_range("1999-01-01", periods=14, freq="YS")

        @dask.delayed
        def read_row(year_index, row_index):
            reads.append((year_index, row_index))
            return np.full((1, 3, 1, 6), year_index + row_index, dtype=float)

        values = da.concatenate([
            da.concatenate([
                da.from_delayed(read_row(t, y), shape=(1, 3, 1, 6), dtype=float)
                for y in range(7)
            ], axis=2)
            for t in range(14)
        ], axis=0)
        data = sample_dataset().assign_coords(time=time)
        data["uo"] = (("time", "lev", "y", "x"), values)
        section = self.namespace["extract_section_9_5N"](
            self.namespace["subset_region"](data)
        )
        self.assertEqual(reads, [])
        section.mean("time").compute(scheduler="single-threaded")
        self.assertEqual(set(reads), {(t, 3) for t in range(1, 12)})
        self.assertEqual(len(reads), 11)

    def test_selected_pipeline_executes_fewer_tasks(self):
        data = sample_dataset(curvilinear=True)
        original = original_section(data, self.namespace).isel(time=slice(1, 12)).mean("time")
        selected = self.namespace["extract_section_9_5N"](
            self.namespace["subset_region"](data)
        ).mean("time")
        counts = []
        for section in (original, selected):
            tasks = []
            with Callback(pretask=lambda key, graph, state: tasks.append(key)):
                section.compute(scheduler="single-threaded")
            counts.append(len(tasks))
        self.assertLess(counts[1], counts[0] / 2)

    def test_section_profile_pipeline_and_loading_reuse(self):
        data = sample_dataset(curvilinear=True)
        ns = self.namespace
        ns.update(
            model_datasets={"test-model": ns["subset_region"](data)},
            pd=pd, display=lambda value: None, ProgressBar=ProgressBar, tqdm=tqdm,
        )
        with redirect_stdout(io.StringIO()):
            exec(CELLS["aaaa68db"], ns)
        self.assertEqual(ns["section_failures"], {})
        section = ns["model_sections"]["test-model"]
        expected = original_section(data, ns).isel(time=slice(1, 12)).mean(
            "time", keep_attrs=True
        )
        expected = expected.assign_coords(lev=expected.lev / 100)
        xr.testing.assert_allclose(section, expected)
        self.assertEqual(section.lev.attrs["units"], "m")
        longitude = expected.lon % 360
        expected_profile = expected.uo.where(
            (longitude >= 201) & (longitude <= 255.75), drop=True
        ).mean("x", keep_attrs=True).rename("uo_mean_profile")
        xr.testing.assert_allclose(ns["zonal_velocity_profiles"]["test-model"], expected_profile)
        with redirect_stdout(io.StringIO()):
            exec(CELLS["957fc521"], ns)
        self.assertIsNone(ns["model_sections"]["test-model"].uo.chunks)
        self.assertIsNone(ns["zonal_velocity_profiles"]["test-model"].chunks)
        with patch.object(xr.Dataset, "compute", side_effect=AssertionError("Repeated read")):
            with redirect_stdout(io.StringIO()):
                exec(CELLS["957fc521"], ns)

    def test_interruption_preserves_completed_models(self):
        ns = self.namespace
        section = ns["extract_section_9_5N"](ns["subset_region"](sample_dataset())).mean("time")
        ns.update(
            model_sections={"first": section, "second": section},
            zonal_velocity_profiles={},
            ProgressBar=ProgressBar, tqdm=tqdm,
        )
        loaded = section.compute()
        with patch.object(xr.Dataset, "compute", side_effect=[loaded, KeyboardInterrupt]):
            with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                exec(CELLS["957fc521"], ns)
        self.assertIsNone(ns["model_sections"]["first"].uo.chunks)
        self.assertIsNone(ns["zonal_velocity_profiles"]["first"].chunks)
        self.assertIsNotNone(ns["model_sections"]["second"].uo.chunks)
        with patch.object(xr.Dataset, "compute", return_value=loaded) as compute:
            with redirect_stdout(io.StringIO()):
                exec(CELLS["957fc521"], ns)
            self.assertEqual(compute.call_count, 1)

    def test_invalid_intervals_report_errors(self):
        ns = self.namespace
        ns["TIME_START"] = 2050
        ns["TIME_END"] = 2060
        with self.assertRaisesRegex(ValueError, "No data"):
            ns["subset_region"](sample_dataset())
        ns = notebook_namespace()
        ns["LATITUDE_MIN"] = 80
        ns["LATITUDE_MAX"] = 90
        with self.assertRaisesRegex(ValueError, "No grid cells"):
            ns["subset_region"](sample_dataset())

    def test_loader_requests_native_chunks(self):
        ns = self.namespace
        calls = {}
        data = sample_dataset()

        class Catalog:
            def search(self, **query):
                calls["query"] = query
                return self

            def to_dataset_dict(self, **kwargs):
                calls.update(kwargs)
                return {"test": data}

        ns["cmip6"] = Catalog()
        plan = pd.Series(dict(source_id="test", member_id="r1i1p1f1", uo_grid_label="gn"))
        actual = ns["open_uo"](plan)
        self.assertEqual(calls["xarray_open_kwargs"], {"chunks": {}, "consolidated": True})
        self.assertEqual(calls["query"]["variable_id"], "uo")
        xr.testing.assert_identical(actual, data)


if __name__ == "__main__":
    unittest.main()
