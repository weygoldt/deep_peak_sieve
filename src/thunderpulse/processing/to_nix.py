import pathlib

import neo.io as neo
import nixio
import numpy as np
import polars as pl
import quantities as pq
from IPython import embed
from probeinterface.io import read_probeinterface
from rich.progress import track


def write_nix_file(
    path: pathlib.Path, savepath: pathlib.Path, probe_path: pathlib.Path, bt_overwrite
):
    try:
        dataset = neo.OpenEphysBinaryIO(path).read(lazy=True)
    except FileNotFoundError:
        return ""

    data_array = dataset[0].segments[0].analogsignals[0]

    nix_file_path = savepath / "spikesorter.nix"

    if nix_file_path.exists() and nix_file_path.is_file() and not bt_overwrite:
        return nix_file_path

    nix_file = nixio.File.open(str(nix_file_path), nixio.FileMode.Overwrite)
    block = nix_file.create_block(name="raw_data", type_="neo.dataset")

    try:
        probe = read_probeinterface(probe_path)
    except ValueError:
        return None

    section = nix_file.create_section("recording", type_="recording_metadata")
    probe_section = section.create_section("probe", type_="probeinterface.Probe")
    section.create_property(
        "samplerate",
        [str(data_array.sampling_rate.magnitude), str(data_array.sampling_rate.units)],
    )
    section.create_property("channels", str(data_array.shape[1]))

    # NOTE: adding probeframe to nixfile
    probe_dict = pl.DataFrame(probe.to_numpy())
    block.create_data_frame(
        "probe_frame",
        "probeinterface.probe",
        col_names=probe_dict.columns,
        data=probe_dict.to_numpy(),
    )

    probe_dict = probe.to_dict()
    write_keys = ["ndim", "si_units", "annotations"]
    for p in probe_dict["probes"]:
        for k in write_keys:
            probe_value = p[k]

            if isinstance(probe_value, dict):
                subdict = probe_section.create_section(k, type_="probeinterface.Probe")
                for subkey in probe_value:
                    subdict.create_property(subkey, probe_value[subkey])
            else:
                probe_section.create_property(k, probe_value)

    nix_data_array = block.create_data_array(
        name="data",
        array_type="neo.data_array",
        shape=data_array.shape,
        dtype=nixio.DataType.Int16,
    )
    chunk_size = 80 * pq.s
    start_data_array = data_array.t_start
    chunked_data = np.arange(data_array.t_start, data_array.t_stop, chunk_size) * pq.s

    for start, stop in track(
        zip(chunked_data[:-1], chunked_data[1:]),
        description="Saving Raw Data",
        total=len(chunked_data),
    ):
        data_chunk = data_array.load(time_slice=(start, stop))
        start_index = np.round(
            (start.magnitude - start_data_array.magnitude)
            * data_array.sampling_rate.magnitude,
            0,
        ).astype(int)
        stop_index = np.round(
            (stop.magnitude - start_data_array.magnitude)
            * data_array.sampling_rate.magnitude,
            0,
        ).astype(int)

        nix_data_array[start_index:stop_index] = data_chunk.magnitude

    last_chunk_size = data_array.t_stop - chunked_data[-1]
    data_chunk = data_array.load(
        time_slice=(data_array.t_stop - last_chunk_size, data_array.t_stop)
    )
    last_chunk_index = np.round(
        last_chunk_size.magnitude * data_array.sampling_rate.magnitude, 0
    ).astype(int)

    nix_data_array[-last_chunk_index:] = data_chunk
    # print(f"Processing Last Chunk {i} / {chunked_data.shape[0] - 1}")

    assert nix_data_array.shape == data_array.shape, (
        "nix data array does not match neo data array"
    )

    nix_file.close()
    return nix_file_path
